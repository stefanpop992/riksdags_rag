#!/usr/bin/env python3
"""BM25 - klassisk nyckelordssökning som komplement till vektorsökningen.

Vektorsökningen i rag.py hittar det som BETYDER ungefär samma sak. Den är svag
på det motsatta: exakta ord. Ett efternamn, en lagparagraf, "NATO", ett årtal -
sådant vill man matcha bokstavligt, och då är BM25 fortfarande svårslaget.

Körs så här:
    python bm25.py            # bygger indexet från ChromaDB, sparar i data/bm25/
    python bm25.py --test "elpriser norra sverige"

SÅ HÄR FUNGERAR DET
BM25 poängsätter ett dokument mot en sökfråga med tre idéer:
  1. Ju oftare ordet finns i dokumentet, desto bättre (tf) - men med avtagande
     effekt, så att ett ord som upprepas 20 gånger inte ger 20 gånger poängen.
  2. Ju ovanligare ordet är i hela materialet, desto mer betyder en träff (idf).
     "att" säger ingenting, "kärnkraftsreaktor" säger allt.
  3. Långa dokument får en nedskrivning, annars vinner de bara på att vara långa.

Poängen för ett (dokument, ord)-par beror alltså inte på frågan. Därför räknar
vi ut alla vikter EN gång när vi bygger indexet och sparar dem i matrisen. En
sökning blir då bara "plocka ut kolumnerna för frågans ord och summera raderna",
vilket tar millisekunder även över 215 000 chunks.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

INDEXKATALOG = "data/bm25"

# BM25:s två rattar, med de värden som är standard i litteraturen.
# k1 styr hur snabbt nyttan av upprepade ord planar ut, b hur hårt långa
# dokument straffas (b=0 stänger av längdkorrigeringen helt).
K1 = 1.5
B = 0.75

# Samma mönster som CountVectorizer använder som standard: minst två tecken,
# och \w täcker å, ä och ö eftersom vi kör med unicode-flaggan.
ORDMONSTER = re.compile(r"(?u)\b\w\w+\b")


def dela_i_ord(text):
    """Gör om en text till gemena ord. Används både vid bygge och sökning."""
    return ORDMONSTER.findall(text.lower())


class Bm25:
    """Ett färdigbyggt BM25-index som kan sökas i.

    matris - gles matris med förberäknade vikter, en rad per chunk och en
             kolumn per ord i ordlistan. Sparad i CSC-format, som är det
             format där "hämta en hel kolumn" är billigt.
    ordlista - ord -> kolumnnummer
    ider - radnummer -> chunk-id (samma id som i ChromaDB)
    """

    def __init__(self, matris, ordlista, ider):
        self.matris = matris
        self.ordlista = ordlista
        self.ider = ider

    def sok(self, fraga, antal=200):
        """Returnerar [(chunk_id, poäng)] för de bäst matchande chunksen.

        Ord som inte finns i materialet hoppas tyst över. Innehåller frågan
        inga kända ord alls blir svaret en tom lista - då får vektorsökningen
        bära resultatet ensam.
        """
        kolumner = [self.ordlista[o] for o in dela_i_ord(fraga) if o in self.ordlista]
        if not kolumner:
            return []

        # Summan av frågans kolumner ger varje chunk sin poäng. Ett ord som
        # upprepas i frågan räknas en gång per förekomst, vilket är rimligt:
        # skriver man ordet två gånger menar man det.
        poang = np.asarray(self.matris[:, kolumner].sum(axis=1)).ravel()

        antal = min(antal, poang.size)
        # argpartition tar fram de N största utan att sortera hela listan.
        topp = np.argpartition(-poang, antal - 1)[:antal]
        topp = topp[np.argsort(-poang[topp])]
        return [(self.ider[i], float(poang[i])) for i in topp if poang[i] > 0]


def ladda(katalog=INDEXKATALOG):
    """Läser in ett tidigare byggt index från disk."""
    katalog = Path(katalog)
    if not (katalog / "matris.npz").exists():
        raise FileNotFoundError(
            f"Hittade inget BM25-index i {katalog}/. Bygg det med:\n    python bm25.py"
        )
    matris = sparse.load_npz(katalog / "matris.npz").tocsc()
    ordlista = json.loads((katalog / "ordlista.json").read_text(encoding="utf-8"))
    ider = json.loads((katalog / "ider.json").read_text(encoding="utf-8"))
    return Bm25(matris, ordlista, ider)


def las_ur_chroma(chroma_dir="chroma_db", collection="anforanden", batch=20000):
    """Hämtar alla chunks ur ChromaDB, i portioner så minnet räcker.

    Vi läser ur samma databas som vektorsökningen använder i stället för ur
    JSONL-filerna. Då är de två indexen garanterat överens om vilka chunks som
    finns och vilka id de har - annars kan sammanslagningen peka på chunks som
    inte går att slå upp.
    """
    import chromadb

    samling = chromadb.PersistentClient(path=chroma_dir).get_collection(collection)
    totalt = samling.count()
    ider, texter = [], []
    for offset in range(0, totalt, batch):
        svar = samling.get(limit=batch, offset=offset, include=["documents"])
        ider.extend(svar["ids"])
        texter.extend(svar["documents"])
        print(f"  läst {len(ider)} / {totalt} chunks")
    return ider, texter


def bygg(texter, k1=K1, b=B):
    """Räknar fram BM25-vikterna och returnerar (matris, ordlista).

    Resultatet är en matris där värdet på rad d, kolumn t är den poäng chunk d
    får om frågan innehåller ordet t. Sökningen behöver då bara summera.
    """
    print("Räknar ord ...")
    vektoriserare = CountVectorizer(tokenizer=dela_i_ord, lowercase=False, token_pattern=None)
    antal_forekomster = vektoriserare.fit_transform(texter).tocsr()
    N, V = antal_forekomster.shape
    print(f"  {N} chunks, {V} unika ord, {antal_forekomster.nnz} nollskilda värden")

    # Dokumentlängd = antal ord i chunken, och genomsnittet över alla chunks.
    langd = np.asarray(antal_forekomster.sum(axis=1)).ravel()
    medellangd = langd.mean()

    # df = i hur många chunks ordet förekommer. Kolumnräkning på en CSR-matris
    # görs enklast genom att räkna hur ofta varje kolumnindex dyker upp.
    df = np.bincount(antal_forekomster.indices, minlength=V)
    idf = np.log(1 + (N - df + 0.5) / (df + 0.5)).astype(np.float32)

    print("Räknar BM25-vikter ...")
    tf = antal_forekomster.data.astype(np.float32)
    # Vilken rad varje nollskilt värde tillhör: rad i har indptr[i+1]-indptr[i]
    # värden efter varandra i data-arrayen.
    rad = np.repeat(np.arange(N), np.diff(antal_forekomster.indptr))
    vikt = tf * (k1 + 1) / (tf + k1 * (1 - b + b * langd[rad] / medellangd))
    # Multiplicera in idf, som sitter på kolumnen (= ordet).
    vikt *= idf[antal_forekomster.indices]

    matris = sparse.csr_matrix(
        (vikt, antal_forekomster.indices, antal_forekomster.indptr), shape=(N, V)
    )
    return matris.tocsc(), vektoriserare.vocabulary_


def spara(matris, ordlista, ider, katalog=INDEXKATALOG):
    """Sparar indexet på disk."""
    katalog = Path(katalog)
    katalog.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(katalog / "matris.npz", matris)
    # vocabulary_ har numpy-heltal som värden, som json inte kan skriva.
    (katalog / "ordlista.json").write_text(
        json.dumps({o: int(i) for o, i in ordlista.items()}, ensure_ascii=False),
        encoding="utf-8")
    (katalog / "ider.json").write_text(json.dumps(ider), encoding="utf-8")
    storlek = sum(f.stat().st_size for f in katalog.iterdir()) / 1e6
    print(f"Sparade indexet i {katalog}/ ({storlek:.0f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Bygger BM25-index över chunksen i ChromaDB.")
    parser.add_argument("--chroma-dir", default="chroma_db")
    parser.add_argument("--collection", default="anforanden")
    parser.add_argument("--index-dir", default=INDEXKATALOG)
    parser.add_argument("--test", help="Sök i ett redan byggt index i stället för att bygga.")
    args = parser.parse_args()

    if args.test:
        index = ladda(args.index_dir)
        for chunk_id, poang in index.sok(args.test, antal=10):
            print(f"{poang:6.2f}  {chunk_id}")
        return

    print("Läser chunks ur ChromaDB ...")
    ider, texter = las_ur_chroma(args.chroma_dir, args.collection)
    matris, ordlista = bygg(texter)
    spara(matris, ordlista, ider, args.index_dir)


if __name__ == "__main__":
    main()
