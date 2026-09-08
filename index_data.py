#!/usr/bin/env python3
"""Chunkar anförandena i data/, skapar embeddings på GPU och sparar i ChromaDB.

Körs så här:
    python index_data.py --limit 500     # testkörning
    python index_data.py                 # allt
"""

import argparse
import glob
import json
import re
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MODELL = "intfloat/multilingual-e5-large"

# Modellen klipper tyst allt över 512 tokens. Svenska ger ca 1,44 tokens per ord
# (1,60 vid p95), så 250 ord landar på ~360 tokens och ryms med marginal.
MAX_ORD = 250
OVERLAPP_ORD = 60

# Kortare än så här är i praktiken bara "Fru talman! Ja." - de kan aldrig bli
# användbara sökträffar men kan däremot tränga undan riktiga svar.
MIN_ORD = 15


def stycken_ur_html(rahtml):
    """Plockar ut styckena ur anförandets HTML.

    Vi använder <p>-taggarna som brytpunkter i stället för att klippa blint
    på teckenantal. Då hamnar aldrig ett snitt mitt i en mening.
    """
    stycken = []
    for bit in re.findall(r"<p>(.*?)</p>", rahtml, re.S):
        text = re.sub(r"<[^>]+>", " ", bit)
        text = re.sub(r"\s+", " ", text).strip()
        if text and "STYLEREF" not in text:
            stycken.append(text)
    return stycken


def dela_langt_stycke(stycke, max_ord):
    """Delar ett ovanligt långt stycke på meningsgräns.

    Behövs för 2 stycken av 707 155, men utan det skulle de bli chunks
    som överskrider modellens tokentak och tyst klipps.
    """
    delar = []
    for mening in re.split(r"(?<=[.!?])\s+", stycke):
        ord_lista = mening.split()
        if len(ord_lista) <= max_ord:
            delar.append(mening)
        else:
            # Enstaka anföranden är skrivna som rundmeningar på flera hundra
            # ord utan en enda punkt. Då finns ingen meningsgräns att klippa
            # på, och vi får ta ordgränsen i stället för att tyst trunkeras.
            for i in range(0, len(ord_lista), max_ord):
                delar.append(" ".join(ord_lista[i:i + max_ord]))

    bitar, nuvarande, antal = [], [], 0
    for mening in delar:
        n = len(mening.split())
        if nuvarande and antal + n > max_ord:
            bitar.append(" ".join(nuvarande))
            nuvarande, antal = [], 0
        nuvarande.append(mening)
        antal += n
    if nuvarande:
        bitar.append(" ".join(nuvarande))
    return bitar


def dela_i_chunks(stycken, max_ord=MAX_ORD, overlapp_ord=OVERLAPP_ORD):
    """Packar stycken till chunks på högst max_ord ord.

    Ett anförande som ryms under gränsen blir exakt en chunk - det gäller
    ungefär 70 procent av dem. Längre anföranden delas, och varje ny chunk
    inleds med föregående stycke som överlapp, så att ett resonemang som
    sträcker sig över en styckegräns inte kapas mitt itu.
    """
    # Alla stycken måste rymmas i en chunk, annars spricker budgeten nedan.
    normaliserade = []
    for stycke in stycken:
        if len(stycke.split()) > max_ord:
            normaliserade.extend(dela_langt_stycke(stycke, max_ord))
        else:
            normaliserade.append(stycke)

    chunks, nuvarande, antal = [], [], 0
    for stycke in normaliserade:
        n = len(stycke.split())
        if nuvarande and antal + n > max_ord:
            chunks.append(" ".join(nuvarande))
            # Ta med förra stycket in i nästa chunk, men bara om det är kort
            # nog att inte äta upp budgeten.
            sista = nuvarande[-1]
            if len(sista.split()) <= overlapp_ord and len(sista.split()) + n <= max_ord:
                nuvarande, antal = [sista], len(sista.split())
            else:
                nuvarande, antal = [], 0
        nuvarande.append(stycke)
        antal += n
    if nuvarande:
        chunks.append(" ".join(nuvarande))
    return chunks


def bygg_chunkar(anforande):
    """Gör om ett anförande till en lista av (id, text, metadata)."""
    stycken = stycken_ur_html(anforande["anforandetext_html"])
    if not stycken:
        return []

    bitar = dela_i_chunks(stycken)
    poster = []
    for i, text in enumerate(bitar):
        # Metadatan följer med varje chunk, så att filtrering på t.ex. parti
        # fungerar oavsett vilken chunk i ett långt tal som råkar ge träff.
        poster.append((
            f"{anforande['anforande_id']}:{i}",
            text,
            {
                "anforande_id": anforande["anforande_id"],
                "talare": anforande["talare"],
                "parti": anforande["parti"],
                "datum": anforande["datum"],
                "ar": int(anforande["datum"][:4]) if anforande["datum"] else 0,
                "debattrubrik": anforande["debattrubrik"],
                "kammaraktivitet": anforande["kammaraktivitet"],
                "riksmote": anforande["riksmote"],
                "dok_id": anforande["dok_id"],
                "chunk_nr": i,
                "antal_chunkar": len(bitar),
                "antal_ord": len(text.split()),
                "url": anforande["url"],
            },
        ))
    return poster


def batcha_hela_anforanden(per_anforande, max_storlek):
    """Packar chunks till batchar utan att någonsin klyva ett anförande.

    Återupptagningen ser ett anförande som klart så snart någon av dess chunks
    finns i databasen. Det stämmer bara om alla chunks för ett anförande skrivs
    i samma databasoperation - annars kan ett avbrott mitt i en fil lämna ett
    anförande halvindexerat, och då hoppas resten över för alltid.

    En batch kan därför bli något större än max_storlek, men aldrig delad mitt
    i ett tal.
    """
    batch = []
    for chunkar in per_anforande:
        if batch and len(batch) + len(chunkar) > max_storlek:
            yield batch
            batch = []
        batch.extend(chunkar)
    if batch:
        yield batch


def las_anforanden(fil):
    """Läser en JSONL-fil och returnerar en lista med anföranden."""
    anforanden = []
    with open(fil, encoding="utf-8") as f:
        for rad in f:
            try:
                anforanden.append(json.loads(rad))
            except json.JSONDecodeError:
                pass
    return anforanden


def main():
    parser = argparse.ArgumentParser(description="Indexerar anförandena i ChromaDB.")
    parser.add_argument("--limit", type=int,
                        help="Indexera som mest så här många anföranden (för test).")
    parser.add_argument("--data-dir", default="data", help="Var JSONL-filerna ligger.")
    parser.add_argument("--chroma-dir", default="chroma_db", help="Var databasen sparas.")
    parser.add_argument("--collection", default="anforanden", help="Namn på samlingen.")
    parser.add_argument("--model", default=MODELL, help="Embeddingmodell.")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Antal chunks per GPU-batch.")
    parser.add_argument("--reset", action="store_true",
                        help="Radera samlingen och börja om från början.")
    args = parser.parse_args()

    filer = sorted(glob.glob(str(Path(args.data_dir) / "anforanden_*.jsonl")))
    if not filer:
        raise SystemExit(f"Hittade inga anforanden_*.jsonl i {args.data_dir}/. Kör fetch_data.py först.")

    klient = chromadb.PersistentClient(path=args.chroma_dir)
    if args.reset:
        try:
            klient.delete_collection(args.collection)
            print(f"Raderade samlingen {args.collection}.")
        except Exception:
            pass

    # cosine passar normaliserade embeddings, som är vad modellen ger oss.
    samling = klient.get_or_create_collection(args.collection,
                                              metadata={"hnsw:space": "cosine"})

    # Återupptagning: vilka anföranden ligger redan i databasen? Chunk-id:t är
    # "<anforande_id>:<nr>", så vi kan läsa av vilka tal som är klara.
    befintliga = samling.get(include=[])["ids"]
    klara = {chunk_id.rsplit(":", 1)[0] for chunk_id in befintliga}
    if klara:
        print(f"Hittade {len(befintliga)} chunks från {len(klara)} anföranden sedan tidigare.")

    print(f"Laddar {args.model} ...")
    modell = SentenceTransformer(args.model, device="cuda")
    modell.half()  # halv precision, dubbelt så snabbt och gott nog för sökning

    kvar = args.limit
    nya_anforanden = nya_chunkar = hoppade_korta = 0

    for fil in filer:
        if kvar is not None and kvar <= 0:
            break

        anforanden = las_anforanden(fil)
        att_gora = []
        for a in anforanden:
            if a["anforande_id"] in klara:
                continue
            if len(a["text"].split()) < MIN_ORD:
                hoppade_korta += 1
                continue
            att_gora.append(a)

        if kvar is not None:
            att_gora = att_gora[:kvar]

        print(f"\n{Path(fil).name}: {len(anforanden)} anföranden, {len(att_gora)} att indexera.")
        if not att_gora:
            continue

        # Vi bygger alla chunks för filen först, och embeddar dem sedan i
        # batchar. GPU:n är snabbast när den får många texter på en gång.
        per_anforande = [bygg_chunkar(a) for a in att_gora]
        batchar = list(batcha_hela_anforanden(per_anforande, args.batch_size))
        antal_chunkar_i_filen = sum(len(b) for b in batchar)

        for grupp in tqdm(batchar, unit="batch"):
            ider = [p[0] for p in grupp]
            texter = [p[1] for p in grupp]
            metadata = [p[2] for p in grupp]

            # E5 kräver prefixet "passage:" på det som indexeras. Frågor ska
            # senare embeddas med "query:" i stället. Prefixet är bara till för
            # modellen, så vi sparar texten utan det.
            vektorer = modell.encode(
                [f"passage: {t}" for t in texter],
                batch_size=args.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            samling.upsert(ids=ider, documents=texter, metadatas=metadata,
                           embeddings=vektorer.tolist())

        nya_anforanden += len(att_gora)
        nya_chunkar += antal_chunkar_i_filen
        if kvar is not None:
            kvar -= len(att_gora)

    print(f"\nKlart. {nya_chunkar} chunks från {nya_anforanden} anföranden.")
    if hoppade_korta:
        print(f"Hoppade över {hoppade_korta} anföranden kortare än {MIN_ORD} ord.")
    print(f"Samlingen innehåller nu {samling.count()} chunks i {args.chroma_dir}/")


if __name__ == "__main__":
    main()
