#!/usr/bin/env python3
"""Hämtar anföranden (tal i kammaren) från riksdagens öppna API till data/.

Körs så här:
    python fetch_data.py --limit 100     # testkörning
    python fetch_data.py                 # allt, ~104 000 anföranden
"""

import argparse
import html
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from tqdm import tqdm

API = "https://data.riksdagen.se"

# Ett "riksmöte" är riksdagens arbetsår, ungefär september till juni.
# 2018/19 börjar i september 2018 = starten på mandatperioden 2018-2022.
RIKSMOTEN = [
    "2018/19", "2019/20", "2020/21", "2021/22",
    "2022/23", "2023/24", "2024/25", "2025/26",
]


def filnamn(riksmote):
    """2022/23 -> 2022_23, så det funkar som filnamn."""
    return riksmote.replace("/", "_")


def rensa_html(rahtml):
    """Gör om API:ets HTML till ren löptext."""
    if not rahtml:
        return ""
    # Riksdagens Word-export lämnar kvar fältkoder i egna stycken,
    # t.ex. "<p> STYLEREF Kantrubrik \* MERGEFORMAT Svar på interpellationer</p>".
    # De är inte något någon sagt, så hela stycket åker bort.
    text = re.sub(r"<p>[^<]*STYLEREF.*?</p>", " ", rahtml, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def hamta_json(session, url, forsok=3):
    """Hämtar en URL och returnerar JSON. Försöker igen vid tillfälliga fel."""
    for n in range(forsok):
        try:
            svar = session.get(url, timeout=60)
            svar.raise_for_status()
            return svar.json()
        except Exception as fel:
            if n == forsok - 1:
                print(f"\nGav upp på {url}: {fel}", file=sys.stderr)
                return None
            time.sleep(2 ** n)  # 1s, 2s, 4s ...


def hamta_lista(session, riksmote, listkatalog, uppdatera=False):
    """Hämtar metadata för alla anföranden i ett riksmöte.

    Listan innehåller INTE själva talet, bara vem som talade och var.
    Vi sparar den på disk så att en omstart slipper ladda ner 14 MB igen.
    """
    cache = listkatalog / f"anforandelista_{filnamn(riksmote)}.json"
    if cache.exists() and not uppdatera:
        return json.loads(cache.read_text(encoding="utf-8"))

    # sz = hur många träffar vi vill ha. API:et har ingen sidnumrering,
    # så vi tar hela riksmötet i ett svar och låter rm vara vår "sida".
    url = f"{API}/anforandelista/?rm={riksmote}&sz=20000&utformat=json"
    data = hamta_json(session, url)
    if data is None:
        return []

    rader = data.get("anforandelista", {}).get("anforande", [])
    if isinstance(rader, dict):  # API:et hoppar över listan vid exakt en träff
        rader = [rader]

    cache.write_text(json.dumps(rader, ensure_ascii=False), encoding="utf-8")
    return rader


def las_klara_ids(sokvag):
    """Läser vilka anföranden som redan finns sparade, för återupptagning.

    Sista raden kan vara halvskriven om förra körningen dödades mitt i.
    Då slänger vi den raden och skriver om filen utan den.
    """
    if not sokvag.exists():
        return set()

    klara, hela_rader, trasiga = set(), [], 0
    with sokvag.open(encoding="utf-8") as f:
        for rad in f:
            try:
                klara.add(json.loads(rad)["anforande_id"])
                hela_rader.append(rad)
            except (json.JSONDecodeError, KeyError):
                trasiga += 1

    if trasiga:
        sokvag.write_text("".join(hela_rader), encoding="utf-8")
        print(f"  Städade bort {trasiga} trasig(a) rad(er) från förra körningen.")
    return klara


def hamta_anforande(session, rad):
    """Hämtar fulltexten för ett anförande och plockar ut fälten vi vill ha."""
    url = rad.get("anforande_url_xml") or f"{API}/anforande/{rad['dok_id']}-{rad['anforande_nummer']}"
    data = hamta_json(session, url + "/json")
    if data is None:
        return None

    full = data.get("anforande", {})
    rahtml = full.get("anforandetext", "")
    return {
        "anforande_id": rad["anforande_id"],
        "dok_id": rad.get("dok_id", ""),
        "anforande_nummer": rad.get("anforande_nummer", ""),
        "riksmote": rad.get("dok_rm", ""),
        "datum": (rad.get("dok_datum") or "")[:10],
        "talare": rad.get("talare", ""),
        "parti": rad.get("parti", ""),
        "intressent_id": rad.get("intressent_id", ""),
        "debattrubrik": rad.get("avsnittsrubrik", ""),
        "underrubrik": rad.get("underrubrik", ""),
        "kammaraktivitet": rad.get("kammaraktivitet", ""),
        "replik": rad.get("replik", ""),
        "protokoll_titel": rad.get("dok_titel", ""),
        "text": rensa_html(rahtml),
        "anforandetext_html": rahtml,
        "url": rad.get("protokoll_url_www", ""),
    }


def main():
    parser = argparse.ArgumentParser(description="Hämtar riksdagens anföranden till data/.")
    parser.add_argument("--limit", type=int,
                        help="Hämta som mest så här många anföranden totalt (för test).")
    parser.add_argument("--riksmoten", default=",".join(RIKSMOTEN),
                        help="Kommaseparerad lista, t.ex. 2022/23,2023/24.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Antal parallella nedladdningar (standard 8).")
    parser.add_argument("--data-dir", default="data", help="Var filerna hamnar.")
    parser.add_argument("--uppdatera-listor", action="store_true",
                        help="Hämta om metadatalistorna. Behövs för pågående riksmöte.")
    args = parser.parse_args()

    datakatalog = Path(args.data_dir)
    listkatalog = datakatalog / "listor"
    listkatalog.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    kvar = args.limit
    totalt_nya = 0

    for riksmote in [r.strip() for r in args.riksmoten.split(",") if r.strip()]:
        if kvar is not None and kvar <= 0:
            break

        print(f"\nRiksmöte {riksmote}")
        rader = hamta_lista(session, riksmote, listkatalog, args.uppdatera_listor)
        if not rader:
            print("  Hittade inga anföranden.")
            continue

        utfil = datakatalog / f"anforanden_{filnamn(riksmote)}.jsonl"
        klara = las_klara_ids(utfil)

        att_hamta = [r for r in rader if r["anforande_id"] not in klara]
        if kvar is not None:
            att_hamta = att_hamta[:kvar]

        print(f"  {len(rader)} totalt, {len(klara)} redan klara, {len(att_hamta)} att hämta.")
        if not att_hamta:
            continue

        # Trådarna hämtar parallellt, men bara huvudtråden skriver till filen.
        # Då kan två trådar aldrig skriva halva rader i varandra.
        with utfil.open("a", encoding="utf-8") as f:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                jobb = pool.map(lambda r: hamta_anforande(session, r), att_hamta)
                for anforande in tqdm(jobb, total=len(att_hamta), unit="anf"):
                    if anforande is None:
                        continue
                    f.write(json.dumps(anforande, ensure_ascii=False) + "\n")
                    f.flush()  # på disk direkt, så ett avbrott bara tappar det sista
                    totalt_nya += 1
                    if kvar is not None:
                        kvar -= 1

    print(f"\nKlart. {totalt_nya} nya anföranden sparade i {datakatalog}/")


if __name__ == "__main__":
    main()
