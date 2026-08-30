#!/usr/bin/env python3
"""Visar statistik över anförandena i data/.

Körs så här:
    python stats.py
"""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

# Fältet "parti" innehåller inte bara partier, utan även talmansroller.
# De räknas separat så att partisiffrorna blir rättvisande.
PARTINAMN = {
    "S": "Socialdemokraterna",
    "M": "Moderaterna",
    "SD": "Sverigedemokraterna",
    "C": "Centerpartiet",
    "V": "Vänsterpartiet",
    "KD": "Kristdemokraterna",
    "L": "Liberalerna",
    "MP": "Miljöpartiet",
}


def las_anforanden(datakatalog):
    """Läser alla data/anforanden_*.jsonl och returnerar en lista med dictar."""
    filer = sorted(datakatalog.glob("anforanden_*.jsonl"))
    if not filer:
        raise SystemExit(f"Hittade inga anforanden_*.jsonl i {datakatalog}/. Kör fetch_data.py först.")

    anforanden = []
    for fil in filer:
        with fil.open(encoding="utf-8") as f:
            for rad in f:
                try:
                    anforanden.append(json.loads(rad))
                except json.JSONDecodeError:
                    pass  # halvskriven sista rad från ett avbrott
    return anforanden


def rubrik(text):
    print(f"\n{text}")
    print("-" * len(text))


def visa_oversikt(anforanden):
    datum = [a["datum"] for a in anforanden if a["datum"]]
    tomma = sum(1 for a in anforanden if not a["text"])

    rubrik("Översikt")
    print(f"Antal anföranden : {len(anforanden):>8,}".replace(",", " "))
    print(f"Datumspann       : {min(datum)} till {max(datum)}")
    print(f"Utan text        : {tomma:>8,}".replace(",", " "))


def visa_per_parti(anforanden):
    antal = Counter()
    ord_per_parti = defaultdict(list)
    for a in anforanden:
        antal[a["parti"]] += 1
        ord_per_parti[a["parti"]].append(len(a["text"].split()))

    partier = [p for p in antal if p in PARTINAMN]
    ovriga = [p for p in antal if p not in PARTINAMN]
    totalt_partier = sum(antal[p] for p in partier)

    rubrik("Anföranden per parti")
    print(f"{'Parti':<22}{'Antal':>8}{'Andel':>8}{'Snitt ord':>11}")
    for p in sorted(partier, key=lambda p: -antal[p]):
        andel = 100 * antal[p] / totalt_partier
        snitt = statistics.mean(ord_per_parti[p])
        print(f"{PARTINAMN[p]:<22}{antal[p]:>8}{andel:>7.1f}%{snitt:>11.0f}")

    rubrik("Övriga (talmän, statschef, partilösa)")
    for p in sorted(ovriga, key=lambda p: -antal[p]):
        print(f"{(p or '(tomt)').title():<32}{antal[p]:>8}")


def visa_per_ar(anforanden):
    antal = Counter()
    ord_per_ar = defaultdict(list)
    for a in anforanden:
        if a["datum"]:
            ar = a["datum"][:4]
            antal[ar] += 1
            ord_per_ar[ar].append(len(a["text"].split()))

    rubrik("Anföranden per år")
    print(f"{'År':<8}{'Antal':>8}{'Snitt ord':>11}   {'':<20}")
    hogst = max(antal.values())
    for ar in sorted(antal):
        snitt = statistics.mean(ord_per_ar[ar])
        stapel = "█" * round(20 * antal[ar] / hogst)
        print(f"{ar:<8}{antal[ar]:>8}{snitt:>11.0f}   {stapel}")


def visa_langd(anforanden):
    ord_antal = [len(a["text"].split()) for a in anforanden]
    tecken = [len(a["text"]) for a in anforanden]

    rubrik("Längd")
    print(f"{'':<12}{'Ord':>10}{'Tecken':>10}")
    print(f"{'Medel':<12}{statistics.mean(ord_antal):>10.0f}{statistics.mean(tecken):>10.0f}")
    print(f"{'Median':<12}{statistics.median(ord_antal):>10.0f}{statistics.median(tecken):>10.0f}")
    print(f"{'Kortaste':<12}{min(ord_antal):>10}{min(tecken):>10}")
    print(f"{'Längsta':<12}{max(ord_antal):>10}{max(tecken):>10}")


def visa_ytterligheter(anforanden, antal=5):
    """De längsta och kortaste anförandena. Tomma texter räknas inte som kortast."""
    med_text = [a for a in anforanden if a["text"]]
    sorterade = sorted(med_text, key=lambda a: len(a["text"].split()))

    for titel, urval in [
        (f"De {antal} längsta anförandena", sorterade[-antal:][::-1]),
        (f"De {antal} kortaste anförandena (med text)", sorterade[:antal]),
    ]:
        rubrik(titel)
        for a in urval:
            print(f"{len(a['text'].split()):>6} ord  {a['datum']}  {a['talare']}")
            print(f"{'':>12}{a['debattrubrik'][:70]}")


def main():
    parser = argparse.ArgumentParser(description="Statistik över hämtade anföranden.")
    parser.add_argument("--data-dir", default="data", help="Var JSONL-filerna ligger.")
    args = parser.parse_args()

    anforanden = las_anforanden(Path(args.data_dir))
    visa_oversikt(anforanden)
    visa_per_parti(anforanden)
    visa_per_ar(anforanden)
    visa_langd(anforanden)
    visa_ytterligheter(anforanden)
    print()


if __name__ == "__main__":
    main()
