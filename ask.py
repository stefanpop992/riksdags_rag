#!/usr/bin/env python3
"""Terminalversionen: ställ frågor om riksdagens anföranden och få svar med källor.

All RAG-logik ligger i rag.py, som appen (app.py) använder likadant.

Körs så här:
    python ask.py "Vad har V sagt om vinster i välfärden?"
    python ask.py                      # interaktivt läge
    python ask.py "..." --parti SD --fran-ar 2024
"""

import argparse
import sys
from pathlib import Path

import rag


def visa_traffar(traffar):
    """Skriver ut vad sökningen hittade, så att svaret går att granska."""
    print(f"\nHittade {len(traffar)} utdrag:\n")
    for i, t in enumerate(traffar, 1):
        m = t["meta"]
        print(f"  [{i}] {t['likhet']:.3f}  {rag.formatera_talare(m)}  {m['datum']}")
        print(f"      {m['debattrubrik'][:70]}")
        print(f"      {t['text'][:100].strip()}...")
    print()


def visa_steg(steg):
    """Skriver ut agentens arbete medan loopen kör."""
    if steg["typ"] == "plan":
        print(f"  [plan] {len(steg['delfragor'])} delfrågor:", file=sys.stderr)
        for d in steg["delfragor"]:
            parti = f" [{d.parti}]" if d.parti else ""
            print(f"         {d.sokfraga}{parti}", file=sys.stderr)
    elif steg["typ"] == "sokning":
        print(f"  [sök {steg['varv']}] {steg['nya']} nya utdrag, "
              f"{steg['totalt']} totalt", file=sys.stderr)
    elif steg["typ"] == "bedomning":
        besked = "räcker" if steg["racker"] else "räcker inte"
        print(f"  [bedöm {steg['varv']}] {besked}", file=sys.stderr)
        for s_ in steg["saknas"]:
            print(f"         saknas: {s_}", file=sys.stderr)
    elif steg["typ"] == "stopp":
        print(f"  [stopp] {steg['skal']}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Frågar Claude om riksdagens anföranden.")
    parser.add_argument("fraga", nargs="?", help="Din fråga. Utelämnas för interaktivt läge.")
    parser.add_argument("-n", "--antal", type=int, default=8, help="Antal utdrag att hämta.")
    parser.add_argument("--parti", help="Filtrera på parti, t.ex. S eller SD.")
    parser.add_argument("--fran-ar", type=int, help="Bara anföranden från och med detta år.")
    parser.add_argument("--till-ar", type=int, help="Bara anföranden till och med detta år.")
    parser.add_argument("--chroma-dir", default="chroma_db", help="Var databasen ligger.")
    parser.add_argument("--data-dir", default="data", help="Var talarregistret byggs från.")
    parser.add_argument("--collection", default="anforanden", help="Namn på samlingen.")
    parser.add_argument("--model", default=rag.SVARSMODELL, help="Vilken Claude-modell som svarar.")
    parser.add_argument("--planeringsmodell", default=rag.PLANERINGSMODELL,
                        help="Modell som planerar och granskar i agentiskt läge.")
    parser.add_argument("--per-anforande", type=int, default=1,
                        help="Högst så här många utdrag från samma anförande.")
    parser.add_argument("--talare", help="Filtrera på talare, t.ex. \"Kristersson\".")
    parser.add_argument("--agentisk", action="store_true",
                        help="Låt modellen planera, granska och söka om (flera API-anrop).")
    parser.add_argument("--max-varv", type=int, default=3,
                        help="Högsta antal sökvarv i agentiskt läge.")
    parser.add_argument("--visa-prompt", action="store_true", help="Skriv ut systemprompten och avsluta.")
    args = parser.parse_args()

    if args.visa_prompt:
        print(rag.SYSTEMPROMPT)
        return

    try:
        klient = rag.skapa_klient()
    except RuntimeError as fel:
        raise SystemExit(str(fel))

    samling = rag.oppna_samling(args.chroma_dir, args.collection)
    print(f"Databasen innehåller {samling.count()} utdrag. Laddar sökmodellen ...", file=sys.stderr)
    modell = rag.ladda_modell()

    alla_talare = rag.las_talare(str(Path(args.data_dir)))
    varianter = rag.talarvarianter(args.talare, alla_talare)
    if args.talare and not varianter:
        raise SystemExit(f"Hittade ingen talare som matchar {args.talare!r}.")
    where = rag.bygg_filter(args.parti, args.fran_ar, args.till_ar, varianter)
    if where:
        print(f"Filter: {where}", file=sys.stderr)

    fragor = [args.fraga] if args.fraga else None
    while True:
        if fragor:
            fraga = fragor.pop(0)
        else:
            try:
                fraga = input("\nFråga (tom rad avslutar): ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if not fraga:
                return

        if args.agentisk:
            traffar, saknas, _ = rag.agentisk_sokning(
                klient, samling, modell, fraga,
                anvandar_parti=args.parti, fran_ar=args.fran_ar, till_ar=args.till_ar,
                max_varv=args.max_varv, per_anforande=args.per_anforande,
                planeringsmodell=args.planeringsmodell, pa_steg=visa_steg,
                alla_talare=alla_talare, anvandar_talare=args.talare)
            underlag = rag.bygg_underlag_med_luckor(traffar, saknas)
        else:
            traffar = rag.sok(samling, modell, fraga, args.antal, where, args.per_anforande)
            underlag = rag.bygg_underlag(traffar)

        if not traffar:
            print("Sökningen gav inga träffar. Prova en bredare fråga eller ta bort filtren.")
        else:
            visa_traffar(traffar)

            # Tänkandet går till stderr, svaret till stdout. Då kan du fortfarande
            # pipa svaret vidare till en fil utan att resonemanget följer med.
            skrivit_rubrik = []

            def visa_tanke(bit):
                if not skrivit_rubrik:
                    print("  [tänker] ", end="", file=sys.stderr, flush=True)
                    skrivit_rubrik.append(True)
                print(bit, end="", file=sys.stderr, flush=True)

            print("Svar:\n")
            for bit in rag.stromma_svar(klient, fraga, underlag, args.model, visa_tanke):
                if skrivit_rubrik:
                    print(file=sys.stderr)
                    skrivit_rubrik.clear()
                print(bit, end="", flush=True)
            print("\n")

        if fragor is not None and not fragor:
            return


if __name__ == "__main__":
    main()
