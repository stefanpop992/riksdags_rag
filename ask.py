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


def main():
    parser = argparse.ArgumentParser(description="Frågar Claude om riksdagens anföranden.")
    parser.add_argument("fraga", nargs="?", help="Din fråga. Utelämnas för interaktivt läge.")
    parser.add_argument("-n", "--antal", type=int, default=8, help="Antal utdrag att hämta.")
    parser.add_argument("--parti", help="Filtrera på parti, t.ex. S eller SD.")
    parser.add_argument("--fran-ar", type=int, help="Bara anföranden från och med detta år.")
    parser.add_argument("--till-ar", type=int, help="Bara anföranden till och med detta år.")
    parser.add_argument("--chroma-dir", default="chroma_db", help="Var databasen ligger.")
    parser.add_argument("--collection", default="anforanden", help="Namn på samlingen.")
    parser.add_argument("--model", default=rag.SVARSMODELL, help="Vilken Claude-modell som svarar.")
    parser.add_argument("--per-anforande", type=int, default=1,
                        help="Högst så här många utdrag från samma anförande.")
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

    where = rag.bygg_filter(args.parti, args.fran_ar, args.till_ar)
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

        traffar = rag.sok(samling, modell, fraga, args.antal, where, args.per_anforande)
        if not traffar:
            print("Sökningen gav inga träffar. Prova en bredare fråga eller ta bort filtren.")
        else:
            visa_traffar(traffar)
            print("Svar:\n")
            for bit in rag.stromma_svar(klient, fraga, rag.bygg_underlag(traffar), args.model):
                print(bit, end="", flush=True)
            print("\n")

        if fragor is not None and not fragor:
            return


if __name__ == "__main__":
    main()
