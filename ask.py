#!/usr/bin/env python3
"""Ställer frågor om riksdagens anföranden och låter Claude svara med källor.

Körs så här:
    python ask.py "Vad har V sagt om vinster i välfärden?"
    python ask.py                      # interaktivt läge
    python ask.py "..." --parti SD --fran-ar 2024
"""

import argparse
import os
import re
import sys

import anthropic
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

EMBEDDINGSMODELL = "intfloat/multilingual-e5-large"
SVARSMODELL = "claude-opus-5"

SYSTEMPROMPT = """Du är en researchassistent som svarar på frågor om vad som sagts i Sveriges riksdags kammare.

Du får en fråga och ett antal utdrag ur riksdagsanföranden. Utdragen är ditt enda underlag.

REGLER

1. Använd endast det som står i utdragen. Du har med säkerhet egen kunskap om svensk politik - använd den inte. Går ett påstående inte att belägga med utdragen ska det inte stå i svaret.

2. Ange källa direkt efter varje påstående, på formen [talare (parti), datum]. Till exempel:
   Elpriserna beskrevs som alarmerande i norra Sverige [Jimmie Åkesson (SD), 2025-12-04].
   Ett påstående utan källa är ett fel.

3. Räcker inte underlaget, säg det rakt ut. Skriv att utdragen inte räcker och beskriv kort vad som saknas. Gissa aldrig och fyll aldrig ut med rimliga antaganden. Ett kort svar som konstaterar att underlaget saknas är alltid bättre än ett långt svar som verkar täcka frågan.

4. Skilj på vad någon SADE och vad som ÄR SANT. Utdragen är politiska anföranden, inte fakta. Skriv "Karin Rågsjö hävdade att köerna vuxit" - inte "köerna har vuxit".

5. Säger utdragen emot varandra, redovisa båda sidor med var sin källa i stället för att välja en.

6. Utdragen är hämtade med en sökning och kan innehålla sådant som inte hör till frågan. Använd bara det som faktiskt är relevant, och låt bli resten.

7. Svara på svenska, kort och sakligt. Hoppa över inledande artighetsfraser."""


PARTIER = {"S", "M", "SD", "C", "V", "KD", "L", "MP"}


def formatera_talare(meta):
    """Ger "Namn (parti)" utan dubblering.

    Fältet talare innehåller redan partiet, t.ex. "Ida Gabrielsson (V)", så vi
    tar bort den parentesen innan vi lägger på partiet igen. Talmän och
    statschefen har ingen partibokstav och får bara sitt namn.
    """
    namn = re.sub(r"\s*\([^)]*\)\s*$", "", meta["talare"]).strip()
    parti = (meta.get("parti") or "").strip()
    if parti in PARTIER:
        return f"{namn} ({parti})"
    return namn or parti.title()


def bygg_filter(parti, fran_ar, till_ar):
    """Bygger ChromaDB:s where-villkor av flaggorna. Returnerar None om inget satts."""
    villkor = []
    if parti:
        villkor.append({"parti": parti.upper()})
    if fran_ar:
        villkor.append({"ar": {"$gte": fran_ar}})
    if till_ar:
        villkor.append({"ar": {"$lte": till_ar}})

    if not villkor:
        return None
    if len(villkor) == 1:
        return villkor[0]
    return {"$and": villkor}


def sok(samling, modell, fraga, antal, where):
    """Söker fram de mest relevanta chunksen ur ChromaDB."""
    # E5 kräver prefixet "query:" på frågor, precis som indexeringen använde
    # "passage:" på texterna. Utan det tappar modellen precision.
    vektor = modell.encode([f"query: {fraga}"], normalize_embeddings=True)
    svar = samling.query(
        query_embeddings=vektor.tolist(),
        n_results=antal,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    traffar = []
    for dok, meta, avstand in zip(svar["documents"][0], svar["metadatas"][0], svar["distances"][0]):
        traffar.append({"text": dok, "meta": meta, "likhet": 1 - avstand})
    return traffar


def visa_traffar(traffar):
    """Skriver ut vad sökningen hittade, så att svaret går att granska."""
    print(f"\nHittade {len(traffar)} utdrag:\n")
    for i, t in enumerate(traffar, 1):
        m = t["meta"]
        print(f"  [{i}] {t['likhet']:.3f}  {formatera_talare(m)}  {m['datum']}")
        print(f"      {m['debattrubrik'][:70]}")
        print(f"      {t['text'][:100].strip()}...")
    print()


def bygg_underlag(traffar):
    """Formaterar utdragen till den text som skickas med frågan till Claude."""
    delar = []
    for i, t in enumerate(traffar, 1):
        m = t["meta"]
        delar.append(
            f"[KÄLLA {i}] {formatera_talare(m)}, {m['datum']}\n"
            f"Debatt: {m['debattrubrik']}\n"
            f"{t['text']}"
        )
    return "\n\n".join(delar)


def fraga_claude(klient, modell, fraga, underlag):
    """Skickar fråga och underlag till Claude och strömmar ut svaret."""
    meddelande = (
        f"Fråga: {fraga}\n\n"
        f"Här är utdragen ur riksdagens anföranden:\n\n{underlag}"
    )
    print("Svar:\n")
    with klient.messages.stream(
        model=modell,
        max_tokens=2000,
        temperature=0,  # vi vill ha trogna svar, inte kreativa
        system=SYSTEMPROMPT,
        messages=[{"role": "user", "content": meddelande}],
    ) as strom:
        for bit in strom.text_stream:
            print(bit, end="", flush=True)
    print("\n")


def main():
    parser = argparse.ArgumentParser(description="Frågar Claude om riksdagens anföranden.")
    parser.add_argument("fraga", nargs="?", help="Din fråga. Utelämnas för interaktivt läge.")
    parser.add_argument("-n", "--antal", type=int, default=8, help="Antal utdrag att hämta.")
    parser.add_argument("--parti", help="Filtrera på parti, t.ex. S eller SD.")
    parser.add_argument("--fran-ar", type=int, help="Bara anföranden från och med detta år.")
    parser.add_argument("--till-ar", type=int, help="Bara anföranden till och med detta år.")
    parser.add_argument("--chroma-dir", default="chroma_db", help="Var databasen ligger.")
    parser.add_argument("--collection", default="anforanden", help="Namn på samlingen.")
    parser.add_argument("--model", default=SVARSMODELL, help="Vilken Claude-modell som svarar.")
    parser.add_argument("--visa-prompt", action="store_true", help="Skriv ut systemprompten och avsluta.")
    args = parser.parse_args()

    if args.visa_prompt:
        print(SYSTEMPROMPT)
        return

    load_dotenv()
    nyckel = os.environ.get("ANTHROPIC_API_KEY")
    if not nyckel:
        raise SystemExit(
            "ANTHROPIC_API_KEY saknas.\n"
            "Lägg den i en fil som heter .env i projektmappen:\n"
            "    ANTHROPIC_API_KEY=sk-ant-..."
        )

    samling = chromadb.PersistentClient(path=args.chroma_dir).get_collection(args.collection)
    print(f"Databasen innehåller {samling.count()} utdrag. Laddar sökmodellen ...", file=sys.stderr)
    modell = SentenceTransformer(EMBEDDINGSMODELL, device="cuda")
    modell.half()
    klient = anthropic.Anthropic(api_key=nyckel)

    where = bygg_filter(args.parti, args.fran_ar, args.till_ar)
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

        traffar = sok(samling, modell, fraga, args.antal, where)
        if not traffar:
            print("Sökningen gav inga träffar. Prova en bredare fråga eller ta bort filtren.")
        else:
            visa_traffar(traffar)
            fraga_claude(klient, args.model, fraga, bygg_underlag(traffar))

        if fragor is not None and not fragor:
            return


if __name__ == "__main__":
    main()
