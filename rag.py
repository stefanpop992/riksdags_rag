#!/usr/bin/env python3
"""Gemensam RAG-logik för terminalversionen (ask.py) och webbappen (app.py).

Modulen känner varken till terminalen eller Streamlit. Den söker, bygger
underlag och frågar Claude - hur svaret sedan visas bestämmer den som anropar.
"""

import os
import re
from collections import Counter

import anthropic
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

EMBEDDINGSMODELL = "intfloat/multilingual-e5-large"
SVARSMODELL = "claude-opus-5"
PARTIER = ["S", "M", "SD", "C", "V", "KD", "L", "MP"]

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


def oppna_samling(chroma_dir="chroma_db", collection="anforanden"):
    """Öppnar ChromaDB-samlingen."""
    return chromadb.PersistentClient(path=chroma_dir).get_collection(collection)


def ladda_modell(namn=EMBEDDINGSMODELL):
    """Laddar embeddingmodellen på GPU:n i halv precision."""
    modell = SentenceTransformer(namn, device="cuda")
    modell.half()
    return modell


def skapa_klient():
    """Skapar Anthropic-klienten. Kastar med tydligt besked om nyckeln saknas."""
    load_dotenv()
    nyckel = os.environ.get("ANTHROPIC_API_KEY")
    if not nyckel:
        raise RuntimeError(
            "ANTHROPIC_API_KEY saknas. Lägg den i en fil som heter .env "
            "i projektmappen:\n    ANTHROPIC_API_KEY=sk-ant-..."
        )
    return anthropic.Anthropic(api_key=nyckel)


def las_filtervarden(samling):
    """Läser ut vilka partier och årtal som faktiskt finns i databasen.

    Vi frågar med limit=1 per kandidatvärde i stället för att läsa all
    metadata, vilket tar sekunder i stället för minuter.
    """
    partier = [p for p in PARTIER if samling.get(where={"parti": p}, limit=1)["ids"]]
    ar = [a for a in range(2010, 2041) if samling.get(where={"ar": a}, limit=1)["ids"]]
    if not ar:
        return partier, (2018, 2026)
    return partier, (min(ar), max(ar))


def bygg_filter(parti=None, fran_ar=None, till_ar=None):
    """Bygger ChromaDB:s where-villkor. Returnerar None om inget filter satts."""
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


def sok(samling, modell, fraga, antal=8, where=None, per_anforande=1):
    """Söker fram de mest relevanta chunksen ur ChromaDB.

    Eftersom chunks överlappar ligger flera bitar av samma tal nära varandra i
    betydelse, så ett enda anförande kan lägga beslag på flera platser i
    träfflistan. Vi hämtar därför fler kandidater än vi behöver och behåller
    högst per_anforande stycken per tal.
    """
    # E5 kräver prefixet "query:" på frågor, precis som indexeringen använde
    # "passage:" på texterna. Utan det tappar modellen precision.
    vektor = modell.encode([f"query: {fraga}"], normalize_embeddings=True)
    svar = samling.query(
        query_embeddings=vektor.tolist(),
        n_results=antal * 3,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    traffar, tagna = [], Counter()
    for dok, meta, avstand in zip(svar["documents"][0], svar["metadatas"][0], svar["distances"][0]):
        anforande = meta["anforande_id"]
        if tagna[anforande] >= per_anforande:
            continue
        tagna[anforande] += 1
        traffar.append({"text": dok, "meta": meta, "likhet": 1 - avstand})
        if len(traffar) == antal:
            break
    return traffar


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


def stromma_svar(klient, fraga, underlag, modell=SVARSMODELL):
    """Frågar Claude och lämnar tillbaka svaret bit för bit.

    Funktionen är en generator i stället för att skriva ut något själv. Då kan
    ask.py printa bitarna medan de kommer, och app.py skicka samma bitar till
    st.write_stream - utan att den här koden vet vilket som gäller.
    """
    meddelande = (
        f"Fråga: {fraga}\n\n"
        f"Här är utdragen ur riksdagens anföranden:\n\n{underlag}"
    )
    # Ingen temperature-parameter: den är borttagen på Opus 5, Sonnet 5 och
    # 4.7/4.8-familjen. Det är systemprompten som håller svaret vid källorna.
    with klient.messages.stream(
        model=modell,
        max_tokens=2000,
        system=SYSTEMPROMPT,
        messages=[{"role": "user", "content": meddelande}],
    ) as strom:
        yield from strom.text_stream
