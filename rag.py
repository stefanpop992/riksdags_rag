#!/usr/bin/env python3
"""Gemensam RAG-logik för terminalversionen (ask.py) och webbappen (app.py).

Modulen känner varken till terminalen eller Streamlit. Den söker, bygger
underlag och frågar Claude - hur svaret sedan visas bestämmer den som anropar.
"""

import glob
import json
import os
import re
from collections import Counter
from pathlib import Path

import anthropic
import chromadb
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

EMBEDDINGSMODELL = "intfloat/multilingual-e5-large"
SVARSMODELL = "claude-opus-5"
# Planeringen och granskningen är strukturerade rutinuppgifter: bryt ned en
# fråga, bedöm om träffarna räcker. De behöver inte samma modell som skriver
# det slutliga svaret, och en snabbare modell märks direkt i väntetiden.
PLANERINGSMODELL = "claude-sonnet-5"
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


def las_talare(datakatalog="data"):
    """Alla unika talarsträngar i materialet, cachade till disk.

    Det är bara 765 stycken, så vi kan hålla hela listan i minnet och slippa
    läsa metadata ur ChromaDB (som bara kan exakt-matcha, inte söka delsträng).
    """
    cache = Path(datakatalog) / "talare.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    namn = set()
    for fil in glob.glob(str(Path(datakatalog) / "anforanden_*.jsonl")):
        with open(fil, encoding="utf-8") as f:
            for rad in f:
                try:
                    namn.add(json.loads(rad)["talare"])
                except (json.JSONDecodeError, KeyError):
                    pass
    lista = sorted(n for n in namn if n)
    if lista:
        cache.write_text(json.dumps(lista, ensure_ascii=False), encoding="utf-8")
    return lista


def talarvarianter(namn, alla_talare):
    """Alla skrivningar av en talare, matchat på delsträng.

    Samma person förekommer med och utan titel - "Ulf Kristersson (M)" och
    "Statsministern Ulf Kristersson (M)" - så ett exakt namn räcker inte.
    Vi plockar fram alla varianter och låter ChromaDB filtrera med $in.
    """
    if not namn:
        return []
    nyckel = namn.strip().lower()
    return [t for t in alla_talare if nyckel in t.lower()]


def bygg_filter(parti=None, fran_ar=None, till_ar=None, talare=None):
    """Bygger ChromaDB:s where-villkor. Returnerar None om inget filter satts.

    talare är en lista med exakta talarsträngar, från talarvarianter().
    """
    villkor = []
    if parti:
        villkor.append({"parti": parti.upper()})
    if talare:
        villkor.append({"talare": {"$in": list(talare)}})
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


def stromma_svar(klient, fraga, underlag, modell=SVARSMODELL, pa_tanke=None):
    """Frågar Claude och lämnar tillbaka svaret bit för bit.

    Funktionen är en generator i stället för att skriva ut något själv. Då kan
    ask.py printa bitarna medan de kommer, och app.py skicka samma bitar till
    st.write_stream - utan att den här koden vet vilket som gäller.

    pa_tanke är en valfri funktion som får modellens resonemang medan det
    pågår. Den hålls utanför det som yield:as, så att tänkandet aldrig kan
    hamna i svarstexten - samma uppdelning som pa_steg i agentisk_sokning.
    """
    meddelande = (
        f"Fråga: {fraga}\n\n"
        f"Här är utdragen ur riksdagens anföranden:\n\n{underlag}"
    )
    # Ingen temperature-parameter: den är borttagen på Opus 5, Sonnet 5 och
    # 4.7/4.8-familjen. Det är systemprompten som håller svaret vid källorna.
    #
    # Opus 5 tänker igenom svaret först, vilket tar några sekunder innan första
    # tecknet kommer. Tänkandet sker oavsett och kostar lika mycket - display
    # styr bara om vi får se en sammanfattning av det. Utan den ser pausen ut
    # som att appen har hängt sig.
    with klient.messages.stream(
        model=modell,
        max_tokens=2000,
        thinking={"type": "adaptive", "display": "summarized"},
        system=SYSTEMPROMPT,
        messages=[{"role": "user", "content": meddelande}],
    ) as strom:
        for handelse in strom:
            if handelse.type != "content_block_delta":
                continue
            if handelse.delta.type == "text_delta":
                yield handelse.delta.text
            elif handelse.delta.type == "thinking_delta" and pa_tanke:
                pa_tanke(handelse.delta.thinking)


# ===========================================================================
# Agentisk retrieval
# ===========================================================================
#
# Naivt läge gör ett svep: fråga -> en sökning -> svar. Det räcker för smala
# frågor men missar breda ("vad tycker partierna om X?"), eftersom en enda
# vektorsökning bara returnerar det som liknar frågan mest - inte ett
# representativt urval.
#
# Agentiskt läge låter modellen planera sökningarna, granska vad den fick och
# söka om. Tre roller, tre anrop:
#   planera()  - bryter ned frågan i delfrågor, en per parti eller aspekt
#   bedom()    - läser vad sökningen gav och avgör om det räcker
#   sista svaret genereras av samma stromma_svar() som naivt läge använder


class Delfraga(BaseModel):
    """En sökning som agenten vill göra."""

    sokfraga: str = Field(description="Sökord på svenska, formulerat som talarna skulle uttrycka sig")
    parti: str | None = Field(description="Partikod att filtrera på, eller null för alla partier")
    talare: str | None = Field(description="Efternamn eller helt namn att filtrera på, eller null")
    motivering: str = Field(description="Kort: varför denna sökning behövs")


class Plan(BaseModel):
    delfragor: list[Delfraga]


class Bedomning(BaseModel):
    """Agentens egen granskning av det den hittat."""

    racker_underlaget: bool = Field(description="Går frågan att besvara med det som hittats?")
    saknas: list[str] = Field(description="Vad som saknas underlag för. Tom lista om inget saknas.")
    nya_sokningar: list[Delfraga] = Field(description="Omformulerade sökningar att prova. Tom lista om underlaget räcker.")


PLANERARPROMPT = """Du planerar sökningar i en databas med svenska riksdagsanföranden från 2018 till 2026.

Databasen söks med vektorlikhet, inte nyckelord. Varje sökning kan dessutom filtreras
på parti och på talare.
Partikoder: S, M, SD, C, V, KD, L, MP.

Bryt ned användarens fråga i de sökningar som behövs för att besvara den ordentligt.

VÄLJ RÄTT FILTER FÖR FRÅGAN
- Gäller frågan en namngiven person: sätt talare till personens namn och lämna parti som null.
  Skriv hela namnet, inte bara efternamnet - "Åkesson" matchar både Jimmie Åkesson (SD)
  och Anders Åkesson (C).
- Gäller frågan "partierna" eller jämför ståndpunkter: gör en sökning per parti,
  och lämna talare som null.
- Gäller frågan ett ämne utan person eller partiavgränsning: lämna båda som null.

ÖVRIGT
- Är frågan bred: dela upp den i delaspekter.
- Är frågan smal och specifik: en eller två sökningar räcker. Hitta inte på fler än nödvändigt.
- Formulera sökfrågorna som riksdagsledamöter faktiskt uttrycker sig, inte som en sökmotorfråga.
- Gäller frågan en person kan det ändå löna sig att variera ämnesorden mellan sökningarna."""

GRANSKARPROMPT = """Du granskar om ett sökresultat räcker för att besvara en fråga om svenska riksdagsanföranden.

Du får frågan, vilka sökningar som gjorts och en sammanfattning av vad de gav.

Ribban är "går frågan att besvara?", inte "är täckningen perfekt?". Ett parti som
är tunt representerat men ändå har ett tydligt uttalande räcker. Kräv inte fler
utdrag för sakens skull - varje extra varv kostar tid och pengar.

Sätt racker_underlaget till FALSE bara när något väsentligt saknas helt, till
exempel att ett parti som frågan uttryckligen gäller inte har ett enda utdrag.

Två fällor att undvika:
- Föreslå aldrig en sökning som i praktiken upprepar en redan gjord. Gav den
  inget första gången ger en omformulering sällan något heller.
- Saknas ett ämne för att det inte diskuterats i kammaren går det inte att söka
  fram. Skriv det i saknas och sätt nya_sokningar till tom lista.

Du får lista saker i saknas även när racker_underlaget är true - då redovisas de
som förbehåll i svaret utan att ett nytt sökvarv körs."""


def planera(klient, fraga, modell=PLANERINGSMODELL):
    """Låter modellen bryta ned frågan i sökningar."""
    svar = klient.messages.parse(
        model=modell,
        max_tokens=2000,
        # effort styr hur djupt modellen tänker innan den svarar. Standard är
        # "high", vilket är onödigt för att fylla i ett fast JSON-schema.
        output_config={"format": {"type": "json_schema", "schema": Plan.model_json_schema()},
                       "effort": "low"},
        output_format=Plan,
        system=PLANERARPROMPT,
        messages=[{"role": "user", "content": f"Fråga: {fraga}"}],
    )
    return svar.parsed_output.delfragor


def _sammanfatta_for_granskning(traffar, tecken=400):
    """Kort sammandrag av träffarna - nog för att bedöma täckning, utan att
    skicka hela underlaget en gång till i varje granskningsvarv."""
    rader = []
    for i, t in enumerate(traffar, 1):
        m = t["meta"]
        rader.append(f"[{i}] {formatera_talare(m)}, {m['datum']}, {m['debattrubrik']}\n"
                     f"    {t['text'][:tecken]}")
    return "\n".join(rader) if rader else "(sökningen gav inga träffar alls)"


def bedom(klient, fraga, delfragor, traffar, modell=PLANERINGSMODELL):
    """Låter modellen granska sitt eget sökresultat."""
    gjorda = "\n".join(
        f"- {d.sokfraga}" + (f" (parti: {d.parti})" if d.parti else "") for d in delfragor)
    innehall = (
        f"Fråga: {fraga}\n\n"
        f"Sökningar som gjorts:\n{gjorda}\n\n"
        f"Vad de gav ({len(traffar)} utdrag):\n{_sammanfatta_for_granskning(traffar)}"
    )
    svar = klient.messages.parse(
        model=modell,
        max_tokens=2000,
        output_config={"format": {"type": "json_schema", "schema": Bedomning.model_json_schema()},
                       "effort": "low"},
        output_format=Bedomning,
        system=GRANSKARPROMPT,
        messages=[{"role": "user", "content": innehall}],
    )
    return svar.parsed_output


def agentisk_sokning(klient, samling, modell, fraga, anvandar_parti=None,
                     fran_ar=None, till_ar=None, per_delfraga=3, max_utdrag=24,
                     max_varv=3, per_anforande=1, planeringsmodell=PLANERINGSMODELL,
                     pa_steg=None, alla_talare=None, anvandar_talare=None):
    """Planerar, söker, granskar och söker om tills underlaget räcker.

    pa_steg är en valfri funktion som anropas med varje steg medan loopen kör.
    Appen använder den för att visa arbetet live; terminalen för att skriva ut
    det. Modulen behöver inte veta vilket.

    Returnerar (traffar, saknas, spar):
      traffar - de utdrag som hittades, bäst först
      saknas  - vad agenten själv säger att den inte hittade underlag för
      spar    - vad som hände i varje varv, för att kunna visas i efterhand
    """
    spar = []

    def logga(steg):
        spar.append(steg)
        if pa_steg:
            pa_steg(steg)

    if alla_talare is None:
        alla_talare = las_talare()

    delfragor = planera(klient, fraga, planeringsmodell)
    logga({"typ": "plan", "delfragor": delfragor})

    funna = {}   # chunk-id -> träff, så samma utdrag inte räknas två gånger
    saknas = []

    for varv in range(1, max_varv + 1):
        nya = 0
        for d in delfragor:
            # Användarens egna filter i sidopanelen vinner över planens.
            namn = anvandar_talare or d.talare
            where = bygg_filter(anvandar_parti or d.parti, fran_ar, till_ar,
                                talarvarianter(namn, alla_talare))
            for t in sok(samling, modell, d.sokfraga, per_delfraga, where, per_anforande):
                nyckel = f"{t['meta']['anforande_id']}:{t['meta']['chunk_nr']}"
                if nyckel not in funna:
                    funna[nyckel] = t
                    nya += 1

        traffar = sorted(funna.values(), key=lambda t: -t["likhet"])[:max_utdrag]
        logga({"typ": "sokning", "varv": varv, "nya": nya, "totalt": len(traffar)})

        bedomning = bedom(klient, fraga, delfragor, traffar, planeringsmodell)
        saknas = bedomning.saknas
        logga({"typ": "bedomning", "varv": varv,
               "racker": bedomning.racker_underlaget,
               "saknas": bedomning.saknas,
               "nya_sokningar": bedomning.nya_sokningar})

        if bedomning.racker_underlaget or not bedomning.nya_sokningar:
            break
        if varv == max_varv:
            logga({"typ": "stopp", "skal": f"nådde taket på {max_varv} varv"})
            break
        delfragor = bedomning.nya_sokningar

    return sorted(funna.values(), key=lambda t: -t["likhet"])[:max_utdrag], saknas, spar


def bygg_underlag_med_luckor(traffar, saknas):
    """Samma underlag som naivt läge, plus agentens lista över luckor.

    Systemprompten är identisk i båda lägena. Skillnaden ligger i vad som
    hämtats, inte i hur modellen instrueras - annars gick lägena inte att
    jämföra rättvist i eval.py.
    """
    underlag = bygg_underlag(traffar)
    if saknas:
        punkter = "\n".join(f"- {s}" for s in saknas)
        underlag += (
            "\n\n[LUCKOR I UNDERLAGET]\n"
            "Sökningen hittade inget underlag för följande. Redovisa detta ärligt "
            "i slutet av svaret i stället för att gissa:\n" + punkter
        )
    return underlag
