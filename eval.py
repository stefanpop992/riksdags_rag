#!/usr/bin/env python3
"""Mäter naivt mot agentiskt läge på ett fast frågeset.

Två mått:
  citatprecision - pekar citaten i svaret på källor som faktiskt fanns i
                   underlaget? Mäts mekaniskt, utan modell.
  groundedness   - står det som påstås faktiskt i den angivna källan? Mäts av
                   en bedömarmodell som får svaret och källorna.

Körs så här:
    python eval.py --limit 2      # rökprov, ett par frågor
    python eval.py                # hela frågesetet
    python eval.py --rapport      # bygg om tabellen av sparade resultat
"""

import argparse
import json
import re
import statistics
import time
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field
from tqdm import tqdm

import rag

RESULTATFIL = Path("eval_resultat.json")
RAPPORTFIL = Path("eval_resultat.md")

# Fyra kategorier, för att kunna se VAR agentiken lönar sig i stället för att
# dränka skillnaden i ett medelvärde över blandade frågor.
FRAGOR = [
    # --- smala faktafrågor: en sökning borde räcka ---
    ("smal", "Vad har sagts om slutförvar av använt kärnbränsle?"),
    ("smal", "Vilka invändningar har rests mot ett vinsttak i friskolor?"),
    ("smal", "Vad har sagts om a-kassans ersättningsnivåer?"),
    ("smal", "Hur har elprisstödet till hushållen beskrivits i kammaren?"),
    ("smal", "Vad har sagts om Sveriges militära stöd till Ukraina?"),
    ("smal", "Vilka skäl har angetts för att höja försvarsanslagen till två procent av BNP?"),
    ("smal", "Vad har sagts om vårdköer och vårdgarantin?"),
    # --- breda partijämförelser: här borde agentiken vinna ---
    ("bred", "Vad tycker partierna om kärnkraft?"),
    ("bred", "Hur skiljer sig partiernas syn på migrationspolitiken?"),
    ("bred", "Vad tycker partierna om vinster i välfärden?"),
    ("bred", "Hur ser partierna på Sveriges Nato-medlemskap?"),
    ("bred", "Vad tycker partierna om höjda försvarsanslag?"),
    ("bred", "Hur skiljer sig partiernas klimatpolitik?"),
    ("bred", "Vad tycker partierna om straffskärpningar mot gängkriminalitet?"),
    ("bred", "Hur ser partierna på arbetskraftsinvandring?"),
    # --- personfrågor: kräver talarfiltret ---
    ("person", "Vad har Ulf Kristersson sagt om Nato?"),
    ("person", "Vad har Jimmie Åkesson sagt om invandring?"),
    ("person", "Vad har Nooshi Dadgostar sagt om vinster i välfärden?"),
    ("person", "Vad har Magdalena Andersson sagt om svensk ekonomi?"),
    ("person", "Vad har Ebba Busch sagt om elpriser?"),
    ("person", "Vad har Annie Lööf sagt om företagande?"),
    ("person", "Vad har Märta Stenevi sagt om klimatpolitiken?"),
    ("person", "Vad har Johan Pehrson sagt om skolan?"),
    # --- obesvarbara: mäter om systemet avstår i stället för att gissa ---
    ("omojlig", "Vad kostar en liter mjölk i en svensk butik?"),
    ("omojlig", "Vilket lag vann Allsvenskan 2024?"),
    ("omojlig", "Hur långt är det mellan Stockholm och Göteborg?"),
    ("omojlig", "Vilken är Sveriges högsta byggnad?"),
    ("omojlig", "Hur många mål gjorde Zlatan Ibrahimovic i landslaget?"),
    ("omojlig", "Vad kostar ett månadskort i Stockholms kollektivtrafik?"),
    ("omojlig", "Vem vann Melodifestivalen 2023?"),
]

# Citaten ska se ut som [Ulf Kristersson (M), 2022-05-16]
CITAT = re.compile(r"\[([^\[\]]+?),\s*(\d{4}-\d{2}-\d{2})\]")


class Pastaende(BaseModel):
    pastaende: str = Field(description="Påståendet, kort återgivet")
    stods_av_kallorna: bool = Field(description="Står detta faktiskt i den angivna källan?")
    motivering: str = Field(description="Kort motivering")


class Granskning(BaseModel):
    pastaenden: list[Pastaende]
    avstod_helt: bool = Field(description="Säger svaret att underlaget inte räcker för att besvara frågan?")


BEDOMARPROMPT = """Du granskar om ett svar håller sig till sina källor.

Du får en fråga, ett svar och de källor svaret bygger på.

Plocka ut varje sakpåstående i svaret och avgör för vart och ett om det faktiskt
står i källorna. Var strikt:
- Ett påstående som går utöver källan, även rimligt, ska markeras som ej stött.
- Att svaret korrekt återger vad någon SADE räknas som stött, även om utsagan i
  sig är politiskt omtvistad. Vi mäter trohet mot källan, inte sanningshalt.
- Rena övergångsmeningar och sammanfattande formuleringar utan sakinnehåll ska
  du hoppa över, inte lista som påståenden.

Sätt avstod_helt till true om svaret i huvudsak säger att underlaget inte räcker
för att besvara frågan."""


def citatprecision(svar, traffar):
    """Andel citat i svaret som pekar på en källa som faktiskt fanns med.

    Mäts mekaniskt: vi plockar ut [Namn, datum] ur svaret och kollar mot
    metadatan i de utdrag som skickades. Ingen modell inblandad, så måttet kan
    inte påverkas av samma fel som det ska upptäcka.
    """
    facit = {(rag.formatera_talare(t["meta"]), t["meta"]["datum"]) for t in traffar}
    # Efternamn + datum som mildare jämförelse, ifall modellen kortar titeln
    facit_efternamn = {(namn.split("(")[0].strip().split()[-1], datum)
                       for namn, datum in facit}

    citat = CITAT.findall(svar)
    if not citat:
        return None, 0, 0

    ratta = 0
    for namn, datum in citat:
        namn = namn.strip()
        if (namn, datum) in facit:
            ratta += 1
        elif namn.split("(")[0].strip():
            efternamn = namn.split("(")[0].strip().split()[-1]
            if (efternamn, datum) in facit_efternamn:
                ratta += 1
    return ratta / len(citat), ratta, len(citat)


def bedom_svar(klient, fraga, svar, underlag, modell=rag.SVARSMODELL):
    """Låter en bedömarmodell kontrollera varje påstående mot källorna."""
    innehall = (f"FRÅGA\n{fraga}\n\nSVAR SOM SKA GRANSKAS\n{svar}\n\nKÄLLOR\n{underlag}")
    # Ett agentiskt svar med 24 källor ger många påståenden att lista. För lågt
    # tak kapar JSON:en mitt i och Pydantic-valideringen faller.
    r = klient.messages.parse(
        model=modell,
        max_tokens=16000,
        output_config={"format": {"type": "json_schema", "schema": Granskning.model_json_schema()}},
        output_format=Granskning,
        system=BEDOMARPROMPT,
        messages=[{"role": "user", "content": innehall}],
    )
    return r.parsed_output


def kor_en(klient, samling, modell, alla_talare, fraga, lage, max_varv):
    """Kör en fråga i ett läge och returnerar svar, underlag och träffar."""
    t0 = time.time()
    if lage == "agentiskt":
        traffar, saknas, spar = rag.agentisk_sokning(
            klient, samling, modell, fraga, max_varv=max_varv,
            alla_talare=alla_talare)
        underlag = rag.bygg_underlag_med_luckor(traffar, saknas)
        varv = max([s["varv"] for s in spar if s["typ"] == "bedomning"], default=0)
        anrop = 1 + varv + 1          # plan + granskningar + svar
    else:
        traffar = rag.sok(samling, modell, fraga, antal=8)
        underlag = rag.bygg_underlag(traffar)
        varv, anrop = 1, 1

    svar = "".join(rag.stromma_svar(klient, fraga, underlag))
    return {
        "svar": svar,
        "underlag": underlag,
        "traffar": [{"talare": rag.formatera_talare(t["meta"]),
                     "parti": t["meta"]["parti"],
                     "datum": t["meta"]["datum"]} for t in traffar],
        "antal_utdrag": len(traffar),
        "partier": sorted({t["meta"]["parti"] for t in traffar}),
        "varv": varv,
        "api_anrop": anrop,
        "sekunder": round(time.time() - t0, 1),
    }


def las_resultat():
    if RESULTATFIL.exists():
        return json.loads(RESULTATFIL.read_text(encoding="utf-8"))
    return {}


def spara_resultat(resultat):
    RESULTATFIL.write_text(json.dumps(resultat, ensure_ascii=False, indent=1), encoding="utf-8")


def bygg_rapport(resultat):
    """Sammanställer mätningarna till en markdown-tabell för README."""
    per = {}   # (kategori, lage) -> lista med mätvärden
    for post in resultat.values():
        nyckel = (post["kategori"], post["lage"])
        per.setdefault(nyckel, []).append(post)

    kategorier = ["smal", "bred", "person", "omojlig"]
    namn = {"smal": "Smal faktafråga", "bred": "Bred partijämförelse",
            "person": "Personfråga", "omojlig": "Obesvarbar"}

    rader = []
    for kat in kategorier:
        for lage in ("naivt", "agentiskt"):
            poster = per.get((kat, lage), [])
            if not poster:
                continue
            prec = [p["citatprecision"] for p in poster if p["citatprecision"] is not None]
            grund = [p["groundedness"] for p in poster if p["groundedness"] is not None]
            rader.append({
                "kategori": namn[kat],
                "lage": lage,
                "n": len(poster),
                "citatprecision": statistics.mean(prec) if prec else None,
                "groundedness": statistics.mean(grund) if grund else None,
                "partier": statistics.mean([len(p["partier"]) for p in poster]),
                "utdrag": statistics.mean([p["antal_utdrag"] for p in poster]),
                "avstod": statistics.mean([1 if p["avstod_helt"] else 0 for p in poster]),
                "sekunder": statistics.mean([p["sekunder"] for p in poster]),
                "anrop": statistics.mean([p["api_anrop"] for p in poster]),
            })

    def procent(x):
        return "–" if x is None else f"{100 * x:.0f} %"

    rrader = ["# Utvärdering: naivt mot agentiskt läge", "",
              f"{len({p['fraga'] for p in resultat.values()})} frågor, fyra kategorier. "
              "Samma systemprompt i båda lägena – ",
              "skillnaden ligger enbart i hur underlaget hämtas.", "",
              "| Kategori | Läge | n | Citatprecision | Groundedness | Partier | Utdrag | Avstod | Sek | Anrop |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rader:
        rrader.append(
            f"| {r['kategori']} | {r['lage']} | {r['n']} | {procent(r['citatprecision'])} | "
            f"{procent(r['groundedness'])} | {r['partier']:.1f} | {r['utdrag']:.1f} | "
            f"{procent(r['avstod'])} | {r['sekunder']:.0f} | {r['anrop']:.1f} |")

    # Totalrad per läge
    rrader += ["", "## Totalt", "",
               "| Läge | Citatprecision | Groundedness | Sek/fråga | Anrop/fråga |",
               "|---|---|---|---|---|"]
    for lage in ("naivt", "agentiskt"):
        poster = [p for p in resultat.values() if p["lage"] == lage]
        if not poster:
            continue
        prec = [p["citatprecision"] for p in poster if p["citatprecision"] is not None]
        grund = [p["groundedness"] for p in poster if p["groundedness"] is not None]
        rrader.append(
            f"| {lage} | {procent(statistics.mean(prec) if prec else None)} | "
            f"{procent(statistics.mean(grund) if grund else None)} | "
            f"{statistics.mean([p['sekunder'] for p in poster]):.0f} | "
            f"{statistics.mean([p['api_anrop'] for p in poster]):.1f} |")

    rrader += [
        "",
        "## Så mäts det",
        "",
        "- **Citatprecision** mäts mekaniskt: citaten `[Namn, datum]` plockas ur svaret med",
        "  reguljärt uttryck och jämförs mot metadatan i de utdrag som faktiskt skickades.",
        "  Ingen modell inblandad, så måttet kan inte drabbas av samma fel det ska upptäcka.",
        "- **Groundedness** bedöms av Claude, som får frågan, svaret och källorna och prövar",
        "  varje påstående för sig. Att korrekt återge vad någon *sade* räknas som stött -",
        "  vi mäter trohet mot källan, inte sanningshalt.",
        "- **Partier** och **Utdrag** är genomsnitt över frågorna i kategorin.",
        "- **Avstod** = andel svar som i huvudsak säger att underlaget inte räcker.",
        "",
        "## Förbehåll",
        "",
        "- **Kategorin \"obesvarbar\" mäter inte vad den var tänkt att mäta.** Bara 3 av 7",
        "  frågor saknade verkligen underlag (Melodifestivalen, Allsvenskan, Zlatan). De",
        "  övriga - mjölkpris, avståndet Stockholm-Göteborg, Sveriges högsta byggnad,",
        "  månadskort - *förekommer* i anförandena, eftersom ledamöter använder vardagsfakta",
        "  retoriskt. Systemet svarade korrekt genom att tillskriva uppgiften den som sade",
        "  den. Avstod-siffran i den raden är därför utspädd, inte ett tecken på gissningar.",
        "- **Naivt läge har inget talarfilter.** Det är avsiktligt: filtret är ett verktyg",
        "  agenten kan välja att använda. Det förklarar hela skillnaden i personraden.",
        "- **Kolumnen Partier är bara meningsfull för breda frågor.** För personfrågor är",
        "  1.0 det korrekta värdet, inte ett dåligt.",
        "- **Bedömaren är samma modellfamilj som den som svarar.** Groundedness-siffran bör",
        "  läsas som en indikation, inte ett oberoende facit.",
        "",
    ]
    return "\n".join(rrader)


def main():
    parser = argparse.ArgumentParser(description="Utvärderar naivt mot agentiskt läge.")
    parser.add_argument("--limit", type=int, help="Kör bara så här många frågor (rökprov).")
    parser.add_argument("--max-varv", type=int, default=3)
    parser.add_argument("--rapport", action="store_true",
                        help="Bygg bara om rapporten av redan sparade resultat.")
    args = parser.parse_args()

    resultat = las_resultat()

    if args.rapport:
        RAPPORTFIL.write_text(bygg_rapport(resultat), encoding="utf-8")
        print(bygg_rapport(resultat))
        return

    klient = rag.skapa_klient()
    samling = rag.oppna_samling()
    print("Laddar sökmodellen ...")
    modell = rag.ladda_modell()
    alla_talare = rag.las_talare()

    fragor = FRAGOR[:args.limit] if args.limit else FRAGOR
    jobb = [(kat, f, lage) for kat, f in fragor for lage in ("naivt", "agentiskt")]
    # Återupptagning: hoppa över det som redan är mätt, precis som de andra skripten.
    jobb = [j for j in jobb if f"{j[1]}|{j[2]}" not in resultat]
    print(f"{len(jobb)} körningar kvar av {len(fragor) * 2}.\n")

    for kategori, fraga, lage in tqdm(jobb, unit="körning"):
        try:
            post = kor_en(klient, samling, modell, alla_talare, fraga, lage, args.max_varv)
            granskning = bedom_svar(klient, fraga, post["svar"], post["underlag"])
            prec, ratta, totalt = citatprecision(post["svar"],
                [{"meta": {"talare": t["talare"], "parti": t["parti"], "datum": t["datum"]}}
                 for t in post["traffar"]])
            stodda = [p.stods_av_kallorna for p in granskning.pastaenden]
            post.update({
                "kategori": kategori, "fraga": fraga, "lage": lage,
                "citatprecision": prec, "citat_ratta": ratta, "citat_totalt": totalt,
                "groundedness": (sum(stodda) / len(stodda)) if stodda else None,
                "antal_pastaenden": len(stodda),
                "avstod_helt": granskning.avstod_helt,
                "ostodda": [p.pastaende for p in granskning.pastaenden if not p.stods_av_kallorna],
            })
            post.pop("underlag")  # spara plats i resultatfilen
            resultat[f"{fraga}|{lage}"] = post
            spara_resultat(resultat)   # efter varje körning, så avbrott inte kostar om
        except Exception as fel:
            print(f"\nFEL på {fraga!r} ({lage}): {fel}")

    RAPPORTFIL.write_text(bygg_rapport(resultat), encoding="utf-8")
    print("\n" + bygg_rapport(resultat))
    print(f"Rådata i {RESULTATFIL}, tabell i {RAPPORTFIL}")


if __name__ == "__main__":
    main()
