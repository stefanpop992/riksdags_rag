#!/usr/bin/env python3
"""Jämför quick/deep med vector/hybrid på ett fast frågeset.

Två mått:
  citatprecision - pekar citaten i svaret på källor som faktiskt fanns i
                   underlaget? Mäts mekaniskt, utan modell.
  groundedness   - står det som påstås faktiskt i den angivna källan? Mäts av
                   en bedömarmodell som får svaret och källorna.

Körs så här:
    python eval.py --limit 2 --run-dir eval_runs/smoke
    python eval.py --resume --run-dir eval_runs/smoke
    python eval.py --rapport --run-dir eval_runs/smoke
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import re
import statistics
import subprocess
import time
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field
from tqdm import tqdm

import rag
import bm25

ROOT = Path(__file__).resolve().parent
CONFIGS = ("quick-vector", "quick-hybrid", "deep-vector", "deep-hybrid")
SCHEMA_VERSION = 1

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


class MatKlient:
    """Mäter logiska SDK-anrop och rapporterad usage utan att ändra RAG-koden."""

    def __init__(self, klient):
        self.original = klient.messages
        self.messages = self
        self.anrop = 0
        self.usage = []

    def registrera(self, svar):
        self.usage.append({
            "model": svar.model,
            **{key: getattr(svar.usage, key, 0) or 0 for key in (
                "input_tokens", "output_tokens", "cache_read_input_tokens",
                "cache_creation_input_tokens")},
        })

    def parse(self, **kwargs):
        self.anrop += 1
        svar = self.original.parse(**kwargs)
        self.registrera(svar)
        return svar

    @contextmanager
    def stream(self, **kwargs):
        self.anrop += 1
        with self.original.stream(**kwargs) as strom:
            yield strom
            self.registrera(strom.get_final_message())


def kor_en(klient, samling, modell, alla_talare, fraga, config, settings, bm25_index=None):
    """Kör appens sökväg; spara också underlaget för mänsklig granskning."""
    if config not in CONFIGS:
        raise ValueError(f"Okänd konfiguration: {config}")
    deep, hybrid = config.startswith("deep-"), config.endswith("-hybrid")
    if hybrid and bm25_index is None:
        raise ValueError("Hybridutvärdering kräver ett BM25-index.")
    index = bm25_index if hybrid else None
    t0 = time.perf_counter()
    saknas, spar = [], []
    if deep:
        traffar, saknas, spar = rag.agentisk_sokning(
            klient, samling, modell, fraga, max_varv=settings["max_varv"],
            per_anforande=settings["per_speech"],
            per_delfraga=settings["per_subquery"], max_utdrag=settings["max_excerpts"],
            alla_talare=alla_talare, bm25_index=index,
            planeringsmodell=settings["planning_model"])
        underlag = rag.bygg_underlag_med_luckor(traffar, saknas)
    else:
        kwargs = dict(antal=settings["count"], per_anforande=settings["per_speech"])
        traffar = (rag.sok_hybrid(samling, modell, fraga, index, **kwargs) if hybrid
                   else rag.sok(samling, modell, fraga, **kwargs))
        underlag = rag.bygg_underlag(traffar)
    # Samma beteende som server.py: inga träffar innebär inget svarsanrop.
    svar = ("".join(rag.stromma_svar(klient, fraga, underlag, modell=settings["answer_model"]))
            if traffar else "Inga anföranden matchade sökningen.")
    return {
        "svar": svar, "underlag": underlag, "kallor": traffar, "saknas": saknas,
        "antal_utdrag": len(traffar),
        "partier": sorted({t["meta"]["parti"] for t in traffar}),
        "varv": max((s["varv"] for s in spar if s["typ"] == "bedomning"), default=0),
        "spar": json.loads(json.dumps(spar, default=lambda obj: obj.model_dump())),
        "sekunder": round(time.perf_counter() - t0, 3),
    }


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def korning_settings(args, sparade=None):
    defaults = {
        "configs": list(CONFIGS), "limit": None, "max_varv": 3, "count": 8,
        "per_speech": 1, "per_subquery": 3, "max_excerpts": 24,
        "answer_model": rag.SVARSMODELL, "planning_model": rag.PLANERINGSMODELL,
        "judge_model": rag.SVARSMODELL, "embedding_model": rag.EMBEDDINGSMODELL,
    }
    settings = dict(sparade or defaults)
    for key in defaults:
        if getattr(args, key, None) is not None:
            settings[key] = getattr(args, key)
    # Dubbletter eller annan flaggordning ska inte skapa extra jobb.
    settings["configs"] = [c for c in CONFIGS if c in settings["configs"]]
    if sparade and settings != sparade:
        raise ValueError("Inställningarna skiljer sig från den sparade körningen. Välj en ny --run-dir.")
    return settings


def kodversion():
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=ROOT, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    status = git("status", "--porcelain")
    return {
        "commit": git("rev-parse", "HEAD"), "dirty": bool(status) if status is not None else None,
        "sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                   for name in ("eval.py", "rag.py", "bm25.py", "index_data.py")},
    }


def filhash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataversion(samling, talare, hybrid):
    # Samlingsnamn och antal är en grov kontroll, inte en innehållshash av Chroma.
    return {
        "collection": samling.name, "count": samling.count(),
        "chroma_dir": str(Path("chroma_db").resolve()),
        "speakers_sha256": hashlib.sha256(json.dumps(talare, ensure_ascii=False).encode()).hexdigest(),
        "bm25_sha256": {name: filhash(Path(bm25.INDEXKATALOG) / name)
                        for name in ("matris.npz", "ordlista.json", "ider.json")} if hybrid else None,
    }


def ny_korning(settings):
    fragor = FRAGOR[:settings["limit"]] if settings["limit"] else FRAGOR
    return {
        "schema_version": SCHEMA_VERSION, "created_at": utc_now(),
        "settings": settings, "questions": [list(q) for q in fragor],
        "code": kodversion(),
        "packages": {name: version(name) for name in (
            "anthropic", "chromadb", "sentence-transformers", "torch", "numpy", "scipy", "scikit-learn")},
        "dataset": None, "results": {}, "errors": {},
    }


def las_korning(katalog):
    run = json.loads((Path(katalog) / "results.json").read_text(encoding="utf-8"))
    if run.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Okänt resultatformat. Den historiska eval_resultat.json återupptas inte här.")
    return run


def spara_korning(run, katalog):
    """Ersätt filen atomiskt, så avbrott inte förstör tidigare mätningar."""
    path = Path(katalog) / "results.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def jobb_for(run):
    return [(kat, fraga, config) for kat, fraga in run["questions"]
            for config in run["settings"]["configs"]]


def jobbnyckel(fraga, config):
    return f"{config}|{fraga}"


def kor_jobb(run, katalog, klient, samling, modell, talare, index):
    """Checkpointa svaret före bedömningen; återuppta bara ofärdiga steg."""
    settings = run["settings"]
    jobb = [(kat, f, c) for kat, f, c in jobb_for(run)
            if run["results"].get(jobbnyckel(f, c), {}).get("status") != "complete"]
    print(f"{len(jobb)} körningar kvar av {len(jobb_for(run))}.")
    for kategori, fraga, config in tqdm(jobb, unit="körning"):
        key = jobbnyckel(fraga, config)
        matning = MatKlient(klient)
        phase = "generation"
        try:
            post = run["results"].get(key)
            if post is None:
                post = kor_en(matning, samling, modell, talare, fraga, config, settings, index)
                post.update(kategori=kategori, fraga=fraga, config=config, status="answer_ready",
                            api_anrop=matning.anrop, usage=list(matning.usage), created_at=utc_now())
                run["results"][key] = post
                spara_korning(run, katalog)
            phase = "judge"
            domare = MatKlient(klient)
            matning = domare
            t0 = time.perf_counter()
            if post["antal_utdrag"]:
                granskning = bedom_svar(domare, fraga, post["svar"], post["underlag"],
                                       modell=settings["judge_model"])
            else:
                granskning = Granskning(pastaenden=[], avstod_helt=True)
            prec, ratta, totalt = citatprecision(post["svar"], post["kallor"])
            stodda = [p.stods_av_kallorna for p in granskning.pastaenden]
            post.update(
                citatprecision=prec, citat_ratta=ratta, citat_totalt=totalt,
                groundedness=sum(stodda) / len(stodda) if stodda else None,
                antal_pastaenden=len(stodda), avstod_helt=granskning.avstod_helt,
                granskning=granskning.model_dump(),
                ostodda=[p.pastaende for p in granskning.pastaenden if not p.stods_av_kallorna],
                judge_api_anrop=domare.anrop, judge_usage=list(domare.usage),
                judge_seconds=round(time.perf_counter() - t0, 3),
                status="complete", completed_at=utc_now())
            run["errors"].pop(key, None)
        except Exception as error:
            # Misslyckade försök finns kvar även efter lyckad återupptagning.
            failure = {"at": utc_now(), "phase": phase, "type": type(error).__name__,
                       "message": str(error), "api_anrop": matning.anrop,
                       "usage": list(matning.usage)}
            run.setdefault("failed_attempts", []).append({"key": key, **failure})
            run["errors"][key] = failure
            print(f"\nFEL på {fraga!r} ({config}): {error}")
        spara_korning(run, katalog)
        (Path(katalog) / "report.md").write_text(bygg_rapport(run), encoding="utf-8")


def bygg_rapport(run):
    """Håll varje konfiguration separat, även i en ofullständig körning."""
    legacy = "schema_version" not in run
    if legacy:
        # Läs historiken för rapportvisning; skriv aldrig om originalfilen.
        posts = [dict(p, config={"naivt": "quick-vector", "agentiskt": "deep-vector"}[p["lage"]])
                 for p in run.values()]
        configs = [c for c in CONFIGS if any(p["config"] == c for p in posts)]
        expected, errors = len(posts), 0
    else:
        posts = [p for p in run["results"].values() if p["status"] == "complete"]
        configs = run["settings"]["configs"]
        expected, errors = len(jobb_for(run)), len(run["errors"])

    def mean(rows, key):
        values = [p[key] for p in rows if p.get(key) is not None]
        return statistics.mean(values) if values else None

    def number(value, percent=False):
        if value is None:
            return "–"
        return f"{100 * value:.0f} %" if percent else f"{value:.1f}"

    def tokens(rows, field):
        values = [sum(u["input_tokens"] + u["output_tokens"] +
                      u.get("cache_read_input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
                      for u in p[field]) for p in rows if field in p]
        return number(statistics.mean(values) if values else None)

    lines = ["# Utvärdering: quick/deep × vector/hybrid", "",
             f"Slutförda: {len(posts)}/{expected}. Jobb med fel: {errors}.", ""]
    if not legacy:
        lines += [f"Start (UTC): {run['created_at']}",
                  f"Git: `{run['code']['commit']}`; lokala ändringar: {run['code']['dirty']}.",
                  f"Svar: `{run['settings']['answer_model']}`; planering: `{run['settings']['planning_model']}`; "
                  f"bedömare: `{run['settings']['judge_model']}`.",
                  "Inställningar, kodhashar, paketversioner och källutdrag finns i `results.json`.", ""]
    else:
        lines += ["Historiska vektorresultat; modell- och körningsmetadata saknas.", ""]
    covered = [{p["fraga"] for p in posts if p["config"] == c} for c in configs]
    if len(posts) != expected or (covered and any(q != covered[0] for q in covered)):
        lines += ["**Ofullständig/obalanserad jämförelse:** konfigurationerna kan ha olika "
                  "frågeunderlag. Medelvärdena ska inte användas som en rättvis jämförelse ännu.", ""]
    lines += ["## Totalt per konfiguration", "",
              "| Konfiguration | n | Citatprecision | Groundedness | Sek/fråga | Anrop/fråga | Token/fråga | Domartoken/fråga |",
              "|---|---|---|---|---|---|---|---|"]
    for config in configs:
        rows = [p for p in posts if p["config"] == config]
        lines.append(f"| {config} | {len(rows)} | {number(mean(rows, 'citatprecision'), True)} | "
                     f"{number(mean(rows, 'groundedness'), True)} | {number(mean(rows, 'sekunder'))} | "
                     f"{number(mean(rows, 'api_anrop'))} | {tokens(rows, 'usage')} | {tokens(rows, 'judge_usage')} |")
    lines += ["", "## Per kategori", "",
              "| Kategori | Konfiguration | n | Citatprecision | Groundedness | Partier | Utdrag | Avstod | Sek |",
              "|---|---|---|---|---|---|---|---|---|"]
    names = {"smal": "Smal faktafråga", "bred": "Bred partijämförelse",
             "person": "Personfråga", "omojlig": "Obesvarbar (ej validerad)"}
    for category, label in names.items():
        for config in configs:
            rows = [dict(p, party_count=len(p["partier"])) for p in posts
                    if p["kategori"] == category and p["config"] == config]
            if rows:
                lines.append(f"| {label} | {config} | {len(rows)} | "
                             f"{number(mean(rows, 'citatprecision'), True)} | "
                             f"{number(mean(rows, 'groundedness'), True)} | "
                             f"{number(mean(rows, 'party_count'))} | {number(mean(rows, 'antal_utdrag'))} | "
                             f"{number(mean(rows, 'avstod_helt'), True)} | {number(mean(rows, 'sekunder'))} |")
    lines += ["", "## Tolkning och begränsningar", "",
              "- Citatprecision kontrollerar referenser mot hämtad metadata, inte om påståendet stöds av citatet.",
              "- Groundedness bedöms av en modell och behöver mänsklig kontroll. Partier mäter täckning, inte korrekthet.",
              "- Frågor utan citat/påståenden saknar respektive poäng och ingår inte i det måttets medelvärde; tomma träffar räknas som avstående.",
              "- Sekunder avser sökning och svar, exklusive modellstart och separat svarbedömning.",
              "- Anrop är logiska SDK-anrop för sökning/svar, exklusive domaren och SDK:ns interna återförsök.",
              "- Token är rapporterade in-/ut-/cachetoken; domarens usage redovisas separat. Detta är inte ett kostnadsbelopp.",
              "- Misslyckade försök ingår inte i tabellernas medelvärden; de sparas separat. Usage kan saknas vid avbrutna anrop.",
              "- Quick-lägena tolkar inte automatiskt talarnamn som filter. Deep-lägena kan göra det; personfrågor mäter därför mer än rangordning.",
              "- Flera frågor i kategorin obesvarbar förekommer i arkivet. Frågesetet är ännu inte manuellt validerat för avstående.",
              "- Chroma identifieras med katalog, samlingsnamn och antal, inte full innehållshash. Frys datasetet under en körning.",
              "- Modellalias, svar och latenser kan variera mellan körningar. Alla konfigurationer körs i fast ordning.", ""]
    return "\n".join(lines)


def positivt_heltal(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("Värdet måste vara större än noll.")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description="Jämför quick/deep med vector/hybrid i separata körningar.")
    parser.add_argument("--run-dir", type=Path, help="Ny katalog, eller befintlig med --resume/--rapport.")
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--resume", action="store_true", help="Återuppta ofärdiga jobb i --run-dir.")
    operation.add_argument("--rapport", action="store_true", help="Rapport utan API-anrop; utan --run-dir visas historiken.")
    parser.add_argument("--configs", nargs="+", choices=CONFIGS, help="Standard: alla fyra konfigurationer.")
    parser.add_argument("--limit", type=positivt_heltal, help="Antal frågor; varje fråga körs i valda konfigurationer.")
    parser.add_argument("--max-varv", type=int, choices=range(1, 5))
    parser.add_argument("--count", type=int, choices=range(3, 16), help="Antal utdrag i quick (standard 8).")
    parser.add_argument("--per-speech", type=int, choices=range(1, 4), help="Utdrag per anförande (standard 1).")
    parser.add_argument("--answer-model")
    parser.add_argument("--planning-model")
    parser.add_argument("--judge-model")
    args = parser.parse_args(argv)
    try:
        if args.rapport:
            if args.run_dir:
                report = bygg_rapport(las_korning(args.run_dir))
                (args.run_dir / "report.md").write_text(report, encoding="utf-8")
            else:
                report = bygg_rapport(json.loads((ROOT / "eval_resultat.json").read_text(encoding="utf-8")))
            print(report)
            return 0
        if args.resume:
            if not args.run_dir:
                parser.error("--resume kräver --run-dir.")
            run = las_korning(args.run_dir)
            settings = korning_settings(args, run["settings"])
            if run["code"]["sha256"] != kodversion()["sha256"]:
                raise ValueError("Utvärderings-/sökkoden har ändrats. Starta en ny körning med --run-dir.")
            if all(run["results"].get(jobbnyckel(f, c), {}).get("status") == "complete"
                   for _, f, c in jobb_for(run)):
                print("Körningen är redan klar. Använd --rapport för att visa rapporten.")
                return 0
        else:
            settings = korning_settings(args)
            args.run_dir = args.run_dir or Path("eval_runs") / (
                datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
            if args.run_dir.exists():
                raise ValueError("Katalogen finns redan. Använd --resume eller välj en ny --run-dir.")
            run = ny_korning(settings)

        hybrid = any(c.endswith("-hybrid") for c in settings["configs"])
        index = bm25.ladda() if hybrid else None
        klient = rag.skapa_klient()
        samling = rag.oppna_samling()
        talare = rag.las_talare()
        dataset = dataversion(samling, talare, hybrid)
        if args.resume and run["dataset"] != dataset:
            raise ValueError("Dataset/index skiljer sig från den sparade körningen. Välj en ny --run-dir.")
        run["dataset"] = dataset
        if not args.resume:
            args.run_dir.mkdir(parents=True, exist_ok=False)
        spara_korning(run, args.run_dir)
        print(f"Resultat: {args.run_dir / 'results.json'}")
        print("Laddar sökmodellen ...")
        modell = rag.ladda_modell()
        kor_jobb(run, args.run_dir, klient, samling, modell, talare, index)
        report = bygg_rapport(run)
        (args.run_dir / "report.md").write_text(report, encoding="utf-8")
        print("\n" + report)
        return 1 if run["errors"] else 0
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"FEL: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
