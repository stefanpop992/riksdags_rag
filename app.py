#!/usr/bin/env python3
"""Streamlit-gränssnitt för att fråga om riksdagens anföranden.

All RAG-logik ligger i rag.py, samma modul som terminalversionen använder.

Startas så här:
    streamlit run app.py
"""

import streamlit as st

import bm25
import rag

st.set_page_config(page_title="Riksdagens anföranden", page_icon="🏛️", layout="wide",
                   initial_sidebar_state="expanded")

st.html("""
<style>
    .stMainBlockContainer { max-width: 1120px; padding-top: 2.5rem; padding-bottom: 4rem; }
    [data-testid="stSidebar"] h2 { font-family: sans-serif; font-size: 1.05rem; }
    h1 { letter-spacing: -0.035em; }
    h2, h3 { letter-spacing: -0.02em; }
    [data-testid="stForm"] { border-radius: 12px; padding: 1.5rem; }
    [data-testid="stExpander"] { border-radius: 8px; }
    .intro-label { color: #66839c; font-size: .75rem; font-weight: 700;
                   letter-spacing: .16em; margin-bottom: .5rem; }
    @media (max-width: 640px) {
        .stMainBlockContainer { padding-top: 1.25rem; }
        [data-testid="stForm"] { padding: 1rem; }
    }
</style>
""")


# @st.cache_resource kör funktionen EN gång och återanvänder resultatet i alla
# senare körningar av skriptet. Utan den skulle embeddingmodellen (2,2 GB)
# laddas om varje gång du rör en knapp.
@st.cache_resource(show_spinner="Laddar sökmodell och databas ...")
def ladda_allt():
    return (rag.oppna_samling(), rag.ladda_modell(), rag.skapa_klient(),
            rag.las_talare())


@st.cache_resource(show_spinner="Laddar BM25-index ...")
def ladda_bm25():
    """BM25-indexet, eller None om det inte är byggt än.

    Appen ska gå att starta utan indexet - det är 126 MB och byggs separat med
    `python bm25.py`. Saknas det stänger vi bara av hybridläget.
    """
    try:
        return bm25.ladda()
    except FileNotFoundError:
        return None


# @st.cache_data cachar returvärden i stället för objekt. Understrecket i
# _samling säger åt Streamlit att inte försöka hasha det argumentet.
@st.cache_data(show_spinner=False)
def filtervarden(_samling):
    return rag.las_filtervarden(_samling)


try:
    samling, modell, klient, alla_talare = ladda_allt()
except RuntimeError as fel:
    st.error(str(fel))
    st.stop()

bm25_index = ladda_bm25()
partier, (min_ar, max_ar) = filtervarden(samling)

# ---------------------------------------------------------------- sidopanel
with st.sidebar:
    st.markdown("### Avgränsa din sökning")
    st.caption("Välj hur du vill söka och vilka anföranden som ska ingå.")
    st.header("Sökläge")
    lage = st.radio(
        "Hur underlaget hämtas", ["Snabb överblick", "Fördjupad analys"], horizontal=True,
        captions=["En sökning med ett sammanfattat svar.",
                  "Flera sökningar och granskning. Tar längre tid och kostar mer."],
    )
    if lage == "Fördjupad analys":
        max_varv = st.slider("Max sökvarv", 1, 4, 3)
    else:
        max_varv = 1

    st.header("Filter")
    valt_parti = st.selectbox("Parti", ["Alla partier"] + partier)
    talarsokning = st.text_input("Talare", placeholder="t.ex. Kristersson",
                                 help="Efternamn räcker. Tomt = alla talare.")
    fran_ar, till_ar = st.select_slider(
        "Årtal",
        options=list(range(min_ar, max_ar + 1)),
        value=(min_ar, max_ar),
    )
    # Står reglaget på hela intervallet filtrerar årtalen inte bort något - men
    # ChromaDB vet inte det, utan går igenom all metadata för att komma fram
    # till samma svar. Det kostar en dryg sekund per sökning i stället för två
    # millisekunder. Vi skickar därför inga årtal alls när de täcker allt.
    if (fran_ar, till_ar) == (min_ar, max_ar):
        fran_ar = till_ar = None

    with st.expander("Avancerade sökinställningar"):
        antal = st.slider("Antal utdrag", 3, 15, 8,
                          disabled=lage == "Fördjupad analys",
                          help="Gäller snabb överblick. Fler utdrag ger bredare underlag.")
        per_anforande = st.slider("Utdrag per anförande", 1, 3, 1,
                                  help="Ett utdrag ger fler röster. Fler ger mer sammanhang.")
        hybrid = st.toggle(
            "Matcha både ord och betydelse", value=bm25_index is not None,
            disabled=bm25_index is None,
            help="Kombinerar exakta ordträffar med sökning på betydelse.")
        if bm25_index is None:
            st.caption("Ordsökning är inte tillgänglig. Sökning på betydelse är aktiv.")

    st.divider()
    st.caption(f"{samling.count():,} utdrag i databasen".replace(",", " "))

# ------------------------------------------------------------------- huvudyta
st.html('<div class="intro-label">RIKSDAGEN · UTFORSKA DEBATTEN</div>')
st.title("Vad har sagts i kammaren?")
st.markdown("Utforska riksdagens anföranden. Få en sammanfattning med källor "
            "och läs vad ledamöterna själva har sagt.")
st.caption(f"Anföranden {min_ar}–{max_ar} · {len(partier)} partier · Svar på svenska")


def valj_exempel(fraga):
    st.session_state.fragefalt = fraga


with st.form("sokformular"):
    fraga = st.text_input(
        "Vad vill du veta?", key="fragefalt",
        placeholder="Till exempel: Vad har partierna sagt om vinster i välfärden?").strip()
    sok_klickad = st.form_submit_button("Sök i anförandena", type="primary")
    st.caption("Svaren bygger på hittade utdrag. Du kan granska källorna under svaret.")

talarsokning = talarsokning.strip()
if sok_klickad and not fraga:
    st.warning("Skriv en fråga för att börja söka.")

if st.session_state.get("traffar") is None:
    st.markdown("#### Börja med en fråga")
    for kolumn, (etikett, exempel) in zip(st.columns(3), [
        ("Energi & klimat", "Vad har partierna sagt om kärnkraft?"),
        ("Vård & välfärd", "Vad har sagts om att korta vårdköerna?"),
        ("Skola & utbildning", "Vad har partierna sagt om friskolor?"),
    ]):
        kolumn.button(etikett, on_click=valj_exempel, args=(exempel,),
                      use_container_width=True)
    st.caption("Välj ett exempel, anpassa frågan och tryck på Sök i anförandena.")

# Streamlit kör om skriptet när ett filter ändras. Bara sökknappen får starta
# en sökning. Nyckeln kopplar det sparade svaret till rätt fråga och filter.
nyckel = (fraga, lage, max_varv, valt_parti, talarsokning, fran_ar, till_ar,
          antal, per_anforande, hybrid)

if sok_klickad and fraga:
    parti = None if valt_parti == "Alla partier" else valt_parti
    varianter = rag.talarvarianter(talarsokning, alla_talare)
    if talarsokning and not varianter:
        st.warning(f"Hittade ingen talare som matchar {talarsokning!r}.")
        st.stop()

    if lage == "Fördjupad analys":
        # st.status visar en logg som fylls på medan agenten arbetar. Vi skickar
        # in en callback som skriver in varje steg där.
        with st.status("Söker och granskar underlaget ...", expanded=True) as status:
            def visa_steg(steg):
                if steg["typ"] == "plan":
                    status.write(f"**Plan:** {len(steg['delfragor'])} delfrågor")
                    for d in steg["delfragor"]:
                        status.write(f"· {d.sokfraga}" + (f"  `{d.parti}`" if d.parti else ""))
                elif steg["typ"] == "sokning":
                    status.write(f"**Sökvarv {steg['varv']}:** {steg['nya']} nya utdrag, "
                                 f"{steg['totalt']} totalt")
                elif steg["typ"] == "bedomning":
                    status.write(f"**Granskning {steg['varv']}:** "
                                 + ("underlaget räcker" if steg["racker"] else "räcker inte"))
                    for s in steg["saknas"]:
                        status.write(f"· saknas: {s}")
                elif steg["typ"] == "stopp":
                    status.write(f"**Stopp:** {steg['skal']}")

            traffar, saknas, spar = rag.agentisk_sokning(
                klient, samling, modell, fraga, anvandar_parti=parti,
                fran_ar=fran_ar, till_ar=till_ar, max_varv=max_varv,
                per_anforande=per_anforande, pa_steg=visa_steg,
                alla_talare=alla_talare, anvandar_talare=talarsokning or None,
                bm25_index=bm25_index if hybrid else None)
            status.update(label=f"Granskning klar – {len(traffar)} utdrag", state="complete",
                          expanded=False)
        underlag = rag.bygg_underlag_med_luckor(traffar, saknas)
    else:
        where = rag.bygg_filter(parti, fran_ar, till_ar, varianter)
        with st.spinner("Söker i anförandena ..."):
            if hybrid:
                traffar = rag.sok_hybrid(samling, modell, fraga, bm25_index,
                                         antal, where, per_anforande)
            else:
                traffar = rag.sok(samling, modell, fraga, antal, where, per_anforande)
        saknas, spar = [], None
        underlag = rag.bygg_underlag(traffar)

    st.session_state.nyckel = nyckel
    st.session_state.traffar = traffar
    st.session_state.underlag = underlag
    st.session_state.saknas = saknas
    st.session_state.spar = spar
    st.session_state.svar = None  # None betyder "ska strömmas nu"

traffar = st.session_state.get("traffar")

if traffar is not None:
    sparad_fraga = st.session_state.nyckel[0]
    if st.session_state.nyckel != nyckel:
        st.info("Frågan eller filtren har ändrats. Klicka på Sök i anförandena "
                "för att uppdatera resultatet.")
    st.divider()
    st.markdown("### Sökresultat")
    st.write(sparad_fraga)
    if not traffar:
        st.warning("Sökningen gav inga träffar. Prova en bredare fråga eller "
                   "lossa på filtren i sidopanelen.")
    else:
        st.caption(f"{len(traffar)} källutdrag · {st.session_state.nyckel[1]}")
        st.subheader("Sammanfattning")
        if st.session_state.svar is None and sok_klickad:
            # Första gången strömmar vi svaret medan det skrivs, och sparar
            # den färdiga texten så att senare omkörningar slipper anropet.
            #
            # Innan första tecknet kommer tänker modellen i några sekunder. Vi
            # visar resonemanget i en ruta så länge, så att pausen syns vara
            # arbete och inte en hängd app. Rutan fälls ihop när svaret börjar.
            tankestatus = st.status("Claude tänker igenom underlaget ...", expanded=True)
            tankeruta = tankestatus.empty()
            tanke = []

            def visa_tanke(bit):
                tanke.append(bit)
                tankeruta.markdown("".join(tanke))

            st.session_state.svar = st.write_stream(
                rag.stromma_svar(klient, sparad_fraga, st.session_state.underlag,
                                 pa_tanke=visa_tanke))
            tankestatus.update(label="Claudes resonemang", state="complete",
                               expanded=False)
        elif st.session_state.svar is not None:
            st.markdown(st.session_state.svar)
        else:
            st.info("Svaret blev inte färdigt. Klicka på Sök i anförandena för att försöka igen.")

        if st.session_state.svar:
            st.download_button("Spara svaret", st.session_state.svar,
                               file_name="riksdagen-svar.md", mime="text/markdown")

        if st.session_state.get("spar"):
            with st.expander(f"Så gjordes sökningen ({len(st.session_state.spar)} steg)"):
                for steg in st.session_state.spar:
                    if steg["typ"] == "plan":
                        st.markdown(f"**Plan** – {len(steg['delfragor'])} delfrågor")
                        for d in steg["delfragor"]:
                            markering = f" `{d.parti}`" if d.parti else ""
                            st.markdown(f"- {d.sokfraga}{markering}  \n  *{d.motivering}*")
                    elif steg["typ"] == "sokning":
                        st.markdown(f"**Sökvarv {steg['varv']}** – {steg['nya']} nya, "
                                    f"{steg['totalt']} totalt")
                    elif steg["typ"] == "bedomning":
                        st.markdown(f"**Granskning {steg['varv']}** – "
                                    + ("underlaget räcker" if steg["racker"] else "räcker inte"))
                        for s in steg["saknas"]:
                            st.markdown(f"- saknas: {s}")
                    elif steg["typ"] == "stopp":
                        st.markdown(f"**Stopp** – {steg['skal']}")

        st.subheader(f"Källor ({len(traffar)} utdrag)")
        st.caption("Öppna ett utdrag för att läsa underlaget och gå vidare till hela anförandet.")
        for i, t in enumerate(traffar, 1):
            m = t["meta"]
            rubrik = f"{i}. {rag.formatera_talare(m)} · {m['datum']}"
            with st.expander(rubrik):
                vanster, hoger = st.columns([3, 1])
                with vanster:
                    st.markdown(f"**Debatt:** {m['debattrubrik']}")
                with hoger:
                    st.markdown(f"**Parti:** {m['parti'] or '–'}")
                if m.get("antal_chunkar", 1) > 1:
                    st.caption(f"Del {m['chunk_nr'] + 1} av {m['antal_chunkar']} "
                               f"ur anförandet")
                st.write(t["text"])
                if m.get("url"):
                    st.link_button("Läs hela anförandet hos riksdagen", m["url"])
