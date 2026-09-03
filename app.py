#!/usr/bin/env python3
"""Streamlit-gränssnitt för att fråga om riksdagens anföranden.

All RAG-logik ligger i rag.py, samma modul som terminalversionen använder.

Startas så här:
    streamlit run app.py
"""

import streamlit as st

import rag

st.set_page_config(page_title="Riksdagens anföranden", page_icon="🏛️", layout="wide")


# @st.cache_resource kör funktionen EN gång och återanvänder resultatet i alla
# senare körningar av skriptet. Utan den skulle embeddingmodellen (2,2 GB)
# laddas om varje gång du rör en knapp.
@st.cache_resource(show_spinner="Laddar sökmodell och databas ...")
def ladda_allt():
    return (rag.oppna_samling(), rag.ladda_modell(), rag.skapa_klient(),
            rag.las_talare())


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

partier, (min_ar, max_ar) = filtervarden(samling)

# ---------------------------------------------------------------- sidopanel
with st.sidebar:
    st.header("Läge")
    lage = st.radio(
        "Hur underlaget hämtas", ["Naivt", "Agentiskt"], horizontal=True,
        captions=["En sökning, ett svar. Snabbt.",
                  "Modellen planerar, granskar och söker om. Långsammare och dyrare."],
    )
    if lage == "Agentiskt":
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

    st.header("Sökning")
    antal = st.slider("Antal utdrag", 3, 15, 8,
                      help="Fler utdrag ger bredare underlag men dyrare anrop.")
    per_anforande = st.slider("Utdrag per anförande", 1, 3, 1,
                              help="1 ger bredd, högre ger djup i enskilda tal.")

    st.divider()
    st.caption(f"{samling.count():,} utdrag i databasen".replace(",", " "))

# ------------------------------------------------------------------- huvudyta
st.title("🏛️ Riksdagens anföranden")
st.caption("Ställ en fråga om vad som sagts i kammaren. Svaren bygger enbart "
           "på de utdrag som visas längst ned, och varje påstående ska ha en källa.")

fraga = st.text_input("Din fråga",
                      placeholder="Vad har partierna sagt om vinster i välfärden?")

# Streamlit kör om hela skriptet vid varje interaktion - även när du bara
# fäller ut en källruta. Nyckeln nedan låter oss se om något faktiskt ändrats,
# så att vi inte betalar för ett nytt API-anrop i onödan.
nyckel = (fraga, lage, max_varv, valt_parti, talarsokning, fran_ar, till_ar, antal, per_anforande)

if fraga and st.session_state.get("nyckel") != nyckel:
    parti = None if valt_parti == "Alla partier" else valt_parti
    varianter = rag.talarvarianter(talarsokning, alla_talare)
    if talarsokning and not varianter:
        st.warning(f"Hittade ingen talare som matchar {talarsokning!r}.")
        st.stop()

    if lage == "Agentiskt":
        # st.status visar en logg som fylls på medan agenten arbetar. Vi skickar
        # in en callback som skriver in varje steg där.
        with st.status("Agenten arbetar ...", expanded=True) as status:
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
                alla_talare=alla_talare, anvandar_talare=talarsokning or None)
            status.update(label=f"Agenten klar – {len(traffar)} utdrag", state="complete",
                          expanded=False)
        underlag = rag.bygg_underlag_med_luckor(traffar, saknas)
    else:
        where = rag.bygg_filter(parti, fran_ar, till_ar, varianter)
        with st.spinner("Söker i anförandena ..."):
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

if fraga and traffar is not None:
    if not traffar:
        st.warning("Sökningen gav inga träffar. Prova en bredare fråga eller "
                   "lossa på filtren i sidopanelen.")
    else:
        st.subheader("Svar")
        if st.session_state.svar is None:
            # Första gången strömmar vi svaret medan det skrivs, och sparar
            # den färdiga texten så att senare omkörningar slipper anropet.
            st.session_state.svar = st.write_stream(
                rag.stromma_svar(klient, fraga, st.session_state.underlag))
        else:
            st.markdown(st.session_state.svar)

        if st.session_state.get("spar"):
            with st.expander(f"Agentens arbete ({len(st.session_state.spar)} steg)"):
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
        for i, t in enumerate(traffar, 1):
            m = t["meta"]
            rubrik = (f"{i}. {rag.formatera_talare(m)} · {m['datum']} · "
                      f"likhet {t['likhet']:.3f}")
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
