# Kammaren – Riksdagens anföranden

Fristående webbgränssnitt med strömmade svar, källutdrag, parti- och talarfilter,
årtal samt snabb och fördjupad sökning. Anpassat för mobil och dator.

Starta från projektmappen med befintlig virtuell miljö:

```bash
.venv/bin/python server.py
```

Öppna http://127.0.0.1:8000. Sökmodellen laddas när sidan ansluter till arkivet;
första starten kan ta en stund. Servern är avsedd för lokal användning.

Gränssnittet ligger i `web/` och använder HTML, CSS och JavaScript utan byggsteg.
`server.py` använder Pythons standardbibliotek och söklogiken i `rag.py`.
Samma förutsättningar som tidigare gäller: installerade `requirements.txt`,
CUDA för sökmodellen, befintlig `chroma_db` och `ANTHROPIC_API_KEY` i `.env`.
BM25-indexet är valfritt. Nyckeln används bara på servern.

Det tidigare Streamlit-gränssnittet kan fortfarande startas med
`.venv/bin/python -m streamlit run app.py`.
