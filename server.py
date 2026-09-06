#!/usr/bin/env python3
"""Standalone local web interface. Run: .venv/bin/python server.py"""
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).resolve().parent
LOCK = Lock()
RESOURCES = None


def resources():
    global RESOURCES
    if RESOURCES is None:
        import rag
        import bm25
        collection = rag.oppna_samling()
        client = rag.skapa_klient()
        model = rag.ladda_modell()
        speakers = rag.las_talare()
        try:
            index = bm25.ladda()
        except FileNotFoundError:
            index = None
        parties, years = rag.las_filtervarden(collection)
        RESOURCES = (rag, collection, client, model, speakers, index,
                     dict(parties=parties, years=years, count=collection.count(), hybrid=index is not None))
    return RESOURCES


def validate(data):
    if not isinstance(data, dict):
        raise ValueError('Ogiltig sökning.')
    question = data.get('question', '')
    if not isinstance(question, str) or not 1 <= len(question.strip()) <= 4000:
        raise ValueError('Skriv en fråga med högst 4 000 tecken.')
    data['question'] = question.strip()
    if data.get('mode') not in ('quick', 'deep'):
        raise ValueError('Välj ett giltigt sökläge.')
    if data.get('party') not in (None, '', 'S', 'M', 'SD', 'C', 'V', 'KD', 'L', 'MP'):
        raise ValueError('Välj ett giltigt parti.')
    if not isinstance(data.get('speaker', ''), str) or len(data.get('speaker', '')) > 200:
        raise ValueError('Ogiltigt talarnamn.')
    for key, low, high, default in [('count', 3, 15, 8), ('per_speech', 1, 3, 1), ('rounds', 1, 4, 3)]:
        value = data.get(key, default)
        if type(value) is not int or not low <= value <= high:
            raise ValueError('Ogiltiga sökinställningar.')
        data[key] = value
    for key in ('from_year', 'to_year'):
        if data.get(key) is not None and (type(data[key]) is not int or not 1900 <= data[key] <= 2100):
            raise ValueError('Ogiltigt årtal.')
    if data.get('from_year') and data.get('to_year') and data['from_year'] > data['to_year']:
        raise ValueError('Startåret måste vara före slutåret.')
    if type(data.get('hybrid', True)) is not bool:
        raise ValueError('Ogiltig sökinställning.')
    return data


def search(data, emit):
    rag, collection, client, model, speakers, index, _ = resources()
    speaker = data.get('speaker', '').strip()
    variants = rag.talarvarianter(speaker, speakers)
    if speaker and not variants:
        raise ValueError(f'Ingen talare matchar ”{speaker}”. Prova ett efternamn.')
    index = index if data.get('hybrid', True) else None
    party = data.get('party') or None
    start, end = data.get('from_year'), data.get('to_year')
    emit('status', 'Söker i riksdagens anföranden …')
    if data['mode'] == 'deep':
        def progress(step):
            labels = {'plan': 'Planerar sökningen …', 'sokning': 'Hämtar fler källor …',
                      'bedomning': 'Granskar underlaget …', 'stopp': 'Sökningen är klar.'}
            emit('status', labels.get(step['typ'], 'Arbetar …'))
        hits, missing, _ = rag.agentisk_sokning(
            client, collection, model, data['question'], anvandar_parti=party,
            fran_ar=start, till_ar=end, max_varv=data['rounds'],
            per_anforande=data['per_speech'], pa_steg=progress,
            alla_talare=speakers, anvandar_talare=speaker or None, bm25_index=index)
        context = rag.bygg_underlag_med_luckor(hits, missing)
    else:
        where = rag.bygg_filter(party, start, end, variants)
        args = (collection, model, data['question'])
        hits = (rag.sok_hybrid(*args, index, data['count'], where, data['per_speech'])
                if index is not None else rag.sok(*args, data['count'], where, data['per_speech']))
        context = rag.bygg_underlag(hits)
    emit('sources', [dict(speaker=rag.formatera_talare(h['meta']),
                          date=h['meta']['datum'], title=h['meta']['debattrubrik'],
                          text=h['text'], url=h['meta'].get('url', '')) for h in hits])
    if hits:
        emit('status', 'Skriver en sammanfattning …')
        for chunk in rag.stromma_svar(client, data['question'], context):
            emit('token', chunk)
    emit('done', None)


class Handler(BaseHTTPRequestHandler):
    def reply(self, code, body, mime='application/json; charset=utf-8'):
        payload = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == '/api/config':
            try:
                with LOCK:
                    config = resources()[-1]
                self.reply(200, json.dumps(config))
            except Exception:
                logging.exception('Could not initialize search')
                self.reply(503, json.dumps({'error': 'Sökningen kunde inte startas. Kontrollera serverloggen och försök igen.'}))
            return
        files = {'/': ('index.html', 'text/html'), '/app.js': ('app.js', 'text/javascript'),
                 '/style.css': ('style.css', 'text/css')}
        if self.path not in files:
            self.reply(404, '{}')
            return
        name, mime = files[self.path]
        self.reply(200, (ROOT / 'web' / name).read_bytes(), mime + '; charset=utf-8')

    def do_POST(self):
        if self.path != '/api/search':
            self.reply(404, '{}')
            return
        # This local application only accepts same-origin JSON requests.
        if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
            self.reply(415, '{}')
            return
        origin = self.headers.get('Origin')
        if origin and origin != 'http://' + self.headers.get('Host', ''):
            self.reply(403, '{}')
            return
        try:
            size = int(self.headers.get('Content-Length', 0))
            if not 0 < size <= 20000:
                raise ValueError('För stor eller tom förfrågan.')
            data = validate(json.loads(self.rfile.read(size)))
        except (ValueError, UnicodeError) as error:
            self.reply(400, json.dumps({'error': str(error)}))
            return
        if not LOCK.acquire(blocking=False):
            self.reply(409, json.dumps({'error': 'En sökning pågår. Försök igen om en stund.'}))
            return
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'application/x-ndjson; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            def emit(kind, value):
                self.wfile.write((json.dumps({'type': kind, 'data': value}) + '\n').encode())
                self.wfile.flush()
            try:
                search(data, emit)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as error:
                logging.exception('Search failed')
                emit('error', str(error) if isinstance(error, ValueError) else
                     'Sökningen kunde inte slutföras. Försök igen eller kontrollera serverloggen.')
        finally:
            LOCK.release()


if __name__ == '__main__':
    import os
    os.chdir(ROOT)
    # Visa både loopback-adressen och datorns adress på det lokala nätverket,
    # så att appen går att öppna från t.ex. en telefon på samma WiFi.
    import socket
    # Knep: "koppla upp" en UDP-socket mot en adress ute på internet. Inga paket
    # skickas, men kärnan väljer vilket nätverkskort som skulle använts - och
    # dess IP är datorns adress på det lokala nätverket.
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(('8.8.8.8', 80))
        lan = probe.getsockname()[0]
    except OSError:
        lan = '127.0.0.1'   # ingen nätverksanslutning
    finally:
        probe.close()
    print('Öppna http://127.0.0.1:8000', flush=True)
    print(f'Från telefon på samma nätverk: http://{lan}:8000', flush=True)
    ThreadingHTTPServer(('0.0.0.0', 8000), Handler).serve_forever()
