const $ = id => document.getElementById(id);
let answer = '', sources = [], savedQuestion = '', busy = false;
const mode = () => document.querySelector('input[name=mode]:checked').value;
function error(message) { $('error').textContent = message; $('error').hidden = !message; }
function updateMode() {
  const deep = mode() === 'deep';
  $('count').disabled = deep;
  $('rounds').disabled = !deep;
  $('mode-help').textContent = deep ? 'Flera sökningar och granskning av underlaget. Tar längre tid och kostar mer.' : 'En sökning med ett sammanfattat svar och källor.';
}
document.querySelectorAll('input[name=mode]').forEach(input => input.addEventListener('change', updateMode));
document.querySelectorAll('[data-question]').forEach(button => button.addEventListener('click', () => {
  if (busy) return;
  $('question').value = button.dataset.question;
  $('question').focus();
}));
$('reset').addEventListener('click', () => {
  ['party', 'speaker', 'from-year', 'to-year'].forEach(id => $(id).value = '');
});
$('question').addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    if (!$('submit').disabled) $('search-form').requestSubmit();
  }
});
function renderSources(items) {
  sources = items;
  $('source-count').textContent = items.length;
  $('sources').replaceChildren();
  items.forEach((source, i) => {
    const details = document.createElement('details'); details.className = 'source';
    const summary = document.createElement('summary'); summary.textContent = `${i + 1}. ${source.speaker}`;
    const date = document.createElement('time'); date.textContent = source.date; summary.append(date);
    const title = document.createElement('p'); const strong = document.createElement('strong'); strong.textContent = source.title; title.append(strong);
    const text = document.createElement('p'); text.textContent = source.text;
    details.append(summary, title, text);
    try {
      const url = new URL(source.url);
      if (['https:', 'http:'].includes(url.protocol)) {
        const link = document.createElement('a'); link.href = url.href; link.target = '_blank'; link.rel = 'noopener noreferrer'; link.textContent = 'Läs hela anförandet ↗'; details.append(link);
      }
    } catch { /* Sources without a URL still display their full excerpt. */ }
    $('sources').append(details);
  });
  if (!items.length) $('answer').textContent = 'Inga anföranden matchade sökningen. Prova en bredare fråga eller ta bort något filter.';
}
$('search-form').addEventListener('submit', async event => {
  event.preventDefault();
  if (busy) return;
  const question = $('question').value.trim();
  if (!question) { error('Skriv en fråga för att börja söka.'); return; }
  const from = Number($('from-year').value) || null, to = Number($('to-year').value) || null;
  if (from && to && from > to) { error('Startåret måste vara före slutåret.'); return; }
  const body = {question, mode: mode(), party: $('party').value, speaker: $('speaker').value.trim(), from_year: from, to_year: to,
    count: Number($('count').value), per_speech: Number($('per-speech').value), rounds: Number($('rounds').value), hybrid: $('hybrid').checked};
  error(''); busy = true; $('submit').disabled = true; $('submit').textContent = 'Söker …';
  answer = ''; sources = []; savedQuestion = question;
  $('answer').textContent = 'Hämtar underlag till ditt svar …'; $('sources').replaceChildren(); $('source-count').textContent = '0';
  $('results').hidden = false; $('inspiration').hidden = true; $('download').hidden = true;
  $('result-question').textContent = question;
  $('result-context').textContent = [mode() === 'deep' ? 'Fördjupad analys' : 'Snabb överblick', body.party || 'Alla partier', body.speaker || 'Alla talare', `${from || 'Alla år'}${to ? ' – ' + to : ''}`].join(' · ');
  $('status').textContent = 'Förbereder sökningen …'; $('results').setAttribute('aria-busy', 'true');
  let done = false;
  try {
    const response = await fetch('/api/search', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
    if (!response.ok) { const info = await response.json(); throw new Error(info.error || 'Sökningen kunde inte startas.'); }
    const reader = response.body.getReader(), decoder = new TextDecoder();
    let buffer = '';
    const receive = line => {
      if (!line.trim()) return;
      const event = JSON.parse(line);
      if (event.type === 'error') throw new Error(event.data);
      if (event.type === 'status') $('status').textContent = event.data;
      if (event.type === 'sources') renderSources(event.data);
      if (event.type === 'token') { answer += event.data; $('answer').textContent = answer; }
      if (event.type === 'done') { done = true; $('status').textContent = sources.length ? `Klart · ${sources.length} källutdrag` : 'Sökningen är klar · Inga träffar'; }
    };
    while (true) {
      const chunk = await reader.read();
      buffer += decoder.decode(chunk.value || new Uint8Array(), {stream: !chunk.done});
      const lines = buffer.split('\n'); buffer = lines.pop(); lines.forEach(receive);
      if (chunk.done) break;
    }
    if (buffer.trim()) receive(buffer);
    if (!done) throw new Error('Anslutningen avbröts. Försök igen.');
    $('download').hidden = !answer;
  } catch (err) {
    error(err.message); $('status').textContent = 'Sökningen avbröts. Du kan försöka igen.';
    if (!answer) $('answer').textContent = 'Inget färdigt svar. Se meddelandet ovan.';
    else $('status').textContent += ' Svaret nedan är ofullständigt.';
  } finally {
    busy = false; $('submit').disabled = false; $('submit').textContent = 'Sök i anförandena ↗'; $('results').setAttribute('aria-busy', 'false');
  }
});
$('download').addEventListener('click', () => {
  const text = `# ${savedQuestion}\n\n${answer}\n\n## Källor\n\n` + sources.map((s, i) => `${i + 1}. ${s.speaker} · ${s.date}\n${s.title}\n${s.url}`).join('\n\n');
  const url = URL.createObjectURL(new Blob([text], {type: 'text/markdown;charset=utf-8'}));
  const link = document.createElement('a'); link.href = url; link.download = 'kammaren-svar.md'; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
});
async function initialize() {
  try {
    const response = await fetch('/api/config'); const config = await response.json();
    if (!response.ok) throw new Error(config.error);
    config.parties.forEach(p => $('party').add(new Option(p, p)));
    for (let year = config.years[0]; year <= config.years[1]; year++) {
      ['from-year', 'to-year'].forEach(id => $(id).add(new Option(year, year)));
    }
    $('hybrid').checked = config.hybrid; $('hybrid').disabled = !config.hybrid;
    if (!config.hybrid) $('hybrid').parentElement.title = 'Ordsökning är inte tillgänglig. Sökning på betydelse används.';
    $('database').textContent = `${config.count.toLocaleString('sv-SE')} utdrag · ${config.years.join('–')}`;
    $('submit').disabled = false;
  } catch (err) {
    $('database').textContent = 'Arkivet är inte tillgängligt'; error(`${err.message || 'Kunde inte ansluta.'} Ladda om sidan för att försöka igen.`);
  }
}
initialize();
