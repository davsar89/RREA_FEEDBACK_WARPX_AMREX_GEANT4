#!/usr/bin/env node
// The French version of quiz/rrea-quiz.html. English is the source text; the quiz-fr block of the
// page maps each English string to its French translation.
//
//   OPENROUTER_API_KEY=... node quiz/tools/i18n.mjs   translate new, changed or invalid strings,
//                                                      drop unused ones, write the block
//
// Strings come from the content (I18N.texts), the literal tr('...') calls of the scripts, example
// titles and the data-tr / data-tr-attr marks of the static page. check.mjs imports load, needed
// and invalid from here, so a page whose translation is incomplete or broken fails its checks.
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';

export const PAGE = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..', 'rrea-quiz.html');
const read = () => fs.readFileSync(PAGE, 'utf8');
const block = (html, id) => html.match(new RegExp(`<script id="${id}" type="application/json">([\\s\\S]*?)</script>`))?.[1];
const scripts = (html) => [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map((m) => m[1]);

/** Evaluate the language, content and figure scripts in one language with a minimal DOM stub. */
export function load(lang = 'en', html = read()) {
  const window = {
    URL, URLSearchParams,
    matchMedia: () => ({ matches: false, addEventListener() {} }),
    location: { search: `?lang=${lang}` }, navigator: { language: 'en' }, localStorage: { getItem: () => null },
    document: { documentElement: {}, querySelectorAll: () => [], getElementById: (id) => { const t = block(html, id); return t == null ? null : { textContent: t }; } },
  };
  window.window = window;
  const ctx = vm.createContext(window);
  for (const key of ['window.I18N =', 'window.QUIZ =', 'window.QFIG =']) vm.runInContext(scripts(html).find((s) => s.includes(key)), ctx);
  return { Q: window.QUIZ, F: window.QFIG, I: window.I18N, GEN: JSON.parse(block(html, 'quiz-generated')) };
}

/** Every string the French page looks up, in page order, mapped to a context label for the translator. */
export function needed(html = read()) {
  const { Q, I, GEN } = load('en', html);
  const out = new Map();
  const add = (s, ctx) => { if (s && !out.has(s)) out.set(s, ctx); };
  for (const s of Q.sections) {
    for (const q of s.questions) q.opts.forEach((o) => add(o, `${OPTIONS} of question ${q.id} in lesson "${s.title}"`));
    I.texts(s).forEach((t) => add(t, `lesson "${s.title}"`));
  }
  const GLOSSARY = 'glossary: terms, their definitions, and aliases, which are the words that mark a term where it occurs in the course text, so translate each alias exactly as the course text words it';
  for (const [k, v] of Object.entries(Q)) if (k !== 'sections') I.texts(v).forEach((t) => add(t, k === 'glossary' ? GLOSSARY : k));
  const literal = (src) => [...src.matchAll(/\btr\('((?:[^'\\]|\\.)*)'/g)].map((m) => vm.runInNewContext(`'${m[1]}'`));
  for (const src of scripts(html)) literal(src).forEach((t) => add(t, src.includes('window.QFIG =') ? 'figure or lab (labels have little room)' : 'interface'));
  Object.values(GEN.snippets).forEach((s) => add(s.title, 'title of a runnable Python example'));
  const shell = html.slice(html.indexOf('<body>'), html.indexOf('<!-- generated:begin -->'));
  for (const m of shell.matchAll(/<(\w+)\b[^>]*\sdata-tr(?=[\s>])[^>]*>([\s\S]*?)<\/\1>/g)) add(m[2].trim(), 'interface');
  for (const m of shell.matchAll(/<[^>]*\sdata-tr-attr="([^"]+)"[^>]*>/g)) for (const a of m[1].split(' ')) add(m[0].match(new RegExp(`\\s${a}="([^"]*)"`))[1], 'interface');
  return out;
}

// Answer options must not differ in length because of the translation, or length would hint at the
// answer: an option that grew clearly more than the median of its question is redone within a budget.
const OPTIONS = 'the four answer options';
function overgrown(items, dict) {
  const n = (s) => s.replace(/<[^>]+>/g, '').length;
  const e = items.map((en) => n(dict[en]) / n(en)), m = [...e].sort((a, b) => a - b), mid = (m[1] + m[2]) / 2;
  return items.filter((en) => n(dict[en]) - mid * n(en) > Math.max(3, 0.06 * n(en))).map((en) => [en,
    `longer than the other options of this question: say it in at most ${Math.floor(n(en) * (mid + 0.03))} characters of text, tags not counted`, dict[en]]);
}

const VERBATIM = /<(code|kbd|samp|math)\b[^>]*>[\s\S]*?<\/\1>/g;
const same = (a, b) => a.sort().join('\u0000') === b.sort().join('\u0000');
const numbers = (s) => s.replace(VERBATIM, ' ').replace(/<[^>]+>|\{\d+\}|&#?\w+;/g, ' ').replace(/(\d),(\d)/g, '$1.$2').match(/\d+(?:\.\d+)*/g) || [];

/** Why a translation is unusable, or null: it must keep the markup, code, math, placeholders and numbers. */
export function invalid(en, fr) {
  if (typeof fr !== 'string' || !fr.trim()) return 'empty';
  if (fr.includes('\u2014')) return 'contains an em dash';
  if (!same(en.match(VERBATIM) || [], fr.match(VERBATIM) || [])) return 'the content of <code>, <kbd>, <samp> or <math> changed';
  const tags = (s) => s.replace(VERBATIM, '').match(/<[^>]+>/g) || [];
  if (!same(tags(en), tags(fr))) return `the HTML tags changed (${tags(en).join('') || 'none'} expected)`;
  if (!same(en.match(/\{\d+\}/g) || [], fr.match(/\{\d+\}/g) || [])) return 'the {n} placeholders changed';
  if (!same(numbers(en), numbers(fr))) return `the numbers changed: expected ${numbers(en).join(' ') || 'none'}, got ${numbers(fr).join(' ') || 'none'}`;
  return null;
}

// ------------------------------------------------------------------ translation
const MODEL = 'google/gemini-4-argon';
const SYSTEM = `You translate an interactive course from English into French. The course teaches relativistic runaway electron avalanches (RREA) and their simulation with the WarpX particle-in-cell code and the AMReX framework. Readers are French-speaking physics students and researchers. Write clear, simple, natural French, as in a good university course, addressing the reader as "vous". Translate the meaning rather than word by word, and keep every technical statement exactly as precise as the English.

The input is a JSON object of English strings with a context line. Return a JSON object with the same keys whose values are the French translations.

Keep unchanged:
- every HTML or MathML tag with its attributes, and the whole content of <code>, <kbd>, <samp> and <math> elements;
- the placeholders {0}, {1}, ... (each exactly as often as in the English; move them where French grammar needs them);
- names of software, classes, functions, variables, files, input parameters and options (WarpX, AMReX, MultiFab, BoxArray, ParticleContainer, MFIter, ParallelFor, Geant4, ...), acronyms (RREA, TGF, PIC, MPI, GPU, CFL, PML, MLMG, PSATD, ESTAR, PARMA, ...), author names, chemical, unit and mathematical symbols;
- every number as a numeral with the same digits. Add no numeral that the English lacks (English "two" stays "deux").

French conventions:
- decimal comma in numbers in the text (1.293 kg/m³ becomes 1,293 kg/m³), never inside code or math; never add thousands separators (8192 stays 8192);
- a no-break space (U+00A0) before : ; ! ? and inside « »; the apostrophe ’;
- never an em dash: use a colon, a comma or parentheses. En dashes in numeric ranges stay;
- the four answer options of a question are translated equally tightly: each French option about as much longer than its English as the others, so that no option, the correct one included, stands out by its length. Prefer compact turns (« donc » rather than « de sorte que »);
- sentence case for titles, labels and buttons, which stay short. Labels of figures and labs have little room: keep them about as long as the English, abbreviating if needed.

Terminology, used consistently: runaway electron: électron runaway; relativistic feedback: rétroaction relativiste; terrestrial gamma-ray flash: flash gamma terrestre; drag: force de freinage; threshold field: champ seuil; avalanche length: longueur d’avalanche; mesh: maillage; cell: cellule; ghost cells: cellules fantômes; box: boîte (but the class Box stays Box); tile: tuile; rank: rang; deposit: dépôt; gather: interpolation aux particules (gather); push: poussée; time step: pas de temps; solver: solveur; hook: hook; overlay: overlay; chord: corde; optical depth: profondeur optique; cross section: section efficace; mean free path: libre parcours moyen; stopping power: pouvoir d’arrêt; pair production: production de paires; attachment: attachement; quenching: quenching; seed: germe; macroparticle: macroparticule; weight: poids; resampling: rééchantillonnage; thinning: élagage; checkpoint: checkpoint; restart: reprise; input deck: fichier d’entrée; smoke test: smoke test; fail-closed: sûreté intégrée (fail-closed); lesson: leçon; wrap-up: bilan; study sheet: fiche de révision; hint: indice; glossary: glossaire; mixed practice: entraînement mixte.`;

async function complete(context, items, notes) {
  const ids = Object.keys(items);
  const user = `Context: ${context}.${notes ? `\nAn earlier translation of some of these strings was rejected; fix these problems:\n${notes}` : ''}\n\n${JSON.stringify(items, null, 1)}`;
  const body = {
    model: MODEL, temperature: 0.2, reasoning: { effort: 'low' }, usage: { include: true },
    messages: [{ role: 'system', content: SYSTEM }, { role: 'user', content: user }],
    response_format: { type: 'json_schema', json_schema: { name: 'translation', strict: true, schema: {
      type: 'object', properties: Object.fromEntries(ids.map((k) => [k, { type: 'string' }])), required: ids, additionalProperties: false } } },
  };
  for (let attempt = 1; ; attempt++) {
    try {
      const r = await fetch('https://openrouter.ai/api/v1/chat/completions', {
        method: 'POST', headers: { Authorization: `Bearer ${process.env.OPENROUTER_API_KEY}`, 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      const j = await r.json();
      if (!r.ok || j.error) throw new Error(`${r.status} ${JSON.stringify(j.error || j).slice(0, 200)}`);
      const text = j.choices[0].message.content.replace(/^```(?:json)?\s*|\s*```$/g, '');
      return { out: JSON.parse(text), usage: j.usage };
    } catch (e) {
      if (attempt >= 4) throw e;
      console.log(`  retry ${attempt} (${e.message.slice(0, 120)})`);
      await new Promise((r) => setTimeout(r, 5000 * attempt));
    }
  }
}

function write(dict) {
  const sorted = Object.fromEntries(Object.entries(dict).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)));
  const json = JSON.stringify(sorted).replace(/<\//g, '<\\/');
  const html = read();
  fs.writeFileSync(PAGE, html.replace(/(<script id="quiz-fr" type="application\/json">)[\s\S]*?(<\/script>)/, (m, a, b) => `${a}${json}${b}`));
}

async function translate() {
  if (!process.env.OPENROUTER_API_KEY) { console.error('OPENROUTER_API_KEY is not set'); process.exit(2); }
  const want = needed();
  const old = JSON.parse(block(read(), 'quiz-fr'));
  const dict = Object.fromEntries(Object.entries(old).filter(([en, fr]) => want.has(en) && !invalid(en, fr)));
  const todo = [...want.keys()].filter((en) => !Object.hasOwn(dict, en));
  console.log(`${want.size} strings: ${Object.keys(dict).length} kept, ${Object.keys(old).length - Object.keys(dict).length} dropped, ${todo.length} to translate with ${MODEL}`);
  // Batches of consecutive strings from one context, so a lesson is translated together.
  const batches = [];
  for (const en of todo) {
    const last = batches.at(-1), ctx = want.get(en);
    if (!last || last.ctx !== ctx || last.items.length >= 30 || last.chars + en.length > 7000) batches.push({ ctx, items: [], chars: 0 });
    batches.at(-1).items.push(en); batches.at(-1).chars += en.length;
  }
  let cost = 0, tokIn = 0, tokOut = 0, done = 0;
  const failed = [];
  async function run(b) {
    let pending = b.items, notes = '';
    for (let round = 1; round <= 3 && pending.length; round++) {
      const items = Object.fromEntries(pending.map((en, i) => [`s${i}`, en]));
      const { out, usage } = await complete(b.ctx, items, notes);
      cost += usage?.cost || 0; tokIn += usage?.prompt_tokens || 0; tokOut += usage?.completion_tokens || 0;
      const bad = [];
      for (const [id, en] of Object.entries(items)) {
        const why = invalid(en, out[id]);
        if (why) bad.push([en, why, out[id]]); else dict[en] = out[id];
      }
      if (round === 1 && b.ctx.startsWith(OPTIONS) && !bad.length) bad.push(...overgrown(b.items, dict));
      pending = bad.map(([en]) => en);
      notes = bad.map(([, why, fr], i) => `s${i}: ${why}; rejected translation: ${JSON.stringify(fr)}`).join('\n');
    }
    pending = pending.filter((en) => !Object.hasOwn(dict, en));  // an option still long keeps its valid translation
    failed.push(...pending);
    write(dict);
    console.log(`batch ${++done}/${batches.length} (${b.ctx}, ${b.items.length} strings${pending.length ? `, ${pending.length} failed` : ''}); $${cost.toFixed(3)} so far`);
  }
  const queue = [...batches];
  await Promise.all(Array.from({ length: 6 }, async () => { for (let b; (b = queue.shift());) await run(b); }));
  write(dict);
  console.log(`done: ${tokIn} prompt and ${tokOut} completion tokens, $${cost.toFixed(3)}`);
  if (failed.length) { console.log(`${failed.length} strings still fail validation; rerun to retry:\n${failed.map((s) => `  ${s.slice(0, 100)}`).join('\n')}`); process.exit(1); }
}

if (process.argv[1] && path.resolve(process.argv[1]) === new URL(import.meta.url).pathname) await translate();
