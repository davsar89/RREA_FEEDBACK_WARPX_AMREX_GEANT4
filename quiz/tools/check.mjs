#!/usr/bin/env node
// Checks for quiz/rrea-quiz.html. Exit status 1 on any failure.
//
//   node quiz/tools/check.mjs content          structure, references, house style, French translation,
//                                               answer-cue statistics
//   node quiz/tools/check.mjs browser [outdir]  offline headless Chrome, in English then French: every
//                                               route, fonts, labs, behaviour, 16 layouts, screenshots,
//                                               study sheet PDF
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawn, spawnSync } from 'node:child_process';
import { load, needed, invalid } from './i18n.mjs';

const HERE = path.dirname(new URL(import.meta.url).pathname);
const QUIZ_DIR = path.resolve(HERE, '..');
const PAGE = path.join(QUIZ_DIR, 'rrea-quiz.html');
const html = fs.readFileSync(PAGE, 'utf8');
const errs = [];
const err = (m) => errs.push(m);

// ------------------------------------------------------------------ content
function content() {
  const { Q, F, GEN } = load();
  const KINDS = ['phys', 'num', 'fw', 'arch'];
  for (const f of ['rrea-quiz.html', 'README.md', 'tools/build.py', 'tools/examples.py', 'tools/check.mjs', 'tools/i18n.mjs']) {
    const p = path.join(QUIZ_DIR, f);
    if (!fs.existsSync(p)) continue;
    fs.readFileSync(p, 'utf8').split('\n').forEach((l, i) => { if (l.includes('\u2014')) err(`em dash in quiz/${f}:${i + 1}`); });
  }
  const balanced = (where, txt) => {
    if (typeof txt !== 'string') return;
    for (const t of ['code', 'sub', 'sup', 'em', 'strong', 'b', 'i', 'p', 'math', 'mrow', 'mfrac', 'span', 'a', 'kbd']) {
      const o = (txt.match(new RegExp(`<${t}[\\s>]`, 'g')) || []).length, c = (txt.match(new RegExp(`</${t}>`, 'g')) || []).length;
      if (o !== c) err(`${where}: unbalanced <${t}> (${o} open, ${c} close)`);
    }
  };
  const ex = new Set(Object.keys(GEN.excerpts)), sn = new Set(Object.keys(GEN.snippets));
  const ids = new Set(), lv = {}, kinds = {};
  const all = [];
  for (const s of Q.sections) {
    for (const f of ['id', 'title', 'hue', 'blurb', 'objectives', 'intro', 'takeaways', 'reflect', 'questions']) if (s[f] === undefined) err(`${s.id}: missing ${f}`);
    if (s.objectives?.length !== 3) err(`${s.id}: ${s.objectives?.length} objectives, expected 3`);
    if (!s.reflect?.q || !s.reflect?.a) err(`${s.id}: reflection incomplete`);
    for (const id of s.figs || []) if (!F.FIG[id] || F.FIG[id].quizOnly) err(`${s.id}: figure ${id} missing or quiz-only`);
    if (s.lab && !F.LABS[s.lab]) err(`${s.id}: unknown lab ${s.lab}`);
    if (s.snippet && !sn.has(s.snippet)) err(`${s.id}: unknown example ${s.snippet}`);
    for (const e of s.eqs || []) { balanced(`${s.id} equation`, e.m); balanced(`${s.id} equation note`, e.note); }
    for (const t of [s.intro, s.blurb, ...(s.takeaways || []), ...(s.objectives || []), s.reflect?.q, s.reflect?.a]) balanced(`${s.id} text`, t);
    for (const q of s.questions) {
      all.push(q);
      lv[q.lvl] = (lv[q.lvl] || 0) + 1; kinds[q.kind] = (kinds[q.kind] || 0) + 1;
      if (ids.has(q.id)) err(`duplicate id ${q.id}`);
      ids.add(q.id);
      for (const f of ['q', 'opts', 'why', 'hint', 'exp', 'refs', 'lvl', 'kind']) if (q[f] === undefined) err(`${q.id}: missing ${f}`);
      if (!KINDS.includes(q.kind)) err(`${q.id}: unknown kind ${q.kind}`);
      if (![1, 2, 3].includes(q.lvl)) err(`${q.id}: level ${q.lvl}`);
      if (q.opts?.length !== 4 || new Set(q.opts).size !== 4) err(`${q.id}: needs 4 distinct options`);
      if (q.why?.length !== q.opts?.length || q.why[0] !== '' || q.why.slice(1).some((w) => !w?.trim())) err(`${q.id}: why must be ['' and one reason per wrong option]`);
      if (!q.refs?.length) err(`${q.id}: no refs`);
      for (const r of [...(q.refs || []), ...(q.excerpt ? [q.excerpt] : [])]) if (!ex.has(r)) err(`${q.id}: ${r} has no excerpt`);
      if (q.fig && !F.FIG[q.fig]) err(`${q.id}: unknown figure ${q.fig}`);
      for (const t of [q.q, ...(q.opts || []), ...(q.why || []), q.hint, q.exp]) balanced(q.id, t);
    }
  }
  for (const g of Q.glossary) balanced(`glossary ${g.t}`, g.d);
  console.log(`questions ${all.length} in ${Q.sections.length} sections; levels ${JSON.stringify(lv)}; kinds ${JSON.stringify(kinds)}; glossary ${Q.glossary.length}; notation ${Q.notation.length}`);
  cues(all, 'English');

  // French: every string the page looks up has a valid translation, and nothing else is stored.
  const fr = JSON.parse(html.match(/<script id="quiz-fr" type="application\/json">([\s\S]*?)<\/script>/)[1]);
  const want = needed(html);
  const missing = [...want.keys()].filter((en) => !Object.hasOwn(fr, en));
  if (missing.length) err(`${missing.length} strings have no French translation (run node quiz/tools/i18n.mjs), e.g. "${missing[0].slice(0, 80)}"`);
  for (const en of want.keys()) { const why = Object.hasOwn(fr, en) && invalid(en, fr[en]); if (why) err(`French of "${en.slice(0, 60)}": ${why}`); }
  const unused = Object.keys(fr).filter((en) => !want.has(en));
  if (unused.length) err(`${unused.length} unused French strings (run node quiz/tools/i18n.mjs), e.g. "${unused[0].slice(0, 80)}"`);
  console.log(`French: ${want.size - missing.length} of ${want.size} strings translated`);
  if (!missing.length) cues(load('fr').Q.sections.flatMap((s) => s.questions), 'French');
}

// Answer cues: the correct option should be no more often the longest or shortest than chance,
// no longer on average than the distractors, and (in English) should not carry the cue words more often.
function cues(all, lang) {
  const plain = (s) => s.replace(/<[^>]+>/g, '').replace(/&[a-z]+;/g, 'x');
  let longest = 0, shortest = 0, ratio = 0;
  const CUES = /\b(only|always|never|all|exactly|may|might|typically|usually|every|any|cannot|must)\b/gi;
  let cueRight = 0, cueWrong = 0;
  for (const q of all) {
    const L = q.opts.map((o) => plain(o).length);
    if (L[0] > Math.max(...L.slice(1))) longest++;
    if (L[0] < Math.min(...L.slice(1))) shortest++;
    ratio += L[0] / ((L[1] + L[2] + L[3]) / 3);
    cueRight += (plain(q.opts[0]).match(CUES) || []).length;
    cueWrong += q.opts.slice(1).reduce((n, o) => n + (plain(o).match(CUES) || []).length, 0) / 3;
  }
  const n = all.length, p = (k) => `${(100 * k / n).toFixed(0)}%`;
  console.log(`${lang}: correct option strictly longest ${longest}/${n} (${p(longest)}), strictly shortest ${shortest}/${n} (${p(shortest)}), chance 25%; mean length of correct / mean distractor ${(ratio / n).toFixed(3)}`
    + (lang === 'English' ? `; cue words per option: correct ${(cueRight / n).toFixed(3)}, distractor ${(cueWrong / n).toFixed(3)}` : ''));
  if (longest / n > 0.3) err(`${lang}: correct option is the longest in ${p(longest)} of questions (limit 30%)`);
  if (shortest / n > 0.3) err(`${lang}: correct option is the shortest in ${p(shortest)} of questions (limit 30%)`);
  if (Math.abs(ratio / n - 1) > 0.05) err(`${lang}: length ratio ${(ratio / n).toFixed(3)} is not within 5% of 1`);
}

// ------------------------------------------------------------------ browser
async function chrome(port = 9461) {
  spawnSync('pkill', ['-f', `remote-debugging-port=${port}`]);           // a stale instance would serve the old page
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'rrea-quiz-chrome-'));
  const proc = spawn('google-chrome', ['--headless=new', `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, '--no-first-run',
    '--hide-scrollbars', '--host-resolver-rules=MAP * ~NOTFOUND', 'about:blank'], { stdio: 'ignore' });
  proc.on('exit', () => fs.rmSync(profile, { recursive: true, force: true }));
  process.on('exit', () => proc.kill());                                  // also when a check throws
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  let target;
  for (let i = 0; i < 80 && !target; i++) {
    try { target = (await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()).find((t) => t.type === 'page'); } catch { /* not up yet */ }
    if (!target) await sleep(200);
  }
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((r) => ws.addEventListener('open', r));
  let id = 0;
  const pending = new Map(), problems = [], requests = [], untranslated = new Set();
  ws.addEventListener('message', (e) => {
    const m = JSON.parse(e.data);
    if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
    if (m.method === 'Runtime.exceptionThrown') problems.push(`exception: ${m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text}`);
    if (m.method === 'Runtime.consoleAPICalled' && ['error', 'warning'].includes(m.params.type)) problems.push(`console: ${m.params.args.map((a) => a.value ?? a.description).join(' ')}`);
    if (m.method === 'Log.entryAdded' && m.params.entry.level === 'error') problems.push(`log: ${m.params.entry.text}`);
    if (m.method === 'Network.requestWillBeSent') requests.push(m.params.request.url);
  });
  const send = (method, params = {}) => new Promise((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
  const ev = async (x) => {
    const r = await send('Runtime.evaluate', { expression: `(async () => { ${x} })()`, awaitPromise: true, returnByValue: true });
    if (r.result?.exceptionDetails) throw new Error(`page script failed: ${JSON.stringify(r.result.exceptionDetails).slice(0, 400)}`);
    return r.result?.result?.value;
  };
  for (const d of ['Runtime', 'Log', 'Page', 'Network', 'DOM', 'CSS']) await send(`${d}.enable`);
  await send('Network.emulateNetworkConditions', { offline: true, latency: 0, downloadThroughput: -1, uploadThroughput: -1 });
  const size = (w, h, mobile = false) => send('Emulation.setDeviceMetricsOverride', { width: w, height: h, deviceScaleFactor: 1, mobile });
  // Navigate or reload, then wait for the new document to finish loading and render.
  const ready = async (go) => {
    const r = await send('Runtime.evaluate', { expression: 'window.__stale = true; window.I18N ? [...window.I18N.missing] : []', returnByValue: true });
    (r.result?.result?.value || []).forEach((s) => untranslated.add(s));
    await go();
    for (let i = 0; i < 300; i++) {
      const r = await send('Runtime.evaluate', { expression: "!window.__stale && document.readyState === 'complete' && !!window.QUIZ && document.getElementById('main')?.childElementCount > 0", returnByValue: true });
      if (r.result?.result?.value === true) return;
      await sleep(100);
    }
    throw new Error('page did not load within 30 s');
  };
  // via about:blank, since a navigation that only changes the hash would not reload the page
  const open = (url) => ready(async () => { await send('Page.navigate', { url: 'about:blank' }); await send('Page.navigate', { url }); });
  const reload = () => ready(() => send('Page.reload', {}));
  const shot = async (file) => { const r = await send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true }); fs.writeFileSync(file, Buffer.from(r.result.data, 'base64')); };
  return { send, ev, ready, open, reload, size, shot, sleep, problems, requests, untranslated, close: () => { ws.close(); proc.kill(); } };
}

const FRAME = 'await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));';
const STORE = 'rrea-quiz:v1';

async function browser(outRoot) {
  const c = await chrome();
  for (const lang of ['en', 'fr']) await suite(c, lang, path.join(outRoot, lang));
  c.close();
}

async function suite(c, lang, outDir) {
  fs.mkdirSync(outDir, { recursive: true });
  const url = `file://${PAGE}?lang=${lang}`;
  const res = [];
  const check = (name, ok, info = '') => { res.push(`${ok ? 'PASS' : 'FAIL'}  ${name}${info ? `  [${info}]` : ''}`); if (!ok) err(`${lang}: ${name}`); };
  c.problems.length = 0; c.requests.length = 0; c.untranslated.clear();
  await c.send('Emulation.setEmulatedMedia', { media: '' });           // the previous language ended in print media
  await c.size(1280, 900);
  await c.open(`${url}#/`);
  const go = async (h) => { await c.ev(`location.hash = ${JSON.stringify(h)}; ${FRAME} return true;`); };
  const sections = await c.ev('return window.QUIZ.sections.map((s) => [s.id, s.questions.length]);');

  // 1. every route renders, with no MathML errors, missing excerpts or non-numbers in text
  const routes = ['#/', '#/sheet', '#/mix/1', '#/all/1', '#/review/1'];
  for (const [sid, nq] of sections) { routes.push(`#/${sid}/lesson`, `#/${sid}/done`); for (let i = 1; i <= nq; i++) routes.push(`#/${sid}/${i}`); }
  const AUDIT = `const main = document.getElementById('main'); return [document.querySelectorAll('merror').length, main.querySelectorAll('.ex-missing').length,
    (main.innerText.match(/\\bNaN\\b|\\bundefined\\b|\\bInfinity\\b|\\[object/g) || []).join(','), main.innerText.length];`;
  let bad = [];
  for (const r of routes) {
    await go(r);
    const [me, missing, junk, len] = await c.ev(AUDIT);
    if (me || missing || junk || len < 40) bad.push(`${r}: merror ${me}, missing excerpts ${missing}, ${junk || 'no junk'}, ${len} chars`);
  }
  // answered views: every option feedback, the excerpts and the other-options panel
  await c.ev(`const a = {}; window.QUIZ.sections.forEach((s) => s.questions.forEach((q) => { a[q.id] = { c: 1 }; }));
    localStorage.setItem('${STORE}', JSON.stringify({ answers: a })); return true;`);
  await c.reload();
  for (const [sid, nq] of sections) for (let i = 1; i <= nq; i++) {
    await go(`#/${sid}/${i}`);
    await c.ev(`document.querySelectorAll('.ref').forEach((b) => b.click()); document.querySelector('.others')?.setAttribute('open', ''); ${FRAME} return true;`);
    const [me, missing, junk] = await c.ev(AUDIT);
    if (me || missing || junk) bad.push(`#/${sid}/${i} answered: merror ${me}, missing excerpts ${missing}, ${junk}`);
  }
  check(`${routes.length} routes and ${sections.reduce((n, s) => n + s[1], 0)} answered views render cleanly`, !bad.length, bad.slice(0, 6).join(' | '));
  await c.ev(`localStorage.clear(); sessionStorage.clear(); return true;`);
  await c.reload();

  // 2. labs stay finite while every slider sweeps its range
  bad = [];
  for (const [sid] of sections) {
    await go(`#/${sid}/lesson`);
    const r = await c.ev(`const out = []; for (const lab of document.querySelectorAll('.lab')) {
      for (const inp of lab.querySelectorAll('input[type=range]')) {
        const v0 = inp.value;
        for (const t of [0, 0.13, 0.37, 0.5, 0.71, 0.93, 1]) {
          inp.value = String(+inp.min + t * (inp.max - inp.min)); inp.dispatchEvent(new Event('input', { bubbles: true }));
          const txt = lab.innerText; if (/NaN|undefined|Infinity|\\[object/.test(txt)) out.push(lab.dataset.lab + ' ' + inp.id + '=' + inp.value);
        }
        inp.value = v0; inp.dispatchEvent(new Event('input', { bubbles: true }));
      }
      lab.querySelector('.lab-resample')?.click();
    } return out;`);
    bad.push(...r);
  }
  check('labs stay finite over slider sweeps', !bad.length, bad.slice(0, 5).join(' | '));

  // 3. behaviour
  await go('#/constructor/1');
  check('#/constructor/1 falls back to the overview', await c.ev(`return location.hash === '#/' || document.querySelector('.hero') !== null;`));
  await go('#/big/1');
  let r = await c.ev(`const wrong = [...document.querySelectorAll('.opt')].find((b) => b.dataset.choice !== '0'); wrong.click(); ${FRAME}
    return [document.querySelectorAll('.feedback').length, Object.keys(JSON.parse(localStorage.getItem('${STORE}')).answers).length];`);
  check('answering shows feedback and stores the answer', r[0] === 1 && r[1] === 1, JSON.stringify(r));
  r = await c.ev(`location.hash = '#/'; ${FRAME} document.querySelector('a[data-mode="review"]').click(); ${FRAME} ${FRAME}
    return [location.hash, document.querySelectorAll('.opt[disabled], .feedback').length, Object.keys(JSON.parse(localStorage.getItem('${STORE}')).answers).length];`);
  check('review re-asks the missed question blank and keeps its saved answer', r[0] === '#/review/1' && r[1] === 0 && r[2] === 1, JSON.stringify(r));
  await c.ev(`localStorage.clear(); sessionStorage.clear(); return true;`);
  await c.open(`${url}#/`);
  await go('#/todo/1');
  await c.ev(`document.querySelector('.opt').click(); ${FRAME} return true;`);
  await go('#/todo/2');
  const before = await c.ev(`return document.querySelector('#qtext').textContent.slice(0, 80);`);
  await c.reload();
  const after = await c.ev(`return document.querySelector('#qtext').textContent.slice(0, 80);`);
  check('a reload keeps the frozen question at #/todo/2', before === after, `${before.slice(0, 30)} | ${after.slice(0, 30)}`);
  r = await c.ev(`location.hash = '#/pic/2'; ${FRAME}
    const box = document.getElementById('single-keys'); box.checked = false; box.dispatchEvent(new Event('change'));
    document.body.dispatchEvent(new KeyboardEvent('keydown', { key: '1', bubbles: true })); ${FRAME}
    const off = document.querySelectorAll('.feedback').length;
    box.checked = true; box.dispatchEvent(new Event('change'));
    document.body.dispatchEvent(new KeyboardEvent('keydown', { key: '1', bubbles: true })); ${FRAME}
    return [off, document.querySelectorAll('.feedback').length];`);
  check('single-key shortcuts can be switched off', r[0] === 0 && r[1] === 1, JSON.stringify(r));
  r = await c.ev(`location.hash = '#/pic/3'; ${FRAME} const b = document.querySelector('[data-action="hint"]'); b.focus(); b.click(); ${FRAME}
    return [document.activeElement?.dataset?.action, document.getElementById('hint')?.hidden];`);
  check('the hint button keeps focus after a re-render', r[0] === 'hint' && r[1] === false, JSON.stringify(r));
  r = await c.ev(`document.querySelector('.skip').click(); ${FRAME} return [location.hash, document.activeElement?.id];`);
  check('the skip link focuses the content without leaving the question', r[0] === '#/pic/3' && r[1] === 'main', JSON.stringify(r));
  await c.size(390, 844, true);
  await go('#/amr1/lesson');
  r = await c.ev(`document.querySelector('.menu-btn').click(); ${FRAME}
    const inert = document.getElementById('main').inert, inside = document.activeElement?.closest('#sidebar') !== null;
    document.body.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true })); ${FRAME}
    return [inert, inside, document.getElementById('main').inert, document.activeElement?.classList.contains('menu-btn')];`);
  check('narrow drawer is modal and Escape restores focus', r.every((x, i) => (i === 2 ? x === false : x === true)), JSON.stringify(r));
  r = await c.ev(`document.querySelector('.menu-btn').click(); ${FRAME} return document.getElementById('main').inert;`);
  await c.size(1280, 900);
  const wide = await c.ev(`${FRAME} return [document.getElementById('main').inert, document.body.classList.contains('nav-open')];`);
  check('widening past the drawer breakpoint closes it and clears inert', r === true && !wide[0] && !wide[1], JSON.stringify([r, wide]));
  const other = lang === 'en' ? 'fr' : 'en';
  await c.ready(() => c.ev(`document.querySelector('[data-lang="${other}"]').click(); return true;`));
  r = await c.ev(`return [document.documentElement.lang, new URLSearchParams(location.search).get('lang'), localStorage.getItem('rrea-quiz:lang'), location.hash,
    document.querySelector('[data-lang="${other}"]').getAttribute('aria-pressed')];`);
  check(`the language switch reloads #/amr1/lesson in ${other}`, JSON.stringify(r) === JSON.stringify([other, other, other, '#/amr1/lesson', 'true']), JSON.stringify(r));
  await c.ev(`localStorage.clear(); sessionStorage.clear(); return true;`);
  await c.open(`${url}#/`);

  // 4. only embedded fonts render text
  await c.size(1280, 900);
  const fonts = new Map(), system = [];
  const scanFonts = async (h) => {
    await go(h);
    await c.ev(`document.querySelectorAll('[data-fc]').forEach((e) => e.removeAttribute('data-fc'));
      const w = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
      for (let n; (n = w.nextNode());) { const p = n.parentElement; if (!p || !n.data.trim() || p.closest('script, style, .sr-only, [hidden], dialog:not([open])')) continue;
        const cs = getComputedStyle(p); if (cs.display !== 'none' && cs.visibility !== 'hidden') p.setAttribute('data-fc', ''); } return true;`);
    const root = (await c.send('DOM.getDocument', { depth: -1 })).result.root.nodeId;
    for (const nodeId of (await c.send('DOM.querySelectorAll', { nodeId: root, selector: '[data-fc]' })).result.nodeIds) {
      for (const f of (await c.send('CSS.getPlatformFontsForNode', { nodeId })).result?.fonts || []) {
        fonts.set(`${f.familyName}${f.isCustomFont ? '' : ' (system)'}`, (fonts.get(f.familyName) || 0) + f.glyphCount);
        if (!f.isCustomFont) system.push(`${h}: ${f.familyName} x${f.glyphCount}`);
      }
    }
  };
  for (const h of ['#/', '#/sheet', ...sections.map(([s]) => `#/${s}/lesson`)]) await scanFonts(h);
  await c.ev(`const a = {}; window.QUIZ.sections.forEach((s) => s.questions.forEach((q) => { a[q.id] = { c: 1 }; }));
    localStorage.setItem('${STORE}', JSON.stringify({ answers: a })); return true;`);
  await c.reload();
  for (const [sid, nq] of sections) for (let i = 1; i <= nq; i++) {
    await go(`#/${sid}/${i}`);
    await c.ev(`document.querySelectorAll('.ref').forEach((b) => b.click()); document.querySelector('.others')?.setAttribute('open', ''); ${FRAME} return true;`);
    await scanFonts(`#/${sid}/${i}`);
  }
  check('all text renders with the embedded fonts', !system.length, [...new Set(system)].slice(0, 6).join(' | '));
  console.log('fonts used:', [...fonts.keys()].join(', '));
  await c.ev(`localStorage.clear(); sessionStorage.clear(); return true;`);

  // 5. layouts: no sideways scrolling or clipped content, legible SVG labels
  const LAYOUT = String.raw`const w = document.documentElement.clientWidth, out = [];
    const desc = (el) => el.tagName.toLowerCase() + (typeof el.className === 'string' && el.className ? '.' + el.className.trim().split(/\s+/).join('.') : '') + ' "' + (el.textContent || '').trim().slice(0, 30) + '"';
    if (document.documentElement.scrollWidth > w + 1) out.push('page scrolls sideways ' + document.documentElement.scrollWidth + ' > ' + w);
    for (const root of document.querySelectorAll('main, .topbar')) for (const el of root.querySelectorAll('*')) {
      if (el.closest('pre, .sr-only, svg, math, .fig-table, .pl-host, .tbl-scroll')) continue;
      const s = getComputedStyle(el); if (s.display === 'none' || s.visibility === 'hidden') continue;
      const r = el.getBoundingClientRect(); if (!r.width && !r.height) continue;
      if (r.right > w + 1 || r.left < -1) out.push(desc(el) + ' extends past the screen');
      if (s.overflowX === 'visible' && el.clientWidth > 0 && el.scrollWidth > el.clientWidth + 1) out.push(desc(el) + ' overflows its box');
    }
    let min = 99, where = '';
    for (const t of document.querySelectorAll('main svg text')) {
      const svg = t.ownerSVGElement, rr = svg.getBoundingClientRect(); if (!rr.width) continue;
      const tb = t.getBoundingClientRect();
      if (tb.width && (tb.left < rr.left - 1 || tb.right > rr.right + 1 || tb.top < rr.top - 1 || tb.bottom > rr.bottom + 1))
        out.push('SVG label "' + t.textContent.slice(0, 24) + '" is cut off in ' + (svg.closest('[data-fig]')?.dataset.fig || svg.closest('[data-lab]')?.dataset.lab || '?'));
      const vb = svg.viewBox.baseVal, k = vb && vb.width ? rr.width / vb.width : 1, px = parseFloat(getComputedStyle(t).fontSize) * k;
      if (px < min) { min = px; where = (svg.closest('[data-fig]')?.dataset.fig || svg.closest('[data-lab]')?.dataset.lab || '?') + ' "' + t.textContent.slice(0, 16) + '"'; }
    }
    return [out, min, where];`;
  const SIZES = [[320, 568, true], [360, 740, true], [375, 667, true], [390, 844, true], [414, 896, true], [430, 932, true], [600, 960, true], [768, 1024, true],
    [844, 390, true], [1024, 768, false], [1280, 800, false], [1366, 768, false], [1440, 900, false], [1920, 1080, false], [2560, 1440, false], [3840, 2160, false]];
  const lay = [];
  for (const [w, h, mobile] of SIZES) {
    await c.size(w, h, mobile);
    const issues = new Set();
    let minPx = 99, minWhere = '';
    const views = ['#/', '#/sheet', ...sections.map(([s]) => `#/${s}/lesson`), ...sections.map(([s]) => `#/${s}/2`), ...sections.map(([s]) => `#/${s}/done`)];
    for (const v of views) {
      await go(v);
      const [out, px, where] = await c.ev(LAYOUT);
      out.forEach((o) => issues.add(`${v}: ${o}`));
      if (px < minPx) { minPx = px; minWhere = `${v} ${where}`; }
    }
    if (minPx < 10) issues.add(`smallest SVG label ${minPx.toFixed(1)}px at ${minWhere}`);
    lay.push(`${String(w).padStart(4)}x${String(h).padEnd(5)} ${issues.size ? `${issues.size} issues: ${[...issues].slice(0, 3).join(' | ')}` : 'clean'}  (smallest SVG label ${minPx.toFixed(1)}px)`);
    if (issues.size) err(`layout ${w}x${h}: ${[...issues].slice(0, 3).join(' | ')}`);
  }
  console.log(lay.join('\n'));

  // 6. screenshots in both themes from a fresh state, after the entrance animation, then the study sheet as PDF
  for (const theme of ['light', 'dark']) {
    for (const [w, h, mobile, tag] of [[390, 844, true, 'phone'], [1440, 900, false, 'desktop'], [3840, 2160, false, '4k']]) {
      await c.size(w, h, mobile);
      for (const [v, name] of [['#/', 'home'], ['#/amr2/lesson', 'lesson'], ['#/wx1/3', 'question']]) {
        await c.ev(`localStorage.clear(); sessionStorage.clear(); localStorage.setItem('rrea-quiz:theme', '${theme}'); return true;`);
        await c.open(`${url}${v}`);
        if (name === 'question') await c.ev(`[...document.querySelectorAll('.opt')].find((b) => b.dataset.choice !== '0')?.click(); ${FRAME} return true;`);
        await c.sleep(700);
        // A full-page capture draws sticky bars at the scroll offset, so capture from the top.
        await c.ev(`window.scrollTo({ top: 0, behavior: 'instant' }); ${FRAME} return true;`);
        await c.shot(path.join(outDir, `${tag}-${theme}-${name}.png`));
      }
    }
  }
  await c.ev(`localStorage.clear(); sessionStorage.clear(); localStorage.setItem('rrea-quiz:theme', 'light'); return true;`);
  await c.size(1200, 900);
  await c.open(`${url}#/sheet`); await c.sleep(800);
  await c.ev(`dispatchEvent(new Event('beforeprint')); return true;`);
  await c.send('Emulation.setEmulatedMedia', { media: 'print' }); await c.sleep(800);
  const pdf = await c.send('Page.printToPDF', { printBackground: true, paperWidth: 8.27, paperHeight: 11.69, marginTop: 0.5, marginBottom: 0.5, marginLeft: 0.5, marginRight: 0.5 });
  fs.writeFileSync(path.join(outDir, 'study-sheet.pdf'), Buffer.from(pdf.result.data, 'base64'));

  const external = c.requests.filter((u) => !/^(file|data|about|blob):/.test(u));
  check('no network requests', !external.length, external.slice(0, 3).join(' '));
  check('no exceptions or console errors', !c.problems.length, [...new Set(c.problems)].slice(0, 4).join(' | '));
  (await c.ev('return [...window.I18N.missing];')).forEach((t) => c.untranslated.add(t));
  check('every string the page looked up is translated', !c.untranslated.size, [...c.untranslated].slice(0, 4).join(' | '));
  console.log(`${lang}:\n${res.join('\n')}`);
  console.log(`screenshots and study-sheet.pdf in ${outDir}`);
}

const mode = process.argv[2] || 'content';
if (mode === 'content') content();
else if (mode === 'browser') await browser(path.resolve(process.argv[3] || path.join(QUIZ_DIR, '.check')));
else { console.error('usage: node quiz/tools/check.mjs content|browser [outdir]'); process.exit(2); }
if (errs.length) { console.log(`FAIL (${errs.length}):\n${errs.join('\n')}`); process.exit(1); }
console.log('OK');
