# RREA on WarpX and AMReX: course and quiz

`rrea-quiz.html` is a self-contained interactive course: 14 lessons (seven of them on
AMReX and WarpX themselves and on how this project attaches to them) with figures, labs
and runnable Python examples, 140 questions whose feedback explains every option and shows
the exact source lines, mixed practice, review of missed questions, a glossary and a
printable study sheet, in English and French. Open it in a current browser; it works from
`file://` and makes no network requests. Progress is kept in the browser's local storage.

The EN/FR switch in the top bar reloads the page in the other language and remembers the
choice; `rrea-quiz.html?lang=fr` opens it in French directly. Without either, a browser set
to French gets French.

## Regenerate

From the repository root, with a native build of the project:

    python3 quiz/tools/build.py --probe <build>/bin/rrea_interaction_table_probe

This recomputes the figure data with the project's Python modules and the C++ drag probe,
extracts the excerpt for every `file:line` reference (including the pinned `upstream/`
WarpX and AMReX sources), runs the examples in `tools/examples.py`, and checks every quoted
number and quoted document text against its source, naming the question that drifted. It
rewrites only the block between `<!-- generated:begin -->` and `<!-- generated:end -->`;
lessons and questions are the hand-written `window.QUIZ` script in the page. The examples
also run on their own:

    PYTHONPATH=scripts:quiz/tools RREA_PROBE=<probe> python3 quiz/tools/examples.py

English is the source text. After editing or regenerating it, update the French:

    OPENROUTER_API_KEY=... node quiz/tools/i18n.mjs

It collects every string the page shows (lesson content, figure and lab labels, interface
text), keeps valid existing translations, sends new or changed strings to
`google/gemini-4-argon` through OpenRouter with the course terminology, rejects any
translation that alters markup, code, math, placeholders or numbers (retrying with the
reason), drops unused ones and rewrites the `quiz-fr` block. Code, equations and file
references stay in English. A complete translation costs about $2.

## Check

    node quiz/tools/check.mjs content
    node quiz/tools/check.mjs browser [outdir]

`content` validates the question data (four options, feedback for each distractor,
references, figures, house style), requires a valid French translation of every string,
and prints the answer-cue statistics in both languages: the correct option should be the
longest, and the shortest, about a quarter of the time each, no longer on average than the
distractors, and (in English) no more likely to carry absolute words. `browser` drives
offline headless Chrome (Node 22 and `google-chrome` on the `PATH`) in English, then French:
every route and answered view, untranslated strings, lab slider sweeps, behaviour tests
including the language switch, embedded fonts only, 16 layouts from 320 to 3840 px with
figure labels inside their drawings, screenshots in both themes and the study sheet as PDF
(default output `quiz/.check/en` and `quiz/.check/fr`).

The fonts (Inter, JetBrains Mono and Noto Sans Math, SIL OFL 1.1) are embedded as WOFF2
subsets covering Latin, Greek, punctuation, arrows and mathematical operators plus every
character the page used when they were cut. Text that falls back to a system font fails the
browser check; a character outside the subsets needs the fonts re-cut from the upstream
releases with fontTools, keeping the MATH table of Noto Sans Math.

`external-review.md` records the independent review and how each finding was handled.
