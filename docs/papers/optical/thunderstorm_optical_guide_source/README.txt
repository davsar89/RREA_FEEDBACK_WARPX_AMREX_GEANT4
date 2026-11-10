FROM RUNAWAY ELECTRONS TO FEGS SIGNALS
A beginner-friendly synthesis of the 22 September 2026 conversation.

FILES
  thunderstorm_optical_guide.tex  Main, editable LaTeX source
  figure5.png                    Figure panels cropped from the supplied Xu figure
  README.txt                     This file

COMPILE
  latexmk -pdf thunderstorm_optical_guide.tex

Alternatively:
  pdflatex thunderstorm_optical_guide.tex
  pdflatex thunderstorm_optical_guide.tex

Use pdfLaTeX in a standard TeX Live or Overleaf environment. Upload the ZIP as
an Overleaf project and select thunderstorm_optical_guide.tex as the main file.
References are embedded in the .tex file, so no .bib file, BibTeX or Biber is
needed. No shell escape is required. No custom font files are included.

The .tex file also compiles without figure5.png; in that case an explanatory
placeholder replaces the figure. Keep the PNG beside the .tex file for the
illustrated version. A separately supplied PDF is the compiled 14-page version.

SCOPE
This is a research planning guide, not simulation software or a validated
instrument calibration. Published results, calculated examples and proposed
modelling choices are distinguished. The no-leader source and existing
cloudscat.js propagation component are user-supplied project context.

FIGURE ATTRIBUTION
Figure panels: Xu, Celestin and Pasko (2015), Optical emissions associated
with terrestrial gamma ray flashes, JGR Space Physics, Figure 5, p. 1362.
DOI: 10.1002/2014JA020425. The crop originates from the user-supplied figure.
Original figure rights remain with the original rightsholders. The complete
paper is not included in this source bundle.
