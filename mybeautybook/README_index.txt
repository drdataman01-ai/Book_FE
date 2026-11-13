Adding the index to Financial Engineering and the Evidence
==========================================================

1. Replace your chapters/ch01.tex ... ch26.tex with the tagged versions in chapters/.
   Copy fe.ist next to fe.tex, and chapters/index.tex into chapters/.

2. In the preamble (common/preamble.tex or fe.tex), load imakeidx BEFORE hyperref:

     \usepackage{imakeidx}
     \makeindex[intoc, columns=2, title=Index, options=-s fe.ist]

3. At the very end of fe.tex, after the references and other back matter:

     \input{chapters/index}
     \end{document}

4. Compile with pdflatex as usual. imakeidx runs makeindex automatically
   (it is on TeX Live's allowed shell-escape list), so two or three pdflatex
   runs are enough. With latexmk, `latexmk -pdf fe` also works.

Conventions
- Bold page numbers mark where a term is defined (\index{term|textbf}).
- Plain page numbers mark the first mention of a term in each chapter.
- Sub-entries use "!" (e.g. \index{return!excess}, \index{Value at Risk (VaR)!backtesting}).
- No tags were placed in headings, captions, equations, figures, tables or the Exercises.
- To add a term by hand: write \index{term} right after the word, with no space.

tools/terms.py lists all 450+ terms and their spellings; tools/addindex.py is the
script that placed the tags (it skips any file that already contains \index tags).
