PY := ../code/.venv/bin/python

.PHONY: all tables clean

all: nonsingular-fR-starobinsky-completion.pdf

tables:
	cd ../code && $(PY) scripts/make_ledger.py --p 66 --alpha 2.809064 --xi 0.16666666666666666 --results invwall_p
	cd ../code && $(PY) scripts/make_paper_tables.py
	cd ../code && $(PY) scripts/make_figures.py

nonsingular-fR-starobinsky-completion.pdf: nonsingular-fR-starobinsky-completion.tex refs.bib tables/likelihood.tex tables/evidence.tex tables/cmb_ratios.tex tables/candidate.tex
	latexmk -pdf -interaction=nonstopmode nonsingular-fR-starobinsky-completion.tex

clean:
	latexmk -C
