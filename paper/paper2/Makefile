PY := ../../code/.venv/bin/python

.PHONY: all assets clean

all: quantum-state-frozen-horizon.pdf

assets:
	cd ../../code && $(PY) scripts/make_paper2_assets.py

quantum-state-frozen-horizon.pdf: quantum-state-frozen-horizon.tex refs.bib \
	tables/kernel.tex tables/shells.tex
	latexmk -pdf -interaction=nonstopmode quantum-state-frozen-horizon.tex

clean:
	latexmk -C
