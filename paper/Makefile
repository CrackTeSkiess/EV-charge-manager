# ============================================================
#  EV-Charge-Manager Paper Build System
#  Usage:
#    make          – generate figures + compile PDF
#    make generate – run Python script to extract data & figures
#    make pdf      – compile LaTeX only (requires prior `make generate`)
#    make clean    – remove auxiliary files
#    make distclean – remove all generated files including figures
# ============================================================

PAPER      := main
SCRIPT     := ../scripts/generate_paper.py
PYTHON     := python3
LATEX      := pdflatex
BIBTEX     := bibtex
FIGDIR     := figures

.PHONY: all generate pdf clean distclean

all: generate pdf

## Step 1 – extract data, generate figures, inject macros into main.tex
generate:
	@echo "==> Generating figures and LaTeX macros..."
	$(PYTHON) $(SCRIPT) --paper-dir .
	@echo "==> Done. Figures written to $(FIGDIR)/"

## Step 2 – compile LaTeX (three-pass for references)
pdf: $(PAPER).tex
	@echo "==> Compiling LaTeX (pass 1/3)..."
	$(LATEX) -interaction=nonstopmode $(PAPER).tex
	@echo "==> Running BibTeX..."
	-$(BIBTEX) $(PAPER)
	@echo "==> Compiling LaTeX (pass 2/3)..."
	$(LATEX) -interaction=nonstopmode $(PAPER).tex
	@echo "==> Compiling LaTeX (pass 3/3)..."
	$(LATEX) -interaction=nonstopmode $(PAPER).tex
	@echo "==> PDF written to $(PAPER).pdf"

## Remove auxiliary LaTeX files
clean:
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg \
	       $(PAPER).log $(PAPER).out $(PAPER).toc \
	       $(PAPER).lof $(PAPER).lot

## Remove everything generated (figures + PDF + aux)
distclean: clean
	rm -f $(PAPER).pdf
	rm -rf $(FIGDIR)/*.pdf $(FIGDIR)/*.png
