# Makefile for Guardrails Before the Fall
# Author: Clarence H. Cannon, IV

.PHONY: all pdf clean experiments help

# Default target
all: pdf

# Build PDF from LaTeX
pdf:
	@echo "Building PDF..."
	cd paper && pdflatex main.tex
	cd paper && bibtex main
	cd paper && pdflatex main.tex
	cd paper && pdflatex main.tex
	@echo "PDF built: paper/main.pdf"

# Quick build (no bibliography update)
quick:
	@echo "Quick build..."
	cd paper && pdflatex main.tex
	@echo "PDF built: paper/main.pdf"

# Clean LaTeX build artifacts
clean:
	@echo "Cleaning build artifacts..."
	cd paper && rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.synctex.gz *.fdb_latexmk *.fls
	@echo "Clean complete."

# Deep clean (includes PDF)
distclean: clean
	cd paper && rm -f *.pdf

# Run all experiments
experiments:
	@echo "Running experiments..."
	python experiments/exp_a_jailbreak/run.py
	python experiments/exp_b_drift/run.py
	python experiments/exp_c_efficiency/run.py
	python experiments/exp_d_compositional/run.py
	@echo "Experiments complete. Results in experiments/*/results/"

# Individual experiments
exp-a:
	python experiments/exp_a_jailbreak/run.py

exp-b:
	python experiments/exp_b_drift/run.py

exp-c:
	python experiments/exp_c_efficiency/run.py

exp-d:
	python experiments/exp_d_compositional/run.py

# Install Python dependencies
install:
	pip install -r requirements.txt

# Create necessary directories
setup:
	mkdir -p paper/figures
	mkdir -p data
	mkdir -p experiments/exp_a_jailbreak/results
	mkdir -p experiments/exp_b_drift/results
	mkdir -p experiments/exp_c_efficiency/results
	mkdir -p experiments/exp_d_compositional/results
	touch data/.gitkeep

# Watch for changes and rebuild (requires fswatch)
watch:
	fswatch -o paper/*.tex | xargs -n1 -I{} make quick

# Help
help:
	@echo "Available targets:"
	@echo "  make pdf         - Build PDF (full build with bibliography)"
	@echo "  make quick       - Quick PDF build (no bibliography)"
	@echo "  make clean       - Remove LaTeX build artifacts"
	@echo "  make distclean   - Remove all generated files including PDF"
	@echo "  make experiments - Run all experiments"
	@echo "  make exp-a       - Run Experiment A (jailbreak robustness)"
	@echo "  make exp-b       - Run Experiment B (drift detection)"
	@echo "  make exp-c       - Run Experiment C (safety-per-joule)"
	@echo "  make exp-d       - Run Experiment D (compositional eval)"
	@echo "  make install     - Install Python dependencies"
	@echo "  make setup       - Create directory structure"
	@echo "  make watch       - Auto-rebuild on file changes"
	@echo "  make help        - Show this help message"