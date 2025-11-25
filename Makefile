# Makefile for Guardrails Before the Fall
# Author: Clarence H. Cannon, IV

.PHONY: all pdf clean experiments help

# Default target
all: pdf

# Build PDF from LaTeX
pdf:
	@echo "Building PDF..."
	@command -v pdflatex >/dev/null 2>&1 || { echo "LaTeX not found. Use 'make pdf-docker' or upload to Overleaf."; exit 1; }
	cd paper && pdflatex main.tex
	cd paper && bibtex main
	cd paper && pdflatex main.tex
	cd paper && pdflatex main.tex
	@echo "PDF built: paper/main.pdf"

# Build PDF using Docker (no local LaTeX needed)
pdf-docker:
	@echo "Building PDF with Docker..."
	docker run --rm -v $(PWD)/paper:/workdir texlive/texlive:latest \
		sh -c "cd /workdir && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex"
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
	@echo "Running Experiment A: Jailbreak Robustness..."
	python experiments/exp_a_jailbreak/run.py

exp-b:
	@echo "Running Experiment B: Drift Detection..."
	python experiments/exp_b_drift/run.py

exp-c:
	@echo "Running Experiment C: Efficiency Analysis..."
	python experiments/exp_c_efficiency/run.py

exp-d:
	@echo "Running Experiment D: Compositional Evaluation..."
	python experiments/exp_d_compositional/run.py

# Dry run experiments (no API calls - for testing)
dry-run:
	@echo "Running ALL experiments in DRY RUN mode (no API calls)..."
	python experiments/exp_a_jailbreak/run.py --dry-run
	python experiments/exp_c_efficiency/run.py
	@echo "Dry run complete!"

exp-a-dry:
	@echo "Running Experiment A in DRY RUN mode..."
	python experiments/exp_a_jailbreak/run.py --dry-run

# Install Python dependencies
install:
	pip install -r requirements.txt

# Create necessary directories
setup:
	@echo "Creating directory structure..."
	mkdir -p paper/figures
	mkdir -p data
	mkdir -p experiments/exp_a_jailbreak/results
	mkdir -p experiments/exp_b_drift/results
	mkdir -p experiments/exp_c_efficiency/results
	mkdir -p experiments/exp_d_compositional/results
	touch data/.gitkeep
	@echo "Done. Run 'make experiments' to start."

# Watch for changes and rebuild (requires fswatch)
watch:
	fswatch -o paper/*.tex | xargs -n1 -I{} make quick

# Help
help:
	@echo "Available targets:"
	@echo "  make pdf           - Build PDF (full build with bibliography)"
	@echo "  make pdf-docker    - Build PDF using Docker (no local LaTeX needed)"
	@echo "  make quick         - Quick PDF build (no bibliography)"
	@echo "  make clean         - Remove LaTeX build artifacts"
	@echo "  make distclean     - Remove all generated files including PDF"
	@echo ""
	@echo "  make experiments   - Run all experiments (requires API keys)"
	@echo "  make dry-run       - Run experiments without API calls (testing)"
	@echo "  make exp-a         - Run Experiment A (jailbreak robustness)"
	@echo "  make exp-a-dry     - Run Experiment A without API calls"
	@echo "  make exp-b         - Run Experiment B (drift detection)"
	@echo "  make exp-c         - Run Experiment C (efficiency - no API needed)"
	@echo "  make exp-d         - Run Experiment D (compositional eval)"
	@echo ""
	@echo "  make install       - Install Python dependencies"
	@echo "  make setup         - Create directory structure"
	@echo "  make watch         - Auto-rebuild on file changes"
	@echo "  make help          - Show this help message"