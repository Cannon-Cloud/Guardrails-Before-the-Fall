# Guardrails Before the Fall

**Anticipatory Governance for AI, Quantum, and Neuromorphic Systems**

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![arXiv](https://img.shields.io/badge/arXiv-pending-b31b1b.svg)](https://arxiv.org/)

## Author

**Clarence H. Cannon, IV**  
Chief Technology Officer, Simply Subscribe  
San Francisco, CA  
📧 clarence@simplysubscribe.com

## Abstract

Frontier artificial intelligence, neuromorphic computing, and quantum systems are converging on accelerated timelines, presenting unprecedented governance challenges. Current containment strategies—particularly red-teaming—while necessary, are demonstrably insufficient as standalone control mechanisms. This paper argues that anticipatory governance frameworks must be established _before_ potential market corrections in the AI sector, not after.

## Repository Structure

```
├── paper/
│   ├── main.tex              # Main LaTeX source
│   ├── references.bib        # BibTeX references
│   └── figures/              # Generated figures (after compilation)
├── experiments/
│   ├── README.md             # Experimental protocols
│   ├── exp_a_jailbreak/      # Experiment A: Jailbreak robustness
│   ├── exp_b_drift/          # Experiment B: Drift detection
│   ├── exp_c_efficiency/     # Experiment C: Safety-per-Joule
│   └── exp_d_compositional/  # Experiment D: Compositional eval
├── data/                     # Data files (after experiments)
├── .gitignore
├── LICENSE
├── Makefile
└── README.md
```

## Quick Start

### Building the PDF

**Option 1: Overleaf (Recommended)**

1. Create account at [overleaf.com](https://www.overleaf.com)
2. New Project → Upload Project → upload this repository as ZIP
3. Click "Recompile"

**Option 2: Local Build**

Requires TeX Live, MiKTeX, or MacTeX installed.

```bash
# Clone repository
git clone https://github.com/[your-username]/guardrails-before-the-fall.git
cd guardrails-before-the-fall

# Build PDF
make pdf

# Or manually:
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Running Experiments

```bash
# Install dependencies
pip install -r requirements.txt

# Run all experiments
make experiments

# Run individual experiment
python experiments/exp_a_jailbreak/run.py
```

## Key Contributions

1. **Governance Gap Analysis**: Documents the lag between capability advancement and regulatory implementation
2. **Multi-Layer Framework**: Proposes 5-layer governance architecture beyond red-teaming
3. **Empirical Protocols**: Minimal-compute experiments to validate governance claims
4. **Policy Recommendations**: Concrete proposals for responsible scaling triggers

## Figures

| Figure   | Description                                  |
| -------- | -------------------------------------------- |
| Figure 1 | Regulatory & Technology Timeline (2023-2027) |
| Figure 2 | Multi-layered Governance Framework           |

## Tables

| Table   | Description                                         |
| ------- | --------------------------------------------------- |
| Table 1 | Red-Teaming Only vs. Responsible Scaling Comparison |
| Table 2 | Summary of Proposed Experiments                     |

## Citation

```bibtex
@article{cannon2025guardrails,
  title={Guardrails Before the Fall: Anticipatory Governance for AI, Quantum, and Neuromorphic Systems},
  author={Cannon, Clarence H., IV},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## Related Work

- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)
- [EU AI Act](https://artificialintelligenceact.eu/)
- [Anthropic Responsible Scaling Policy](https://www.anthropic.com/news/anthropics-responsible-scaling-policy)
- [UK AI Safety Institute](https://www.aisi.gov.uk/)

## License

This work is licensed under a [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/).

## Acknowledgments

[To be added]

---

**Simply Subscribe** | [simplysubscribe.com](https://simplysubscribe.com)
