# Experimental Protocols

**Guardrails Before the Fall: Anticipatory Governance for AI, Quantum, and Neuromorphic Systems**

Author: Clarence H. Cannon, IV

---

## Overview

This directory contains minimal-compute experiments designed to substantiate claims in the paper and provide reproducible evidence for policy discussions.

| Experiment | Directory              | Objective                              | Hardware            |
| ---------- | ---------------------- | -------------------------------------- | ------------------- |
| A          | `exp_a_jailbreak/`     | Jailbreak robustness vs. attack budget | API or 8GB GPU      |
| B          | `exp_b_drift/`         | Post-deployment drift detection        | API access          |
| C          | `exp_c_efficiency/`    | Safety-per-Joule analysis              | Neuromorphic or GPU |
| D          | `exp_d_compositional/` | Compositional safety evaluation        | Sandboxed API       |

---

## Setup

### Prerequisites

```bash
# Python 3.10+
python --version

# Install dependencies
pip install -r ../requirements.txt

# Set up API keys (create .env file)
cp .env.example .env
# Edit .env with your API keys
```

### Environment Variables

Create a `.env` file in the repository root:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
HUGGINGFACE_TOKEN=hf_...
WANDB_API_KEY=...
```

---

## Experiment A: Jailbreak Robustness Under Cost-Constrained Attack

### Objective

Estimate effective robustness versus query budget and attack composition.

### Method

1. Deploy 3-5 open models behind standard APIs
2. Run fixed suite of 200 red-team prompts across domains (bio/cyber/physical)
3. Implement three attacker types:
   - **Naive**: Direct policy-violating requests
   - **Paraphrase-mutate**: Automated prompt variation
   - **Autoregressive**: LLM-generated attack sequences

### Run

```bash
cd exp_a_jailbreak
python run.py --models "gpt-4,claude-3,llama-70b" --prompts 200 --attackers all
```

### Output

- `results/attack_success_rate.csv`
- `results/tokens_per_success.csv`
- `results/figures/robustness_curve.png`

### Key Metrics

- Attack success rate vs. query count
- Effective tokens per successful attack
- Cross-model vulnerability correlation

---

## Experiment B: Post-Deployment Drift Detection

### Objective

Detect sandbagging and reward hacking patterns through continuous monitoring.

### Method

1. Maintain hidden evaluation subset (not in public benchmarks)
2. Measure weekly for step-changes in risky capability scores
3. Implement tripwire that halts tool-use upon detecting OOD exploit patterns

### Run

```bash
cd exp_b_drift
python run.py --baseline-weeks 4 --monitoring-weeks 8 --threshold 2.0
```

### Output

- `results/capability_timeseries.csv`
- `results/drift_alerts.json`
- `results/figures/drift_detection.png`

### Key Metrics

- Capability score variance over time
- False positive rate of tripwire
- Detection latency for synthetic drift injection

---

## Experiment C: Safety-per-Joule Analysis

### Objective

Quantify energy/performance trade-offs between neuromorphic and GPU architectures.

### Method

**If neuromorphic hardware available:**

1. Port matmul-free classifier following published architectures
2. Benchmark throughput/Joule vs. GPU baseline
3. Measure inference latency distribution

**If GPU-only:**

1. Reproduce quantized proxy models
2. Extrapolate neuromorphic performance from literature
3. Sensitivity analysis on efficiency claims

### Run

```bash
cd exp_c_efficiency

# GPU baseline
python run.py --mode gpu --model "llama-7b-int4" --batch-sizes "1,8,32"

# Neuromorphic (if available)
python run.py --mode loihi --model "snn-classifier" --spikes-per-inference 1000
```

### Output

- `results/throughput_per_joule.csv`
- `results/latency_distribution.csv`
- `results/figures/efficiency_comparison.png`

### Key Metrics

- Inferences per Joule
- Latency (p50, p95, p99)
- Cost per 1M inferences

---

## Experiment D: Cross-Domain Evaluation Composition

### Objective

Demonstrate that isolated safety evaluations do not predict compositional safety.

### Method

1. Chain model through browser/tool API in sandboxed environment
2. Define escalation pathway: cyber recon → code gen → execution attempt
3. Measure risk escalation vs. base model capabilities
4. Compare isolated eval pass rate vs. compositional eval pass rate

### Run

```bash
cd exp_d_compositional

# Set up sandbox
docker-compose up -d

# Run evaluation
python run.py --sandbox-mode docker --scenarios "recon,codegen,execution" --models "gpt-4,claude-3"
```

### Output

- `results/isolated_vs_compositional.csv`
- `results/escalation_paths.json`
- `results/figures/safety_gap.png`

### Key Metrics

- Isolated eval pass rate
- Compositional eval pass rate
- Risk escalation score (0-10 scale)
- Safety gap = isolated_pass - compositional_pass

---

## Reproducibility Checklist

- [ ] Random seeds documented and fixed (`SEED=42`)
- [ ] Model versions and API endpoints specified in `config.yaml`
- [ ] Prompt templates provided in `prompts/`
- [ ] Statistical tests pre-registered in `analysis/preregistration.md`
- [ ] Code released under MIT license
- [ ] Results logged to Weights & Biases (optional)

---

## Statistical Analysis

All experiments use:

- **Significance level**: α = 0.05
- **Multiple comparison correction**: Bonferroni
- **Effect size reporting**: Cohen's d for continuous, odds ratio for binary
- **Confidence intervals**: 95% bootstrapped (n=10,000)

---

## Ethical Considerations

These experiments involve adversarial testing of AI systems. Guidelines:

1. **No real-world deployment** of successful attacks
2. **Responsible disclosure** to model providers before publication
3. **Sanitized prompts** in published datasets (harmful specifics redacted)
4. **Sandboxed execution** for all tool-use experiments

---

## Citation

If you use these experimental protocols, please cite:

```bibtex
@article{cannon2025guardrails,
  title={Guardrails Before the Fall: Anticipatory Governance for AI, Quantum, and Neuromorphic Systems},
  author={Cannon, Clarence H., IV},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

---

## Contact

Clarence H. Cannon, IV  
clarence@simplysubscribe.com
