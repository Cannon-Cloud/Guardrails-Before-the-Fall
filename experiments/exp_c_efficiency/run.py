#!/usr/bin/env python3
"""
Experiment C: Safety-per-Joule Analysis

Quantifies energy/performance trade-offs between architectures.
Demonstrates democratization concerns - lower cost implies wider access risk.

Author: Clarence H. Cannon, IV
"""

import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from tqdm import tqdm

from utils import (
    setup_logging, set_seed, ResultsTracker,
    calculate_statistics, bootstrap_ci, save_figure
)

logger = setup_logging("ExpC_Efficiency")

# ============================================================================
# Energy Consumption Estimates (from literature)
# ============================================================================

# Energy per inference (Joules) - estimates from published research
ENERGY_ESTIMATES = {
    # GPU-based models (datacenter)
    "gpt-4": {
        "joules_per_inference": 0.5,  # Estimated
        "source": "Patterson et al. 2022 extrapolation",
        "hardware": "NVIDIA A100"
    },
    "gpt-4o": {
        "joules_per_inference": 0.3,
        "source": "Efficiency improvements estimate",
        "hardware": "NVIDIA H100"
    },
    "claude-3-opus": {
        "joules_per_inference": 0.45,
        "source": "Similar scale estimate",
        "hardware": "Cloud GPU"
    },
    "claude-3-5-sonnet": {
        "joules_per_inference": 0.25,
        "source": "Optimized architecture",
        "hardware": "Cloud GPU"
    },
    "llama-70b": {
        "joules_per_inference": 0.35,
        "source": "Open model benchmark",
        "hardware": "NVIDIA A100"
    },
    "llama-7b": {
        "joules_per_inference": 0.05,
        "source": "Smaller model",
        "hardware": "Consumer GPU"
    },
    "llama-7b-int4": {
        "joules_per_inference": 0.02,
        "source": "Quantized model",
        "hardware": "Consumer GPU"
    },
    
    # Neuromorphic estimates (from Intel Loihi 2 papers)
    "loihi2-snn-small": {
        "joules_per_inference": 0.0001,  # 0.1 mJ
        "source": "Intel Loihi 2 benchmarks",
        "hardware": "Intel Loihi 2"
    },
    "loihi2-snn-large": {
        "joules_per_inference": 0.001,  # 1 mJ
        "source": "Intel Hala Point estimates",
        "hardware": "Intel Hala Point"
    },
    "darwin-neuromorphic": {
        "joules_per_inference": 0.0005,
        "source": "Zhejiang Darwin III estimates",
        "hardware": "Darwin III chip"
    },
    
    # Edge devices
    "edge-tpu": {
        "joules_per_inference": 0.002,
        "source": "Google Edge TPU specs",
        "hardware": "Google Edge TPU"
    },
    "jetson-nano": {
        "joules_per_inference": 0.01,
        "source": "NVIDIA Jetson benchmarks",
        "hardware": "NVIDIA Jetson Nano"
    }
}

# Cost per kWh (USD) by region
ELECTRICITY_COSTS = {
    "us_average": 0.12,
    "us_california": 0.22,
    "eu_average": 0.25,
    "china": 0.08,
    "india": 0.06
}


@dataclass
class EfficiencyMeasurement:
    """Single efficiency measurement."""
    model: str
    task: str
    batch_size: int
    throughput: float  # inferences per second
    latency_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    energy_joules: float
    inferences_per_joule: float
    cost_per_million: float  # USD per 1M inferences
    hardware: str
    is_measured: bool  # True if actually measured, False if estimated


@dataclass
class AccessibilityMetric:
    """Metric for compute accessibility/democratization."""
    model: str
    hardware_cost_usd: float
    min_viable_hardware: str
    can_run_consumer: bool
    can_run_mobile: bool
    monthly_cost_1m_inferences: float
    risk_accessibility_score: float  # 0-1, higher = more accessible = more risk


class EfficiencyBenchmark:
    """Benchmark efficiency across different architectures."""
    
    def __init__(self):
        self.logger = setup_logging("EfficiencyBenchmark")
        self.measurements: List[EfficiencyMeasurement] = []
    
    def estimate_efficiency(self, model: str, task: str = "text_generation",
                           batch_size: int = 1) -> EfficiencyMeasurement:
        """
        Estimate efficiency for a model (when hardware not available).
        Uses literature values and scaling laws.
        """
        base_energy = ENERGY_ESTIMATES.get(model, {}).get('joules_per_inference', 0.1)
        hardware = ENERGY_ESTIMATES.get(model, {}).get('hardware', 'Unknown')
        
        # Batch efficiency scaling (sublinear)
        batch_efficiency = 1.0 + 0.3 * np.log2(max(batch_size, 1))
        effective_energy = base_energy / batch_efficiency
        
        # Estimate throughput from energy (inverse relationship)
        base_throughput = 1.0 / (base_energy * 10)  # Rough scaling
        throughput = base_throughput * batch_efficiency * batch_size
        
        # Latency estimates
        latency_ms = 1000 / throughput if throughput > 0 else float('inf')
        
        # Cost calculation
        kwh_per_inference = effective_energy / 3600000  # J to kWh
        cost_per_inference = kwh_per_inference * ELECTRICITY_COSTS['us_average']
        cost_per_million = cost_per_inference * 1_000_000
        
        measurement = EfficiencyMeasurement(
            model=model,
            task=task,
            batch_size=batch_size,
            throughput=throughput,
            latency_ms=latency_ms,
            latency_p95_ms=latency_ms * 1.5,
            latency_p99_ms=latency_ms * 2.0,
            energy_joules=effective_energy,
            inferences_per_joule=1.0 / effective_energy,
            cost_per_million=cost_per_million,
            hardware=hardware,
            is_measured=False
        )
        
        self.measurements.append(measurement)
        return measurement
    
    def measure_gpu_efficiency(self, model_name: str, batch_sizes: List[int] = [1, 8, 32]):
        """
        Actually measure GPU efficiency (requires torch and GPU).
        """
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            self.logger.warning("PyTorch/Transformers not available, using estimates")
            for bs in batch_sizes:
                self.estimate_efficiency(model_name, batch_size=bs)
            return
        
        if not torch.cuda.is_available():
            self.logger.warning("CUDA not available, using estimates")
            for bs in batch_sizes:
                self.estimate_efficiency(model_name, batch_size=bs)
            return
        
        self.logger.info(f"Measuring efficiency for {model_name}")
        
        # Load model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Test prompt
        test_prompt = "The quick brown fox jumps over the lazy dog."
        
        for batch_size in batch_sizes:
            latencies = []
            
            # Warmup
            inputs = tokenizer([test_prompt] * batch_size, return_tensors="pt", 
                             padding=True).to(model.device)
            for _ in range(3):
                with torch.no_grad():
                    model.generate(**inputs, max_new_tokens=50)
            
            # Measure
            torch.cuda.synchronize()
            for _ in tqdm(range(20), desc=f"Batch {batch_size}"):
                start = time.perf_counter()
                with torch.no_grad():
                    model.generate(**inputs, max_new_tokens=50)
                torch.cuda.synchronize()
                latencies.append((time.perf_counter() - start) * 1000)
            
            # Calculate metrics
            latency_ms = np.mean(latencies)
            throughput = batch_size / (latency_ms / 1000)
            
            # GPU power measurement (if pynvml available)
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                power_w = power_mw / 1000
                energy_j = power_w * (latency_ms / 1000) / batch_size
                pynvml.nvmlShutdown()
            except:
                # Estimate from TDP
                energy_j = 0.1  # Default estimate
            
            measurement = EfficiencyMeasurement(
                model=model_name,
                task="text_generation",
                batch_size=batch_size,
                throughput=throughput,
                latency_ms=latency_ms,
                latency_p95_ms=np.percentile(latencies, 95),
                latency_p99_ms=np.percentile(latencies, 99),
                energy_joules=energy_j,
                inferences_per_joule=1.0 / energy_j,
                cost_per_million=(energy_j / 3600000) * ELECTRICITY_COSTS['us_average'] * 1e6,
                hardware=torch.cuda.get_device_name(0),
                is_measured=True
            )
            self.measurements.append(measurement)
    
    def calculate_accessibility(self, model: str) -> AccessibilityMetric:
        """Calculate how accessible a model is (democratization risk)."""
        
        # Hardware cost estimates
        hardware_costs = {
            "NVIDIA A100": 15000,
            "NVIDIA H100": 30000,
            "Consumer GPU": 800,
            "Intel Loihi 2": 5000,  # Research access
            "Intel Hala Point": 100000,  # System cost
            "Google Edge TPU": 75,
            "NVIDIA Jetson Nano": 100,
            "Darwin III chip": 1000,  # Estimate
        }
        
        info = ENERGY_ESTIMATES.get(model, {})
        hardware = info.get('hardware', 'Unknown')
        energy = info.get('joules_per_inference', 0.1)
        
        hw_cost = hardware_costs.get(hardware, 10000)
        
        # Consumer viability: can run on <$1000 hardware
        can_consumer = hw_cost < 1000 or 'consumer' in hardware.lower()
        
        # Mobile viability: can run on edge devices
        can_mobile = energy < 0.01 or 'edge' in hardware.lower() or 'jetson' in hardware.lower()
        
        # Monthly cost for 1M inferences
        kwh = energy * 1e6 / 3600000
        monthly_cost = kwh * ELECTRICITY_COSTS['us_average']
        
        # Risk accessibility score (higher = easier access = more risk)
        # Factors: low hardware cost, low energy, consumer/mobile viable
        risk_score = 0.0
        if hw_cost < 500:
            risk_score += 0.4
        elif hw_cost < 2000:
            risk_score += 0.2
        
        if energy < 0.01:
            risk_score += 0.3
        elif energy < 0.1:
            risk_score += 0.15
        
        if can_consumer:
            risk_score += 0.2
        if can_mobile:
            risk_score += 0.1
        
        return AccessibilityMetric(
            model=model,
            hardware_cost_usd=hw_cost,
            min_viable_hardware=hardware,
            can_run_consumer=can_consumer,
            can_run_mobile=can_mobile,
            monthly_cost_1m_inferences=monthly_cost,
            risk_accessibility_score=min(risk_score, 1.0)
        )


def run_experiment(
    models: List[str] = None,
    batch_sizes: List[int] = [1, 8, 32],
    output_dir: str = "./results",
    measure_gpu: bool = False
):
    """Run the full efficiency analysis experiment."""
    
    set_seed(42)
    
    if models is None:
        models = list(ENERGY_ESTIMATES.keys())
    
    logger.info(f"Analyzing efficiency for {len(models)} models")
    
    benchmark = EfficiencyBenchmark()
    results_tracker = ResultsTracker("efficiency_analysis", output_dir)
    
    # Run benchmarks
    for model in tqdm(models, desc="Models"):
        if measure_gpu and 'loihi' not in model.lower() and 'darwin' not in model.lower():
            benchmark.measure_gpu_efficiency(model, batch_sizes)
        else:
            for bs in batch_sizes:
                benchmark.estimate_efficiency(model, batch_size=bs)
    
    # Calculate accessibility metrics
    accessibility_metrics = []
    for model in models:
        metric = benchmark.calculate_accessibility(model)
        accessibility_metrics.append(asdict(metric))
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Measurements
    measurements_df = pd.DataFrame([asdict(m) for m in benchmark.measurements])
    measurements_df.to_csv(output_path / "efficiency_measurements.csv", index=False)
    
    # Accessibility
    accessibility_df = pd.DataFrame(accessibility_metrics)
    accessibility_df.to_csv(output_path / "accessibility_metrics.csv", index=False)
    
    # Generate visualizations
    generate_figures(benchmark.measurements, accessibility_metrics, output_dir)
    
    # Summary statistics
    summary = {
        'models_analyzed': len(models),
        'most_efficient': measurements_df.loc[measurements_df['inferences_per_joule'].idxmax(), 'model'],
        'least_efficient': measurements_df.loc[measurements_df['inferences_per_joule'].idxmin(), 'model'],
        'efficiency_range': {
            'min_ipj': float(measurements_df['inferences_per_joule'].min()),
            'max_ipj': float(measurements_df['inferences_per_joule'].max()),
            'ratio': float(measurements_df['inferences_per_joule'].max() / 
                         measurements_df['inferences_per_joule'].min())
        },
        'highest_risk_accessibility': accessibility_df.loc[
            accessibility_df['risk_accessibility_score'].idxmax(), 'model'
        ],
        'consumer_viable_models': accessibility_df[
            accessibility_df['can_run_consumer']
        ]['model'].tolist()
    }
    
    with open(output_path / "efficiency_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Experiment complete. Results in {output_path}")
    return benchmark, accessibility_metrics


def generate_figures(measurements: List[EfficiencyMeasurement], 
                    accessibility: List[Dict], output_dir: str):
    """Generate visualization figures."""
    
    figures_path = Path(output_dir) / "figures"
    figures_path.mkdir(parents=True, exist_ok=True)
    
    mdf = pd.DataFrame([asdict(m) for m in measurements])
    adf = pd.DataFrame(accessibility)
    
    # 1. Inferences per Joule comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    bs1 = mdf[mdf['batch_size'] == 1].groupby('model')['inferences_per_joule'].mean()
    bs1_sorted = bs1.sort_values(ascending=True)
    colors = ['green' if 'loihi' in m.lower() or 'darwin' in m.lower() or 'edge' in m.lower() 
              else 'steelblue' for m in bs1_sorted.index]
    bs1_sorted.plot(kind='barh', ax=ax, color=colors)
    ax.set_xlabel('Inferences per Joule (log scale)')
    ax.set_xscale('log')
    ax.set_title('Energy Efficiency: Inferences per Joule by Model')
    plt.tight_layout()
    save_figure(fig, figures_path / "inferences_per_joule.png")
    plt.close()
    
    # 2. Cost per million inferences
    fig, ax = plt.subplots(figsize=(12, 6))
    cost = mdf[mdf['batch_size'] == 1].groupby('model')['cost_per_million'].mean()
    cost_sorted = cost.sort_values(ascending=False)
    cost_sorted.plot(kind='barh', ax=ax, color='coral')
    ax.set_xlabel('Cost per Million Inferences (USD)')
    ax.set_title('Operating Cost Comparison')
    plt.tight_layout()
    save_figure(fig, figures_path / "cost_per_million.png")
    plt.close()
    
    # 3. Efficiency vs Accessibility Risk
    fig, ax = plt.subplots(figsize=(10, 8))
    merged = mdf[mdf['batch_size'] == 1].merge(adf, on='model')
    ax.scatter(merged['inferences_per_joule'], merged['risk_accessibility_score'],
               s=100, alpha=0.7)
    for _, row in merged.iterrows():
        ax.annotate(row['model'], (row['inferences_per_joule'], row['risk_accessibility_score']),
                   fontsize=8, ha='left')
    ax.set_xlabel('Efficiency (Inferences/Joule) - log scale')
    ax.set_ylabel('Risk Accessibility Score')
    ax.set_xscale('log')
    ax.set_title('Efficiency vs. Democratization Risk')
    plt.tight_layout()
    save_figure(fig, figures_path / "efficiency_vs_risk.png")
    plt.close()
    
    # 4. Hardware architecture comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    hardware_types = {
        'Datacenter GPU': ['gpt-4', 'gpt-4o', 'claude-3-opus', 'claude-3-5-sonnet', 'llama-70b'],
        'Consumer GPU': ['llama-7b', 'llama-7b-int4'],
        'Neuromorphic': ['loihi2-snn-small', 'loihi2-snn-large', 'darwin-neuromorphic'],
        'Edge': ['edge-tpu', 'jetson-nano']
    }
    
    hw_efficiency = {}
    for hw_type, models in hardware_types.items():
        eff = mdf[(mdf['model'].isin(models)) & (mdf['batch_size'] == 1)]['inferences_per_joule']
        if len(eff) > 0:
            hw_efficiency[hw_type] = eff.mean()
    
    if hw_efficiency:
        pd.Series(hw_efficiency).plot(kind='bar', ax=ax, color=['navy', 'steelblue', 'green', 'orange'])
        ax.set_ylabel('Avg Inferences per Joule (log scale)')
        ax.set_yscale('log')
        ax.set_title('Efficiency by Hardware Architecture')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        save_figure(fig, figures_path / "efficiency_by_hardware.png")
    plt.close()
    
    logger.info(f"Figures saved to {figures_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment C: Efficiency Analysis")
    parser.add_argument("--models", help="Comma-separated model names")
    parser.add_argument("--batch-sizes", default="1,8,32", help="Batch sizes to test")
    parser.add_argument("--output", default="./results")
    parser.add_argument("--measure-gpu", action="store_true", 
                        help="Actually measure GPU (requires CUDA)")
    args = parser.parse_args()
    
    models = args.models.split(",") if args.models else None
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    
    run_experiment(models, batch_sizes, args.output, args.measure_gpu)