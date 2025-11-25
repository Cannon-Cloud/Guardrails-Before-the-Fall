"""
Shared utilities for Guardrails Before the Fall experiments.
Author: Clarence H. Cannon, IV
"""

import os
import json
import time
import random
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables
load_dotenv()

# Configure logging
def setup_logging(name: str, level: str = "INFO") -> logging.Logger:
    """Set up logging with consistent formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

# Set random seeds for reproducibility
def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

@dataclass
class ExperimentConfig:
    """Base configuration for experiments."""
    name: str
    seed: int = 42
    results_dir: str = "./results"
    log_level: str = "INFO"
    
    def __post_init__(self):
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
    
    def save(self, path: Optional[str] = None):
        """Save configuration to JSON."""
        path = path or f"{self.results_dir}/config.json"
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """Load configuration from JSON."""
        with open(path, 'r') as f:
            return cls(**json.load(f))

@dataclass
class ModelResponse:
    """Standardized model response."""
    model: str
    prompt: str
    response: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    timestamp: str
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)

class ModelClient:
    """Unified interface for different model APIs."""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
        self.client = self._init_client()
        self.logger = setup_logging(f"ModelClient:{model_name}")
    
    def _init_client(self):
        """Initialize the appropriate API client."""
        if "gpt" in self.model_name.lower() or "o1" in self.model_name.lower():
            from openai import OpenAI
            return OpenAI(api_key=self.api_key or os.getenv("OPENAI_API_KEY"))
        elif "claude" in self.model_name.lower():
            from anthropic import Anthropic
            return Anthropic(api_key=self.api_key or os.getenv("ANTHROPIC_API_KEY"))
        elif "llama" in self.model_name.lower() or "mistral" in self.model_name.lower():
            # Use Together AI for hosted open models
            from openai import OpenAI
            return OpenAI(
                api_key=self.api_key or os.getenv("TOGETHER_API_KEY"),
                base_url="https://api.together.xyz/v1"
            )
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=60))
    def generate(self, prompt: str, max_tokens: int = 1024, 
                 temperature: float = 0.7, system: str = None) -> ModelResponse:
        """Generate a response from the model."""
        start_time = time.time()
        
        if "claude" in self.model_name.lower():
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system or "You are a helpful assistant.",
                messages=[{"role": "user", "content": prompt}]
            )
            text = response.content[0].text
            tokens_in = response.usage.input_tokens
            tokens_out = response.usage.output_tokens
        else:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            text = response.choices[0].message.content
            tokens_in = response.usage.prompt_tokens
            tokens_out = response.usage.completion_tokens
        
        latency_ms = (time.time() - start_time) * 1000
        
        return ModelResponse(
            model=self.model_name,
            prompt=prompt,
            response=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            timestamp=datetime.now().isoformat()
        )

class ResultsTracker:
    """Track and save experimental results."""
    
    def __init__(self, experiment_name: str, results_dir: str = "./results"):
        self.experiment_name = experiment_name
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[Dict] = []
        self.logger = setup_logging(f"Results:{experiment_name}")
    
    def add(self, result: Dict):
        """Add a result to the tracker."""
        result['_timestamp'] = datetime.now().isoformat()
        result['_experiment'] = self.experiment_name
        self.results.append(result)
    
    def save_csv(self, filename: Optional[str] = None):
        """Save results to CSV."""
        filename = filename or f"{self.experiment_name}_results.csv"
        df = pd.DataFrame(self.results)
        path = self.results_dir / filename
        df.to_csv(path, index=False)
        self.logger.info(f"Saved {len(self.results)} results to {path}")
        return path
    
    def save_json(self, filename: Optional[str] = None):
        """Save results to JSON."""
        filename = filename or f"{self.experiment_name}_results.json"
        path = self.results_dir / filename
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2)
        self.logger.info(f"Saved {len(self.results)} results to {path}")
        return path
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        return pd.DataFrame(self.results)

def load_prompts(path: str) -> List[Dict]:
    """Load prompts from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)

def save_figure(fig, path: str, dpi: int = 150):
    """Save matplotlib figure."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')

def calculate_statistics(data: List[float]) -> Dict[str, float]:
    """Calculate summary statistics."""
    arr = np.array(data)
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'median': float(np.median(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'p25': float(np.percentile(arr, 25)),
        'p75': float(np.percentile(arr, 75)),
        'p95': float(np.percentile(arr, 95)),
        'n': len(arr)
    }

def bootstrap_ci(data: List[float], n_bootstrap: int = 10000, 
                 ci: float = 0.95) -> tuple:
    """Calculate bootstrapped confidence interval."""
    arr = np.array(data)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(arr, size=len(arr), replace=True)
        boot_means.append(np.mean(sample))
    
    alpha = (1 - ci) / 2
    lower = np.percentile(boot_means, alpha * 100)
    upper = np.percentile(boot_means, (1 - alpha) * 100)
    return float(lower), float(upper)