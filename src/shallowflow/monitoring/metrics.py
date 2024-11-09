from dataclasses import dataclass
from typing import List, Dict
import time
import wandb

@dataclass
class TrainingMetrics:
    loss: float
    learning_rate: float
    gpu_memory_used: float
    throughput: float
    cost_per_hour: float

class MetricsTracker:
    def __init__(self, project_name: str, use_wandb: bool = True):
        self.metrics_history = []
        self.start_time = time.time()
        if use_wandb:
            wandb.init(project=project_name)
            
    def log_metrics(self, metrics: TrainingMetrics):
        self.metrics_history.append(metrics)
        if wandb.run is not None:
            wandb.log({
                'loss': metrics.loss,
                'learning_rate': metrics.learning_rate,
                'gpu_memory_used': metrics.gpu_memory_used,
                'throughput': metrics.throughput,
                'cost_per_hour': metrics.cost_per_hour
            })