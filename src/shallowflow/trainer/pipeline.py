from typing import Optional, Dict, List 
import torch
from ..optimizations import LoRALayer, Quantizer
from ..monitoring import MetricsTracker
from shallowflow.utils import MemoryManager, AWSManager
from shallowflow.utils.config import TrainingConfig, AWSConfig  
 

class TrainingPipeline:
    def __init__(
        self,
        model,
        config: TrainingConfig,
        aws_config: Optional[AWSConfig] = None
    ):
        self.model = model
        self.config = config
        self.memory_manager = MemoryManager(config.gpu_memory)
        self.metrics_tracker = MetricsTracker(config.project_name)
        
        if aws_config:
            self.aws_manager = AWSManager(aws_config)
            
        self._setup_optimizations()
        
    def _setup_optimizations(self):
        if self.config.use_lora:
            self._apply_lora()
        if self.config.use_quantization:
            self.quantizer = Quantizer(bits=8)
            
    def train(
        self,
        train_dataset,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        callbacks: List = None
    ):
        optimizer = self._setup_optimizer()
        scheduler = self._setup_scheduler(optimizer)
        
        for epoch in range(self.config.num_epochs):
            epoch_metrics = self._train_epoch(
                train_dataset,
                optimizer,
                scheduler
            )
            
            if eval_dataset:
                eval_metrics = self._evaluate(eval_dataset)
                
            self.metrics_tracker.log_metrics({
                **epoch_metrics,
                **eval_metrics
            })