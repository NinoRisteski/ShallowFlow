import torch
from transformers import PreTrainedModel
from .base import BaseTrainer, TrainingConfig
from ..utils.memory import MemoryOptimizer
from typing import Optional

class LLMTrainer(BaseTrainer):
    def __init__(
        self,
        model: PreTrainedModel,
        config: Optional[TrainingConfig] = None
    ):
        super().__init__(model, config)
        self.memory_optimizer = MemoryOptimizer()
        self.batch_size = self.memory_optimizer.get_optimal_batch_size(model)
        
    def train(self, train_dataset, eval_dataset=None):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        
        for epoch in range(self.config.num_epochs):
            self._train_epoch(train_dataset, optimizer)
            if eval_dataset:
                self._evaluate(eval_dataset)
                
    def _train_epoch(self, dataset, optimizer):
        self.model.train()
        for batch in dataset:
            loss = self._training_step(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()