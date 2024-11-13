import pytest
import torch
from src.shallowflow.trainer import LLMTrainer

def test_trainer_initialization(small_model, training_config):
    trainer = LLMTrainer(small_model, training_config)
    assert trainer.model is not None
    assert trainer.config == training_config

def test_trainer_to_device(small_model, training_config):
    trainer = LLMTrainer(
        model=small_model,
        config=training_config
    )
    expected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert next(trainer.model.parameters()).device == expected_device

def test_training_step(small_model, training_config, sample_batch):
    trainer = LLMTrainer(small_model, training_config)
    loss = trainer._training_step(sample_batch)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_memory_tracking(small_model, training_config):
    trainer = LLMTrainer(small_model, training_config)
    stats = trainer.memory_manager.get_stats()
    assert stats.gpu_used > 0
    assert stats.gpu_total == training_config.gpu_memory * 1024**3

