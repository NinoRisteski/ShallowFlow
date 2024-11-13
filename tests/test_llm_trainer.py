import pytest
import torch
from src.shallowflow.trainer import LLMTrainer
from transformers import AutoTokenizer
from src.shallowflow.utils.config import LLMConfig, LoRAConfig, QuantizationConfig

@pytest.fixture
def training_config():
    return LLMConfig(
        model_name='sshleifer/tiny-gpt2',
        gpu_memory=4096,
        use_lora=True,
        device='cuda',
        batch_size=32,
        num_epochs=10,
        lora_rank=4,
        lora_config=LoRAConfig(),
        use_quantization=True,
        learning_rate=0.001,
        quantization_config=QuantizationConfig(
            dtype=torch.qint8,
            layers_to_quantize=["Linear"]
        )
    )

def test_trainer_initialization(small_model, training_config):
    tokenizer = AutoTokenizer.from_pretrained(training_config.model_name)
    trainer = LLMTrainer(
        model=small_model,
        config=training_config,
        tokenizer=tokenizer
    )
    assert trainer.model is not None
    assert trainer.config == training_config

def test_trainer_to_device(small_model, training_config):
    tokenizer = AutoTokenizer.from_pretrained(training_config.model_name)
    trainer = LLMTrainer(
        model=small_model,
        config=training_config,
        tokenizer=tokenizer
    )
    expected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert next(trainer.model.parameters()).device == expected_device

def test_training_step(small_model, training_config, sample_batch):
    training_config.device = 'cpu'
    
    tokenizer = AutoTokenizer.from_pretrained(training_config.model_name)
    trainer = LLMTrainer(
        model=small_model,
        config=training_config,
        tokenizer=tokenizer
    )
    
    # Create input_ids and attention_mask with matching sequence length
    seq_length = 20
    sample_batch['input_ids'] = torch.randint(0, 50257, (1, seq_length))
    # Create matching attention mask
    sample_batch['attention_mask'] = torch.ones((1, seq_length))
    sample_batch['labels'] = sample_batch['input_ids'].clone()
    
    loss = trainer._training_step(sample_batch)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_memory_tracking(small_model, training_config):
    tokenizer = AutoTokenizer.from_pretrained(training_config.model_name)
    trainer = LLMTrainer(
        model=small_model,
        config=training_config,
        tokenizer=tokenizer
    )
    stats = trainer.memory_manager.get_stats()
    assert stats.gpu_used > 0
    assert stats.gpu_total == training_config.gpu_memory * 1024**3
