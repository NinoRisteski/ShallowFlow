import pytest
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.shallowflow.config import LLMConfig


@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture(scope="session")
def small_model():
    return AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")

@pytest.fixture(scope="session")
def tokenizer():
    return AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")

@pytest.fixture
def sample_batch():
    return {
        "input_ids": torch.randint(0, 1000, (4, 128)),
        "attention_mask": torch.ones(4, 128)
    }

@pytest.fixture
def training_config():
    return LLMConfig(
        device="cpu",
        model_name="sshleifer/tiny-gpt2",
        batch_size=4,
        num_epochs=1
    )

