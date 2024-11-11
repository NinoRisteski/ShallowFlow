from dataclasses import dataclass
from typing import List, Dict, Optional, Union, Any
from transformers import PreTrainedTokenizer
import torch
import numpy as np

@dataclass
class ProcessorConfig:
    max_length: int = 512
    padding: str = "max_length"
    truncation: bool = True
    return_tensors: str = "pt"
    add_special_tokens: bool = True

class TextProcessor:
    """
    Text processing utility for LLM training data
    """
    def __init__(
        self,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        config: Optional[ProcessorConfig] = None
    ):
        self.tokenizer = tokenizer
        self.config = config or ProcessorConfig()
        
    def process_text(
        self,
        text: Union[str, List[str]]
    ) -> Dict[str, torch.Tensor]:
        """
        Process raw text input
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized")
            
        # Tokenize text
        encoded = self.tokenizer(
            text,
            max_length=self.config.max_length,
            padding=self.config.padding,
            truncation=self.config.truncation,
            return_tensors=self.config.return_tensors,
            add_special_tokens=self.config.add_special_tokens
        )
        
        return encoded
        
    def process_batch(
        self,
        batch: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """
        Process a batch of text data
        """
        texts = [item['text'] for item in batch]
        
        # Process all texts in batch
        processed = self.process_text(texts)
        
        # Add any additional features from batch
        for key in batch[0].keys():
            if key != 'text':
                processed[key] = torch.tensor([item[key] for item in batch])
                
        return processed
        
    def create_attention_mask(
        self,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Create attention mask for input sequences
        """
        return (input_ids != self.tokenizer.pad_token_id).float()
        
    def decode(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True
    ) -> List[str]:
        """
        Decode token IDs back to text
        """
        return self.tokenizer.batch_decode(
            token_ids,
            skip_special_tokens=skip_special_tokens
        )
        
class DatasetProcessor:
    """
    Dataset processing utility for handling different data formats
    """
    def __init__(
        self,
        text_processor: TextProcessor,
        max_samples: Optional[int] = None
    ):
        self.text_processor = text_processor
        self.max_samples = max_samples
        
    def process_dataset(
        self,
        dataset: List[Dict[str, Any]]
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Process entire dataset
        """
        if self.max_samples:
            dataset = dataset[:self.max_samples]
            
        processed_data = []
        for batch in self._create_batches(dataset):
            processed_batch = self.text_processor.process_batch(batch)
            processed_data.append(processed_batch)
            
        return processed_data
        
    def _create_batches(
        self,
        dataset: List[Dict[str, Any]],
        batch_size: int = 32
    ):
        """
        Create batches from dataset
        """
        for i in range(0, len(dataset), batch_size):
            yield dataset[i:i + batch_size]
            
class DataAugmentation:
    """
    Data augmentation techniques for text data
    """
    def __init__(self, probability: float = 0.1):
        self.probability = probability
        
    def random_mask(
        self,
        text: str,
        mask_token: str = "[MASK]"
    ) -> str:
        """
        Randomly mask words in text
        """
        words = text.split()
        for i in range(len(words)):
            if np.random.random() < self.probability:
                words[i] = mask_token
        return " ".join(words)
        
    def random_delete(self, text: str) -> str:
        """
        Randomly delete words from text
        """
        words = text.split()
        words = [word for word in words if np.random.random() > self.probability]
        return " ".join(words)