from dataclasses import dataclass
from typing import Optional
import sagemaker
from sagemaker.huggingface import HuggingFace

@dataclass
class SageMakerConfig:
    instance_type: str = "ml.g4dn.xlarge"
    instance_count: int = 1
    max_epochs: int = 3
    learning_rate: float = 3e-4

class SageMakerManager:
    def __init__(self, config: SageMakerConfig):
        self.config = config
        self.session = sagemaker.Session()
        
    def setup_training(
        self,
        model_name: str,
        script_path: str
    ):
        # Create HuggingFace Estimator
        estimator = HuggingFace(
            entry_point=script_path,
            instance_type=self.config.instance_type,
            instance_count=self.config.instance_count,
            transformers_version="4.26.0",
            pytorch_version="1.13.1",
            py_version="py39",
            role=sagemaker.get_execution_role(),
            hyperparameters={
                "epochs": self.config.max_epochs,
                "learning_rate": self.config.learning_rate,
                "model_name": model_name
            }
        )
        
        return estimator