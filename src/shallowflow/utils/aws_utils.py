from dataclasses import dataclass
import boto3
from typing import Optional, Dict

@dataclass
class AWSConfig:
    instance_type: str = "g4dn.xlarge"
    region: str = "us-west-2"
    spot_instance: bool = True
    volume_size: int = 100
    use_lora: bool = False

    def __init__(self, **kwargs):
        self.use_lora = kwargs.get('use_lora', False)

class AWSManager:
    def __init__(self, config: AWSConfig):
        self.config = config
        self.ec2 = boto3.client('ec2')
        self.cloudwatch = boto3.client('cloudwatch')
        
    def launch_instance(self) -> Dict:
        launch_template = {
            'InstanceType': self.config.instance_type,
            'ImageId': self._get_deep_learning_ami(),
            'BlockDeviceMappings': [{
                'DeviceName': '/dev/xvda',
                'Ebs': {
                    'VolumeSize': self.config.volume_size,
                    'VolumeType': 'gp3'
                }
            }]
        }
        
        if self.config.spot_instance:
            return self._launch_spot_instance(launch_template)
        return self._launch_on_demand_instance(launch_template)

    def get_training_costs(self) -> float:
        # Calculate training costs based on usage
        pass