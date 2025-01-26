from dataclasses import dataclass
import boto3
from typing import Optional, Dict
from datetime import datetime, timedelta
import time

@dataclass
class AWSConfig:
    instance_type: str = "g4dn.xlarge"
    region: str = "us-west-2"
    spot_instance: bool = True
    volume_size: int = 100
    use_lora: bool = False
    ami_id: Optional[str] = None
    vpc_id: Optional[str] = None
    subnet_id: Optional[str] = None
    key_name: Optional[str] = None
    tags: Optional[Dict[str, str]] = None

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class AWSManager:
    def __init__(self, config: AWSConfig):
        """Initialize AWS manager with configuration."""
        self.config = config
        self.session = boto3.Session(region_name=config.region)
        self.ec2 = self.session.client('ec2')
        self.cloudwatch = self.session.client('cloudwatch')
        self._instance_id = None
        
    def _get_deep_learning_ami(self) -> str:
        """Get latest deep learning AMI ID."""
        if self.config.ami_id:
            return self.config.ami_id
            
        filters = [{
            'Name': 'name',
            'Values': ['Deep Learning AMI GPU PyTorch*']
        }]
        
        response = self.ec2.describe_images(
            Filters=filters,
            Owners=['amazon']
        )
        
        # Sort by creation date and get the latest
        amis = sorted(
            response['Images'],
            key=lambda x: x['CreationDate'],
            reverse=True
        )
        return amis[0]['ImageId'] if amis else None

    def _launch_spot_instance(self, launch_template: Dict) -> Dict:
        """Launch a spot instance with the given template."""
        try:
            spot_request = self.ec2.request_spot_instances(
                InstanceCount=1,
                LaunchSpecification=launch_template,
                Type='one-time'
            )
            
            request_id = spot_request['SpotInstanceRequests'][0]['SpotInstanceRequestId']
            
            # Wait for spot instance to be active
            waiter = self.ec2.get_waiter('spot_instance_request_fulfilled')
            waiter.wait(SpotInstanceRequestIds=[request_id])
            
            # Get instance ID
            response = self.ec2.describe_spot_instance_requests(
                SpotInstanceRequestIds=[request_id]
            )
            self._instance_id = response['SpotInstanceRequests'][0]['InstanceId']
            
            return {'InstanceId': self._instance_id}
            
        except Exception as e:
            raise RuntimeError(f"Failed to launch spot instance: {str(e)}")

    def _launch_on_demand_instance(self, launch_template: Dict) -> Dict:
        """Launch an on-demand instance with the given template."""
        try:
            response = self.ec2.run_instances(
                **launch_template,
                MinCount=1,
                MaxCount=1
            )
            self._instance_id = response['Instances'][0]['InstanceId']
            return {'InstanceId': self._instance_id}
            
        except Exception as e:
            raise RuntimeError(f"Failed to launch on-demand instance: {str(e)}")

    def launch_instance(self) -> Dict:
        """Launch an EC2 instance based on configuration."""
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
        
        # Add optional configurations
        if self.config.key_name:
            launch_template['KeyName'] = self.config.key_name
        if self.config.subnet_id:
            launch_template['SubnetId'] = self.config.subnet_id
        if self.config.tags:
            launch_template['TagSpecifications'] = [{
                'ResourceType': 'instance',
                'Tags': [{'Key': k, 'Value': v} for k, v in self.config.tags.items()]
            }]
        
        if self.config.spot_instance:
            return self._launch_spot_instance(launch_template)
        return self._launch_on_demand_instance(launch_template)

    def terminate_instance(self) -> None:
        """Terminate the launched EC2 instance."""
        if self._instance_id:
            try:
                self.ec2.terminate_instances(InstanceIds=[self._instance_id])
                self._instance_id = None
            except Exception as e:
                raise RuntimeError(f"Failed to terminate instance: {str(e)}")

    def get_instance_status(self) -> str:
        """Get current status of the launched instance."""
        if not self._instance_id:
            return "No instance launched"
            
        response = self.ec2.describe_instances(InstanceIds=[self._instance_id])
        if not response['Reservations']:
            return "Instance not found"
            
        return response['Reservations'][0]['Instances'][0]['State']['Name']

    def get_training_costs(self, days: int = 1) -> float:
        """Calculate training costs based on usage for the specified number of days."""
        if not self._instance_id:
            return 0.0
            
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        
        try:
            response = self.cloudwatch.get_metric_statistics(
                Namespace='AWS/EC2',
                MetricName='CPUUtilization',
                Dimensions=[{'Name': 'InstanceId', 'Value': self._instance_id}],
                StartTime=start_time,
                EndTime=end_time,
                Period=3600,
                Statistics=['Average']
            )
            
            # Get instance pricing (simplified - you might want to use AWS Price List API for accurate pricing)
            instance_prices = {
                'g4dn.xlarge': 0.526,  # USD per hour
                'p3.2xlarge': 3.06,
                'p3.8xlarge': 12.24
            }
            
            hourly_rate = instance_prices.get(self.config.instance_type, 0.0)
            if self.config.spot_instance:
                hourly_rate *= 0.3  # Approximate spot instance discount
                
            hours_running = len(response['Datapoints'])
            return round(hours_running * hourly_rate, 2)
            
        except Exception as e:
            raise RuntimeError(f"Failed to calculate training costs: {str(e)}")

    def wait_for_instance_ready(self, timeout: int = 300) -> bool:
        """Wait for instance to be ready and running."""
        if not self._instance_id:
            return False
            
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.get_instance_status()
            if status == 'running':
                return True
            elif status in ['terminated', 'shutting-down']:
                return False
            time.sleep(10)
            
        return False