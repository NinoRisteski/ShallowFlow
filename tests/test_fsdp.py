import unittest
from unittest.mock import patch, MagicMock
import torch
from src.shallowflow.strategies.fsdp import FSDPStrategy, FSDPConfig
from torch.distributed.fsdp import FullyShardedDataParallel

class TestFSDPStrategy(unittest.TestCase):
    def setUp(self):
        # Initialize a simple model for testing
        self.model = torch.nn.Linear(10, 10)
        self.config = FSDPConfig(
            min_num_params=1000,
            cpu_offload=True,
            mixed_precision=False,
            backward_prefetch=False,
            activation_checkpointing=True
        )
        self.strategy = FSDPStrategy(config=self.config)
    
    @patch('src.shallowflow.strategies.fsdp.wrap')
    @patch('src.shallowflow.strategies.fsdp.enable_wrap')
    @patch('torch.cuda.is_available', return_value=False)
    @patch('torch.cuda.current_device', return_value=0)
    @patch('torch.cuda.device_count')
    def test_prepare_model(self, mock_device_count, mock_current_device, mock_cuda_available, mock_enable_wrap, mock_wrap):
        # Mock the wrapped model
        wrapped_fsdp = MagicMock(spec=FullyShardedDataParallel)
        mock_wrap.return_value = wrapped_fsdp
        mock_enable_wrap.return_value.__enter__.return_value = None  
        
        # Call the method under test
        wrapped_model = self.strategy.prepare_model(self.model)
        
        # Assertions to ensure wrapping was called correctly
        mock_enable_wrap.assert_called_once()
        mock_wrap.assert_called_once_with(self.model)
        self.assertIs(wrapped_model, wrapped_fsdp)
    
    def test_prepare_optimizer(self):
        # Choose an optimizer class for testing
        optimizer_class = torch.optim.Adam
        
        # Mock the model parameters
        with patch.object(self.model, 'parameters', return_value=self.model.parameters()):
            # Call the method under test
            optimizer = self.strategy.prepare_optimizer(self.model, optimizer_class, lr=0.001)
            
            # Assertions to ensure optimizer is created correctly
            self.assertIsInstance(optimizer, optimizer_class)
            self.assertEqual(optimizer.defaults['lr'], 0.001)
            self.assertIn('params', optimizer.param_groups[0])

    def test_init_with_default_config(self):
        # Initialize strategy without passing a config
        default_strategy = FSDPStrategy()
        self.assertIsInstance(default_strategy.config, FSDPConfig)
        self.assertEqual(default_strategy.config.min_num_params, 1e6)
        self.assertFalse(default_strategy.config.cpu_offload)
        self.assertTrue(default_strategy.config.mixed_precision)
        self.assertTrue(default_strategy.config.backward_prefetch)
        self.assertFalse(default_strategy.config.activation_checkpointing)
    
    def test_get_mixed_precision_policy(self):
        # Ensure mixed precision is enabled in the config
        self.strategy.config.mixed_precision = True
        
        # Test mixed precision policy when enabled
        with patch('torch.distributed.fsdp.MixedPrecision') as mock_mixed_precision:
            # Call the method to test
            policy = self.strategy._get_mixed_precision_policy()
            
            # Verify the mock was called correctly
            mock_mixed_precision.assert_called_once_with(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16
            )
        
        # Test mixed precision policy when disabled
        self.strategy.config.mixed_precision = False
        policy = self.strategy._get_mixed_precision_policy()
        self.assertIsNone(policy)
    
    def test_get_cpu_offload(self):
        # Test CPU offload when enabled
        with patch('src.shallowflow.strategies.fsdp.CPUOffload') as mock_cpu_offload:
            offload = self.strategy._get_cpu_offload()
            mock_cpu_offload.assert_called_once_with(offload_params=True)
            self.assertIsNotNone(offload)
        
        # Test CPU offload when disabled
        self.strategy.config.cpu_offload = False
        offload = self.strategy._get_cpu_offload()
        self.assertIsNone(offload)

if __name__ == '__main__':
    unittest.main()