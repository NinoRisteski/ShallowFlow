
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
    
    @patch('torch.distributed.fsdp.FullyShardedDataParallel')
    @patch('torch.distributed.fsdp.wrap.wrap')
    @patch('torch.distributed.fsdp.wrap.enable_wrap')
    def test_prepare_model(self, mock_enable_wrap, mock_wrap, mock_fsdp):
        # Mock the wrapped model
        wrapped_fsdp = MagicMock(spec=FullyShardedDataParallel)
        mock_wrap.return_value = wrapped_fsdp
        mock_enable_wrap.return_value.__enter__.return_value = None  
        
        # Call the method under test
        wrapped_model = self.strategy.prepare_model(self.model)
        
        # Assertions to ensure wrapping was called correctly
        mock_enable_wrap.assert_called_once()
        mock_wrap.assert_called_once_with(self.model)
        self.assertEqual(wrapped_model, wrapped_fsdp)
    
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
        self.assertFalse(default_strategy.config.cpu_offload is False)
        self.assertTrue(default_strategy.config.mixed_precision)
    
    def test_get_mixed_precision_policy(self):
        # Test mixed precision policy when enabled
        with patch('src.shallowflow.strategies.fsdp.MixedPrecision') as mock_mixed_precision:
            policy = self.strategy._get_mixed_precision_policy()
            mock_mixed_precision.assert_called_once_with(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16
            )
            self.assertIsNotNone(policy)
        
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

