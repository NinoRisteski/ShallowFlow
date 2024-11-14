import unittest
from unittest.mock import patch, MagicMock
import torch
from src.shallowflow.strategies.fsdp import FSDPStrategy, FSDPConfig
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.distributed.fsdp import MixedPrecision
from unittest import mock

class TestFSDPStrategy(unittest.TestCase):
    def setUp(self):
        # Initialize a simple model for testing
        self.model = torch.nn.Linear(10, 10)
        self.config = FSDPConfig(
            min_num_params=1000,
            cpu_offload=True,
            mixed_precision=True,
            backward_prefetch=False,
            activation_checkpointing=True
        )
        self.strategy = FSDPStrategy(config=self.config)
    
    @patch('src.shallowflow.strategies.fsdp.wrap')
    @patch('src.shallowflow.strategies.fsdp.enable_wrap')
    @patch('torch.cuda.is_available', return_value=False)
    @patch('torch.distributed.fsdp.MixedPrecision')

    def test_prepare_model(self, mixed_precision, is_available, enable_wrap, wrap):
        # Mock the wrapped model
        wrapped_fsdp = MagicMock(spec=FullyShardedDataParallel)
        wrap.return_value = wrapped_fsdp
        enable_wrap.return_value.__enter__.return_value = None
        enable_wrap.return_value.__exit__.return_value = None
        
        # Call the method under test
        wrapped_model = self.strategy.prepare_model(self.model)
        
        # Assertions to ensure wrapping was called correctly
        enable_wrap.assert_called_once()
        wrap.assert_called_once_with(self.model)
        self.assertIs(wrapped_model, wrapped_fsdp)
    
    def test_prepare_optimizer(self):
        # Mock an FSDP wrapped model
        mock_model = MagicMock(spec=FullyShardedDataParallel)
        mock_params = [torch.nn.Parameter(torch.randn(2, 2))]
        mock_model.parameters.return_value = mock_params
        
        optimizer = self.strategy.prepare_optimizer(
            mock_model, 
            torch.optim.Adam,
            lr=0.001
        )
        
        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertEqual(optimizer.defaults['lr'], 0.001)
        self.assertEqual(optimizer.param_groups[0]['params'], mock_params)

    def test_init_with_default_config(self):
        # Initialize strategy without passing a config
        default_strategy = FSDPStrategy()
        self.assertIsInstance(default_strategy.config, FSDPConfig)
        self.assertEqual(default_strategy.config.min_num_params, 1e6)
        self.assertFalse(default_strategy.config.cpu_offload)
        self.assertTrue(default_strategy.config.mixed_precision)
        self.assertTrue(default_strategy.config.backward_prefetch)
        self.assertFalse(default_strategy.config.activation_checkpointing)
    
    @patch('torch.distributed.fsdp.MixedPrecision')
    @patch('src.shallowflow.strategies.fsdp.StrategyInstance')
    def test_get_mixed_precision_policy(self, strategy_instance, mixed_precision):
        # Create a mock MixedPrecision instance with the expected attributes
        mock_policy = MagicMock()
        mock_policy.param_dtype = torch.float16
        mock_policy.reduce_dtype = torch.float16
        mock_policy.buffer_dtype = torch.float16
        mixed_precision.return_value = mock_policy

        # Test when mixed precision is enabled
        policy = self.strategy._get_mixed_precision_policy()
        
        # Verify MixedPrecision was called with correct arguments
        mixed_precision.assert_called_once_with(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16
        )
        
        # Verify the returned policy has correct attributes
        self.assertEqual(policy.param_dtype, torch.float16)
        self.assertEqual(policy.reduce_dtype, torch.float16)
        self.assertEqual(policy.buffer_dtype, torch.float16)

        # Test when mixed precision is disabled
        self.strategy.config.mixed_precision = False
        policy = self.strategy._get_mixed_precision_policy()
        self.assertIsNone(policy)

    def test_mixed_precision_configuration(self):
        # Test the full configuration with mixed precision
        strategy = FSDPStrategy(FSDPConfig(mixed_precision=True))
        policy = strategy._get_mixed_precision_policy()
        
        self.assertIsInstance(policy, MixedPrecision)
        self.assertEqual(policy.param_dtype, torch.float16)
        self.assertEqual(policy.reduce_dtype, torch.float16)
        self.assertEqual(policy.buffer_dtype, torch.float16)
    
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