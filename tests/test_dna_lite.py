"""
Tests for DNA-Lite module
"""

import pytest
import torch
from dna_lite import DNALite


class TestDNALiteInitialization:
    """Test DNALite initialization"""
    
    def test_default_initialization(self):
        """Test default initialization"""
        dna = DNALite()
        assert dna.num_freqs == 32
        assert dna.hidden_size == 64
    
    def test_custom_initialization(self):
        """Test custom initialization"""
        dna = DNALite(num_freqs=16, hidden_size=32)
        assert dna.num_freqs == 16
        assert dna.hidden_size == 32
    
    def test_buffer_registration(self):
        """Test that B is registered as buffer"""
        dna = DNALite()
        assert hasattr(dna, 'B')
        assert isinstance(dna.B, torch.Tensor)
        assert dna.B.shape == (32, 2)
    
    def test_parameter_count(self):
        """Test parameter counting"""
        dna = DNALite(num_freqs=16, hidden_size=32)
        count = dna.count_parameters()
        # Linear(32->32)=1056, Linear(32->1)=33, learnable scale=1 => 1090
        expected = (32*32 + 32) + (32*1 + 1) + 1
        assert count == expected


class TestDNALiteWeightGeneration:
    """Test weight generation"""
    
    def test_generate_small_matrix(self):
        """Test generating small weight matrix"""
        dna = DNALite()
        W = dna.generate_weights(10, 10)
        assert W.shape == (10, 10)
        assert W.dtype == torch.float32
    
    def test_generate_medium_matrix(self):
        """Test generating medium weight matrix"""
        dna = DNALite()
        W = dna.generate_weights(100, 100)
        assert W.shape == (100, 100)
    
    def test_generate_matrix_device(self):
        """Test weight generation on CPU"""
        dna = DNALite()
        W = dna.generate_weights(10, 10, device=torch.device('cpu'))
        assert W.device.type == 'cpu'
    
    def test_weight_matrix_properties(self):
        """Test weight matrix statistical properties"""
        dna = DNALite()
        W = dna.generate_weights(100, 100)
        # Check that weights are not all zeros
        assert W.abs().sum() > 0
        # Check that weights have reasonable scale
        assert W.std() > 0


class TestDNALiteLayerWeights:
    """Test layer weight generation"""
    
    def test_generate_layer_weights(self):
        """Test generating weights for multiple layers"""
        dna = DNALite()
        layer_config = {
            'fc1': (256, 128),
            'fc2': (128, 256),
        }
        weights = dna.generate_layer_weights(layer_config)
        
        assert 'fc1' in weights
        assert 'fc2' in weights
        assert weights['fc1'].shape == (256, 128)
        assert weights['fc2'].shape == (128, 256)


class TestDNALiteDeviceAndDtype:
    """Test device and dtype handling"""
    
    def test_device_consistency(self):
        """Test that all tensors are on same device"""
        dna = DNALite()
        W = dna.generate_weights(10, 10)
        assert W.device == dna.B.device
        assert W.device == dna.scale.device
        for param in dna.net.parameters():
            assert param.device == dna.B.device
    
    def test_dtype_consistency(self):
        """Test that all tensors have same dtype"""
        dna = DNALite()
        W = dna.generate_weights(10, 10)
        assert W.dtype == dna.B.dtype
        assert W.dtype == dna.scale.dtype


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
