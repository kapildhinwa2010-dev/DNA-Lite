"""
DNA-Lite - Lightweight DNA for small models

This module provides a lightweight implementation of DNA (Distributed
Neural Architecture) optimized for small models and limited compute.

Features:
- Lightweight DNA implementation
- Optimized for small models (< 1M parameters)
- Low memory usage
- Fast inference
- GPU and CPU support
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Tuple


class DNALite(nn.Module):
    """
    DNA-Lite - Lightweight DNA generator for small models.
    
    This class generates neural network weights using a simplified
    Fourier feature approach, optimized for small models.
    
    Example:
        >>> dna = DNALite(num_freqs=32, hidden_size=64)
        >>> # Generate weights for a layer
        >>> W = dna.generate_weights(100, 100)
        >>> print(W.shape)  # torch.Size([100, 100])
    
    Args:
        num_freqs: Number of Fourier frequencies (default: 32)
        hidden_size: Size of hidden layer (default: 64)
    """
    
    def __init__(
        self,
        num_freqs: int = 32,
        hidden_size: int = 64
    ):
        super().__init__()
        self.num_freqs = num_freqs
        self.hidden_size = hidden_size
        
        # Fourier projection matrix (fixed)
        self.register_buffer('B', torch.randn(num_freqs, 2) * 20.0)
        input_dim = num_freqs * 2
        
        # Lightweight DNA network
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Learnable scale
        self.scale = nn.Parameter(torch.tensor(0.01))
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with Xavier uniform"""
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def _generate_fourier_features(self, coords: torch.Tensor) -> torch.Tensor:
        """Generate Fourier features from coordinates"""
        proj = 2 * np.pi * coords @ self.B.T
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
    
    def generate_weights(
        self,
        out_dim: int,
        in_dim: int,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Generate a weight matrix of specified size.
        
        Args:
            out_dim: Output dimension
            in_dim: Input dimension
            device: Device to use (optional)
            
        Returns:
            Weight matrix of shape (out_dim, in_dim)
        """
        if device is None:
            device = self.B.device
        
        # Generate coordinate grid
        i = torch.linspace(-1, 1, out_dim, device=device)
        j = torch.linspace(-1, 1, in_dim, device=device)
        grid_i, grid_j = torch.meshgrid(i, j, indexing='ij')
        coords = torch.stack([grid_i.flatten(), grid_j.flatten()], dim=1)
        
        # Generate Fourier features
        feats = self._generate_fourier_features(coords)
        
        # Generate weights
        raw_W = self.net(feats).view(out_dim, in_dim)
        
        # Apply scaling
        xavier_scale = (2.0 / (in_dim + out_dim)) ** 0.5
        W = raw_W * self.scale * xavier_scale
        
        return W
    
    def generate_layer_weights(
        self,
        layer_config: Dict[str, Tuple[int, int]]
    ) -> Dict[str, torch.Tensor]:
        """
        Generate weights for multiple layers.
        
        Args:
            layer_config: Dictionary mapping layer names to (out_dim, in_dim)
            
        Returns:
            Dictionary mapping layer names to weight matrices
        """
        weights = {}
        for name, (out_dim, in_dim) in layer_config.items():
            weights[name] = self.generate_weights(out_dim, in_dim)
        return weights
    
    def count_parameters(self) -> int:
        """Count parameters in DNA network"""
        return sum(p.numel() for p in self.parameters())


# Example usage
if __name__ == "__main__":
    print("DNA-Lite - Lightweight DNA Generator")
    print("=" * 60)
    
    # Create DNA-Lite instance
    dna = DNALite(num_freqs=32, hidden_size=64)
    print(f"DNA-Lite created with {dna.count_parameters():,} parameters")
    
    # Test weight generation
    print("\nGenerating sample weight matrices...")
    
    # Small matrix
    W1 = dna.generate_weights(100, 100)
    print(f"Generated 100x100 matrix: {W1.shape}")
    print(f"  Mean: {W1.mean():.6f}, Std: {W1.std():.6f}")
    
    # Medium matrix
    W2 = dna.generate_weights(200, 200)
    print(f"Generated 200x200 matrix: {W2.shape}")
    print(f"  Mean: {W2.mean():.6f}, Std: {W2.std():.6f}")
    
    print("\n✓ DNA-Lite test completed successfully!")
