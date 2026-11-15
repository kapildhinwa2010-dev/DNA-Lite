# DNA-Lite

[![CI](https://github.com/kapildhinwa2010-dev/DNA-Lite/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/DNA-Lite/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kapildhinwa2010-dev/DNA-Lite/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/DNA-Lite/actions/workflows/codeql.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Overview

**DNA-Lite** is a lightweight implementation of DNA (Distributed Neural Architecture) optimized for small models and limited compute resources. It provides the core DNA functionality with a simplified architecture.

### Key Features

- ✅ **Lightweight**: Optimized for small models (< 1M parameters)
- ✅ **Fast**: Minimal overhead for weight generation
- ✅ **Low Memory**: Efficient implementation
- ✅ **GPU Accelerated**: Full CUDA support
- ✅ **PyTorch Integration**: Seamless integration with PyTorch models

## Installation

```bash
pip install torch numpy pytest
```

Or install from source:

```bash
git clone https://github.com/kapildhinwa2010-dev/DNA-Lite.git
cd DNA-Lite
pip install -e .
```

## Usage

### Basic Weight Generation

```python
from dna_lite import DNALite

# Create DNA-Lite instance
dna = DNALite(num_freqs=32, hidden_size=64)

# Generate a weight matrix
W = dna.generate_weights(100, 100)
print(f"Generated matrix: {W.shape}")
```

### Generate for Multiple Layers

```python
# Generate weights for multiple layers
layer_config = {
    'fc1': (256, 128),
    'fc2': (128, 256),
}
weights = dna.generate_layer_weights(layer_config)

for name, W in weights.items():
    print(f"{name}: {W.shape}")
```

## Architecture

### Simplified Fourier Feature Generation

DNA-Lite uses a simplified neural network to map coordinates to weights:

```
Coordinates → Fourier Features → Lightweight Network → Weights
```

### Network Structure

The DNA-Lite network consists of:
- Input: Fourier features (num_freqs * 2 dimensions)
- Hidden layer: 1 layer with SiLU activation (default: 64 hidden size)
- Output: Scalar weight value
- Trainable scale parameter

### Advantages

- **Faster Training**: Fewer parameters to train
- **Lower Memory**: Smaller network footprint
- **Simpler Architecture**: Easier to understand and debug
- **Good for Small Models**: Optimized for models under 1M parameters

## Comparison with DNA-Pro

| Feature | DNA-Lite | DNA-Pro |
|---------|----------|---------|
| Network Depth | 1 hidden layer | 3 hidden layers |
| Parameters | ~4K (default) | ~257K (default) |
| Memory Usage | Low | Medium |
| Speed | Fast | Medium |
| Best For | Small models | Large models |

## API Reference

### DNALite

```python
DNALite(
    num_freqs: int = 32,
    hidden_size: int = 64
)
```

**Methods:**
- `generate_weights(out_dim: int, in_dim: int) -> Tensor`: Generate a weight matrix
- `generate_layer_weights(layer_config: Dict) -> Dict`: Generate weights for multiple layers
- `count_parameters() -> int`: Count DNA network parameters

## Testing

Run the test suite:

```bash
python -m pytest tests/ -v
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work builds upon research in:
- Fourier feature networks
- Neural network compression
- Implicit neural representations
- Lightweight deep learning
