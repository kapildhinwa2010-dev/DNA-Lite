# DNA-Lite

[![CI](https://github.com/kapildhinwa2010-dev/DNA-Lite/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/DNA-Lite/actions/workflows/ci.yml)
[![CI](https://github.com/kapildhinwa2010-dev/DNA-Lite/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/DNA-Lite/actions/workflows/codeql.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

Smallest member of the DNA weight-generator family: one hidden layer.

## What it does

DNALite maps a 2D coordinate grid through Fourier features and a
single-hidden-layer SiLU MLP (~4K parameters by default) to scalar
weight values, for matrices of any requested shape.

## Installation

    pip install torch numpy pytest

## Usage

    from dna_lite import DNALite

    dna = DNALite(num_freqs=32, hidden_size=64)
    W = dna.generate_weights(100, 100)
    weights = dna.generate_layer_weights({"fc1": (256, 128)})

## API

- DNALite(num_freqs=32, hidden_size=64)
- generate_weights(out_dim, in_dim), generate_layer_weights(config),
  count_parameters()

## Family comparison

| | DNA-Lite | DNA-Pro |
|---|---|---|
| Hidden layers | 1 | 3 |
| Generator params | ~4K | ~257K |
| Best for | experiments, small shapes | larger shapes, smoother fields |

## Honest results and limitations

- Fewer generator parameters mean a smoother, less expressive field:
  reconstruction of any structured matrix is worse than DNA-Pro at the
  same budget. That is the trade, stated plainly.
- An untrained generator produces a smooth random field, not useful
  pretrained weights; it must be fitted to a target matrix first.
- "Lightweight" refers to the generator size only. Materializing a
  matrix still costs its full dense size in memory at generation time.
- No accuracy or reconstruction-error numbers are reported because none
  were measured in a controlled experiment.

## Negative results (from the source experiments)

- The reserv.dean-part-2 log records that generated-weight
  reconstructions of trained models were not usable without per-layer
  fitting. The robust storage design kept in the toolkit is the
  Reed-Solomon slot lattice (ECC-Memory).

## License

MIT - see [LICENSE](LICENSE).
