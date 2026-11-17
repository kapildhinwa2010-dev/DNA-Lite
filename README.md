# DNA-Lite

DNA-Lite is a small PyTorch module that generates dense weight matrices from a two-dimensional coordinate grid. It applies fixed random Fourier projections, sine/cosine features, and a one-hidden-layer SiLU network to produce one scalar per matrix position. The implementation provides weight generation; it does not provide a training or fitting procedure for matching a target model.

## Current scope

The repository contains one implementation module, `dna_lite.py`, with the `DNALite` class. Its defaults are 32 Fourier frequencies and a hidden size of 64. The class can:

- Generate a matrix with a requested output and input dimension using `generate_weights(out_dim, in_dim)`.
- Generate named matrices from a mapping of layer names to `(out_dim, in_dim)` pairs using `generate_layer_weights(...)`.
- Count the PyTorch parameters in the generator with `count_parameters()`; the fixed Fourier projection is a registered buffer, not a parameter.

The generator is randomly initialized. There is no built-in API for fitting its output to existing weights, and the repository reports no measured reconstruction quality, accuracy, or performance results. Generating a dense matrix still materializes all of its entries in memory.

## Requirements and running from source

The project metadata requires Python 3.10 or newer. Runtime dependencies are PyTorch 2.0 or newer and NumPy 1.24 or newer; `requirements.txt` also includes pytest for tests. From a clone, install the listed dependencies and run the included example:

```bash
python -m pip install -r requirements.txt
python dna_lite.py
```

The example creates a generator and prints shapes and basic statistics for 100×100 and 200×200 generated matrices. There is no separate command-line interface. Although `pyproject.toml` declares project metadata and a wheel can be built, a wheel built from this commit contains only distribution metadata and the license, not `dna_lite.py`; it therefore does not install an importable `dna_lite` module. Use the module directly from the repository checkout:

```python
from dna_lite import DNALite

generator = DNALite(num_freqs=32, hidden_size=64)
weights = generator.generate_weights(out_dim=100, in_dim=100)
layer_weights = generator.generate_layer_weights({"fc1": (256, 128)})
```

`generate_weights` defaults its allocation device to the generator's Fourier-buffer device. To use another device, move the module with PyTorch's `.to(device)` method so its parameters and buffer are on the same device; passing a different `device` to `generate_weights` does not move the module itself.

## Tests

On the reviewed commit, `python -m pytest -q` completed locally with **11 tests passed**. The CI workflow specifies pytest, bytecode compilation, and an import/shape smoke check on Python 3.10, 3.11, and 3.12. Locally, bytecode compilation and the workflow's small import/shape smoke check also passed. The workflow configuration was inspected, but remote GitHub Actions results were not verified.

## Limitations

- There is no training loop, target-matrix fitting routine, pretrained model, or demonstrated weight-compression/reconstruction workflow in this repository.
- The API returns dense tensors; the generator does not avoid the memory cost of the requested matrix.
- No benchmark or quality/accuracy results are included, so speed, memory savings, and usefulness for any particular model are not established here.
- The code exposes `generate_layer_weights` for multiple matrices, but it does not define a full neural-network assembly or inference API.
- The wheel build configuration currently omits the `dna_lite.py` module from the built wheel; the source-checkout workflow above is the verified way to use it.

## License

The `LICENSE` file contains the MIT License. See [LICENSE](LICENSE) for its terms.
