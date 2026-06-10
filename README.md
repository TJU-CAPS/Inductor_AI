# Inductor AI — EC Core Loss Prediction with Physics-Informed Neural Networks

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20587725.svg)](https://doi.org/10.5281/zenodo.20587725)

A hybrid approach combining **analytic modeling** and **artificial neural networks** for predicting EC-type inductor performance (inductance, winding loss, core loss). Datasets are generated via **Maxwell FEM simulations** automated with PyAEDT.

## Project Structure

```
├── 1_Data_Generator/               # FEM simulation & data generation
│   ├── Inputs_generator.py         # GPU-accelerated parameter sampler (CuPy)
│   ├── inductor_1.aedt             # Maxwell 3D project file
│   └── output_samples/
│       ├── filtered_samples_20000.csv   # 20,000 training samples
│       └── Test_5000.csv                # 5,000 test samples
│
├── 2_Train/                        # Model training & evaluation
│   ├── analytic_model.py           # Physics-based analytic inductor model
│   ├── Train.py                    # PyTorch training (analytic + ANN hybrid)
│   └── Test.py                     # Model evaluation & prediction
│
├── .gitignore
└── README.md
```

## Workflow

### 1. Data Generation

`1_Data_Generator/Inputs_generator.py` samples the 9-dimensional design space using GPU-accelerated filtering (CuPy). Parameters include core geometry (`C`, `dc1`, `dc2`, `ht`), air gap (`lg1`), frequency (`f`), current (`i`), and winding turns (`Nx`, `Ny`). The sampled inputs are then simulated in **Ansys Maxwell** using the `inductor_1.aedt` project.

### 2. Analytic Model

`2_Train/analytic_model.py` provides a **first-principles physics model** that computes:
- **Inductance L** — via magnetic reluctance network
- **Winding loss Pw** — DC + AC (skin/proximity effect) + air-gap fringing losses
- **Core loss Pc** — Steinmetz-based volumetric loss calculation

This analytic model serves as a baseline and is used alongside the neural network.

### 3. Neural Network Training

`2_Train/Train.py` trains a **multi-branch feedforward neural network** (PyTorch):

- **Shared feature extractor**: 9 → 79 FC + ReLU
- **Branch heads**: Separate prediction paths for L, Pw, Pc
- **Loss**: Weighted MSE
- **Preprocessing**: Log transformation + StandardScaler

### 4. Evaluation

`2_Train/Test.py` loads the trained checkpoint and evaluates on the test set, reporting:
- Relative error (mean, max, 95th percentile)
- R² score per output
- Scatter plots and error histograms

## Input / Output Features

### Inputs (9 parameters)

| Feature | Description | Unit |
|---------|-------------|------|
| `C` | Center leg width | mm |
| `dc1` | Window depth | mm |
| `dc2` | Outer leg width | mm |
| `f` | Frequency | kHz |
| `ht` | Window height | mm |
| `i` | Excitation current | A |
| `lg1` | Air gap length | mm |
| `Nx` | Turns (x-direction) | — |
| `Ny` | Turns (y-direction) | — |

### Outputs (3 targets)

| Target | Description | Unit |
|--------|-------------|------|
| `L` | Inductance | μH |
| `Pw` | Winding loss | W |
| `Pc` | Core loss | W |

## Requirements

- Python 3.10+
- PyTorch
- NumPy, SciPy, Pandas, scikit-learn
- Matplotlib
- CuPy (optional, for GPU-accelerated data generation)
- Ansys Maxwell + PyAEDT (for FEM simulation)

## Citation

If you use this code or data in your research, please cite:

```bibtex
@misc{TJU-CAPS-inductor-ai,
  author       = {TJU-CAPS},
  title        = {Inductor AI: EC Core Loss Prediction with Physics-Informed Neural Networks},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20587725},
  url          = {https://doi.org/10.5281/zenodo.20587725}
}
```

## Funding

This work is supported by **TECH SEED**.

## License

- **Code** — [MIT License](LICENSE)
- **Datasets** (`output_samples/*.csv`) — [CC BY 4.0](LICENSE-DATA)

Attribution is required when using the datasets. The Maxwell project file (`.aedt`) is provided for reproducibility.
