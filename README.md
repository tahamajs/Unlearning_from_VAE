# Selective Amnesia: Machine Unlearning in Conditional VAEs

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository implements **Selective Amnesia**, a principled approach for machine unlearning in Conditional Variational Autoencoders (VAEs). The method enables targeted forgetting of specific classes while preserving performance on retained classes, based on the NeurIPS 2023 paper _"Machine Unlearning of Features and Labels"_.

## Table of Contents

- [Overview](#overview)
- [Mathematical Foundation](#mathematical-foundation)
- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
- [Results](#results)
- [Contributing](#contributing)
- [References](#references)
- [License](#license)

## Overview

### Problem Statement

Machine learning models trained on sensitive data may need to "forget" specific information due to privacy regulations (e.g., GDPR's right to be forgotten) or data poisoning concerns. Traditional retraining from scratch is computationally expensive. Selective Amnesia provides an efficient alternative for generative models.

### Solution Approach

We implement selective unlearning for Conditional VAEs by:

1. **Training** a baseline model on all classes
2. **Computing Fisher Information** to measure parameter importance for retained classes
3. **Optimizing a joint objective** that maximizes error on forget classes while minimizing error on retained classes
4. **Using Elastic Weight Consolidation (EWC)** to maintain stability

### Key Features

- ✅ **Targeted Forgetting**: Selectively unlearn specific classes (e.g., digit '2' from MNIST)
- ✅ **Performance Preservation**: Maintain generation quality for retained classes
- ✅ **Mathematical Rigor**: Based on principled ELBO optimization
- ✅ **Multiple Architectures**: Convolutional and Linear VAE implementations
- ✅ **Comprehensive Evaluation**: Latent space analysis and sample quality assessment

## Mathematical Foundation

### Conditional Generative Models

For conditional VAEs, the joint distribution is:
$$p(x, z|\theta, c) = p(x|\theta, c, z)p(z|\theta, c)$$

The Evidence Lower Bound (ELBO) becomes:
$$\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{z \sim q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) || p_\theta(z))$$

### Selective Amnesia Objective

The unlearning objective balances three terms:

$$\mathcal{L}_{\text{SA}} = \underbrace{\mathbb{E}_{q(x|c_f)}[\mathcal{L}_{\text{ELBO}}(x|\theta, c_f)]}_{\text{Forget Term}} + \underbrace{\mathbb{E}_{p(x|c_r)}[\mathcal{L}_{\text{ELBO}}(x|\theta, c_r)]}_{\text{Remember Term}} - \underbrace{\lambda \sum_i \frac{F_i}{2}(\theta_i - \theta_i^*)^2}_{\text{Stability Term}}$$

Where:

- $c_f$: Forget class
- $c_r$: Retained classes
- $F_i$: Fisher Information Matrix diagonal
- $\lambda$: Regularization strength

### Fisher Information Matrix

Parameter importance is measured as:
$$F_i = \mathbb{E}_{p_{\text{data}}(x, y)} \left[ \left( \frac{\partial}{\partial \theta_i} \mathcal{L}_{\text{ELBO}}(x, y; \theta) \right)^2 \right]$$

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended for faster training)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/selective-amnesia-vae.git
cd selective-amnesia-vae

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
matplotlib>=3.5.0
tqdm>=4.64.0
scikit-learn>=1.0.0
```

## Usage

### Quick Start

```python
from selective_amnesia import SelectiveAmnesiaVAE

# Initialize model
model = SelectiveAmnesiaVAE(z_dim=8, forget_class=2)

# Train baseline model
model.train_baseline(epochs=100)

# Perform selective unlearning
model.selective_amnesia(epochs=20, lambda_reg=100)

# Evaluate results
model.visualize_results()
```

### Detailed Workflow

1. **Baseline Training**:

   ```python
   # Train on full MNIST dataset
   vae = ConditionalVAE()
   train_vae(vae, train_loader, optimizer, epochs=500)
   ```

2. **Compute Fisher Information**:

   ```python
   # Generate replay dataset from retained classes
   replay_data = replay_dataset(vae, device, forget=2)

   # Calculate Fisher Information Matrix
   fisher_info = diag_fisher(vae, replay_data, device)
   ```

3. **Selective Amnesia**:

   ```python
   # Create surrogate dataset for forget class
   surrogate_data = surrogate_dataset(cls=2)

   # Perform unlearning
   selective_amnesia(vae, theta_star, fisher_info, replay_data, surrogate_data)
   ```

4. **Evaluation**:

   ```python
   # Visualize latent space
   plot_latent_space(vae, test_loader, device)

   # Generate samples
   show_grid(vae.sample(25, torch.eye(10)[[2]], device), "Forget Class Samples")
   ```

### Command Line Interface

```bash
# Run with default settings
python main.py

# Custom configuration
python main.py --epochs 100 --sa_epochs 30 --lam 50 --forget 2
```

### Configuration Options

- `--epochs`: Baseline training epochs (default: 50)
- `--sa_epochs`: Selective amnesia epochs (default: 10)
- `--batch`: Batch size (default: 256)
- `--lr`: Learning rate (default: 5e-4)
- `--lam`: EWC regularization strength (default: 100)
- `--seed`: Random seed (default: 49)
- `--quick`: Use fast demo settings

## Architecture

### Model Variants

#### Convolutional Conditional VAE

- **Encoder**: 4-layer CNN with batch normalization
- **Latent Dim**: 8-dimensional Gaussian
- **Decoder**: Symmetric transpose CNN
- **Conditioning**: Label channels concatenated to input

#### Linear Conditional VAE

- **Encoder**: 4-layer MLP with batch normalization
- **Latent Dim**: Configurable (default: 8)
- **Decoder**: Symmetric MLP architecture
- **Conditioning**: Labels concatenated to latent vectors

### Key Components

#### Data Handling

- `loaders()`: MNIST DataLoaders with one-hot encoding
- `replay_dataset()`: Generate retained class samples
- `surrogate_dataset()`: Create forget class noise data

#### Training Functions

- `train_vae()`: Standard VAE training loop
- `elbo()`: ELBO loss computation
- `selective_amnesia()`: Unlearning optimization

#### Evaluation Tools

- `plot_latent_space()`: 2D latent space visualization
- `show_grid()`: Image grid display
- `plot_latent_manifold()`: Manifold traversal

## Results

### Quantitative Results

| Model    | Forget Class (Digit 2) | Retained Classes | Latent Separation |
| -------- | ---------------------- | ---------------- | ----------------- |
| Baseline | High Quality           | High Quality     | Well-separated    |
| After SA | Degraded/Noise         | Preserved        | Maintained        |

### Qualitative Results

#### Sample Generation

- **Before Unlearning**: Clear, recognizable digits for all classes
- **After Unlearning**: Degraded samples for digit '2', preserved quality for others

#### Latent Space Analysis

- **Cluster Preservation**: Retained classes maintain distinct latent clusters
- **Forget Class Collapse**: Digit '2' distribution becomes noisy/overlapping

### Example Visualizations

```
Latent Space Before Unlearning (z_dim=8, PCA)
├── Digit 0: Tight cluster
├── Digit 1: Distinct separation
├── Digit 2: Well-defined region
└── ...

Latent Space After Unlearning (z_dim=8, PCA)
├── Digit 0: Preserved cluster
├── Digit 1: Maintained separation
├── Digit 2: Scattered/noisy distribution
└── ...
```

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to new functions
- Include unit tests for new features
- Update documentation for API changes

## References

### Core Paper

```
@article{golatkar2023machine,
  title={Machine Unlearning of Features and Labels},
  author={Golatkar, Aditya and Achille, Alessandro and Soatto, Stefano},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2023}
}
```

### Related Work

- **Elastic Weight Consolidation**: Kirkpatrick et al., PNAS 2017
- **Machine Unlearning**: Cao & Yang, ICML 2015
- **Variational Autoencoders**: Kingma & Welling, ICLR 2014

### Datasets

- **MNIST**: LeCun et al., MNIST handwritten digit database

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Implementation based on the Selective Amnesia method from NeurIPS 2023
- MNIST dataset courtesy of Yann LeCun and Corinna Cortes
- PyTorch framework for deep learning infrastructure

## Citation

If you use this code in your research, please cite:

```bibtex
@software{selective_amnesia_vae,
  title={Selective Amnesia: Machine Unlearning in Conditional VAEs},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/selective-amnesia-vae}
}
```
