# LECO: Learning from Correctness Without Prompting

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-COLM%202024-green.svg)](https://openreview.net/forum?id=dcbNzhVVQj#discussion)

🚀 **High-quality reproduction** of "Learning From Correctness Without Prompting Makes LLM Efficient Reasoner" (COLM 2024)

## ✨ Highlights

- 📊 **Complete Implementation**: Three-method comparison (Baseline, Original LECO, Improved LECO)
- 🎯 **Enhanced Error Detection**: Statistical threshold strategy outperforming original method
- 💻 **Cross-Platform**: Windows/Linux compatible with comprehensive memory management
- 📈 **Verified Results**: +1.94% overall improvement with detailed analysis
- 🔧 **Production Ready**: Modular design with robust error handling

## 🎯 Key Results

| Method | GSM8K | MATH | Overall | Token Efficiency |
|--------|-------|------|---------|------------------|
| Complex CoT | 92.67% | 38.57% | 61.11% | 1.0× |
| Original LECO | 88.67% | 36.19% | 58.06% | 2.5× |
| **Improved LECO** | **89.33%** | **39.05%** | **60.00%** | 2.5× |

*Results on 360 samples (150 GSM8K + 210 MATH) using DeepSeek-Math-7B-RL*

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (8GB+ VRAM recommended)
- 16GB+ RAM

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/LECO-Reproduction.git
cd LECO-Reproduction

# Install dependencies
pip install -r requirements.txt

# Download and setup model
python scripts/download_model.py

# Download datasets (GSM8K & MATH)
python scripts/setup_datasets.py
