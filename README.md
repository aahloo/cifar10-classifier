# CIFAR-10 Image Classifier

A high-performance deep learning image classifier achieving **90.62% test accuracy** on the CIFAR-10 dataset, significantly outperforming commercial benchmarks.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Project Overview

This project implements a VGG-style convolutional neural network with modern deep learning techniques to classify images from the CIFAR-10 dataset into 10 categories: airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks.

**Key Achievement:** 90.62% test accuracy vs. 70% vendor baseline (+20.62% improvement)

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 90.62% |
| **Validation Accuracy** | 90.38% |
| **Training Accuracy** | 92.80% |
| **Train-Val Gap** | 2.42% (minimal overfitting) |
| **Error Reduction vs Vendor** | 68.7% |

### Per-Class Performance

| Class | Accuracy |
|-------|----------|
| Ship | 96.20% |
| Truck | 94.50% |
| Horse | 92.70% |
| Car | 92.50% |
| Plane | 91.80% |
| Frog | 88.60% |
| Deer | 85.80% |
| Dog | 85.00% |
| Bird | 80.80% |
| Cat | 78.00% |

## 🏗️ Architecture

**VGG-Style CNN with Modern Enhancements:**
- 3 convolutional blocks (64→128→256 channels)
- Batch normalization after each conv layer
- Progressive dropout (0.1→0.2→0.3→0.4)
- MaxPooling for spatial reduction
- 3 fully connected layers (4096→512→256→10)
- **Total Parameters:** ~3.3 million

## 🚀 Features

- ✅ **Proper Train/Val/Test Split** (45k/5k/10k images)
- ✅ **Data Augmentation** (RandomHorizontalFlip, RandomCrop)
- ✅ **Batch Normalization** for training stability
- ✅ **Learning Rate Scheduling** (ReduceLROnPlateau)
- ✅ **Comprehensive Checkpointing** with full training history
- ✅ **GPU Support** with automatic device detection
- ✅ **Reproducible Results** (seed=42)

## 📁 Project Structure

```
cifar10-classifier/
├── CIFAR-10_Image_Classifier-STARTER.ipynb  # Main notebook
├── cifar10_model_proper_split.pth           # Trained model checkpoint
├── README.md                                 # This file
├── requirements.txt                          # Python dependencies
├── .gitignore                               # Git ignore rules
├── data/                                    # CIFAR-10 dataset (auto-downloaded)
└── docs/                                    # Additional documentation
    ├── Rubric_Evaluation.md
    ├── 90_Percent_Accuracy_Analysis.md
    └── Build_vs_Buy_Recommendation_TEXT.md
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, but recommended)

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/YOUR_USERNAME/cifar10-classifier.git
cd cifar10-classifier
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## 🎮 Usage

### Training from Scratch

Open the Jupyter notebook and run all cells:
```bash
jupyter notebook CIFAR-10_Image_Classifier-STARTER.ipynb
```

The notebook will:
1. Download CIFAR-10 dataset automatically (~170MB)
2. Set up data loaders with augmentation
3. Initialize the model
4. Train for 30 epochs (~1-2 hours on GPU)
5. Save the trained model checkpoint

### Using Pre-trained Model

Load the checkpoint to skip training:
```python
import torch

# Load checkpoint
checkpoint = torch.load('cifar10_model_proper_split.pth')

# Initialize model
model = CIFAR10Net()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Access metrics
print(f"Test Accuracy: {checkpoint['final_test_acc']*100:.2f}%")
```

## 📈 Training Details

### Hyperparameters
- **Optimizer:** Adam (lr=0.001, weight_decay=1e-4)
- **Loss Function:** CrossEntropyLoss
- **Batch Size:** 128
- **Epochs:** 30
- **Learning Rate Schedule:** ReduceLROnPlateau (factor=0.5, patience=3)

### Data Augmentation
- RandomHorizontalFlip (p=0.5)
- RandomCrop (size=32, padding=4)
- Normalization (mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])

### Training Time
- GPU (CUDA): ~1-2 hours
- CPU: ~10-12 hours

## 📊 Results Visualization

The notebook generates comprehensive plots:
1. **Validation Loss Curve** - Training stability tracking
2. **Train vs Validation Accuracy** - Overfitting monitoring
3. **Learning Rate Schedule** - Adaptive learning rate changes
4. **Overfitting Analysis** - Train-Val accuracy gap

## 🎯 Build vs Buy Analysis

**Recommendation: BUILD**

Our model achieves 90.62% accuracy compared to the vendor's 70%, representing:
- **+20.62 percentage points** improvement
- **68.7% error reduction** (2,062 fewer errors per 10,000 predictions)
- **$150,000 cost savings** over 5 years
- Complete technological control and in-house ML expertise

## 🔬 Technical Highlights

### Why 90.62% is Impressive:
- Matches **Maxout Networks (2013)** state-of-the-art performance
- Only 5% below **Wide ResNets (2016)**
- Far exceeds basic CNN benchmarks (65-75%)
- Achieved with relatively simple architecture

### Proper Methodology:
✅ No data contamination (separate test set)  
✅ Proper train/val/test split  
✅ Reproducible (fixed random seed)  
✅ Statistical validation (95% CI: 90.05%-91.19%)  
✅ Minimal overfitting (2.42% gap)  

## 🚀 Future Improvements

**Phase 2 Roadmap (Target: 93-95% accuracy):**
1. Implement ResNet50 architecture → +3-4%
2. Transfer learning with ImageNet weights → +1-2%
3. Advanced augmentation (Mixup, Cutout) → +1-2%
4. Ensemble methods → +2-3%

**Potential Final Accuracy: 95-97%**

## 📚 Dependencies

See `requirements.txt` for full list. Key dependencies:
- `torch>=2.0.0` - PyTorch deep learning framework
- `torchvision>=0.15.0` - Vision datasets and models
- `matplotlib>=3.5.0` - Plotting and visualization
- `numpy>=1.21.0` - Numerical computing

## 📄 Documentation

Additional documentation available in `docs/`:
- **Rubric_Evaluation.md** - Complete project requirements compliance
- **90_Percent_Accuracy_Analysis.md** - Detailed performance analysis
- **Build_vs_Buy_Recommendation_TEXT.md** - Full business recommendation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- CIFAR-10 dataset by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton
- PyTorch team for the excellent deep learning framework
- Udacity Machine Learning Engineering Nanodegree program

## 👤 Author

**Austin Sahl**
- DevOps Cloud Engineer & ML Enthusiast
- 20+ years experience in application support
- iOS Developer (SimpliFit - Kettlebell Training App)

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Project Status:** ✅ Production Ready  
**Last Updated:** January 2026  
**Version:** 1.0.0
