# 🧮 Deep Neural Network from Scratch — NumPy only

![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-only-013243?logo=numpy&logoColor=white)

An **L-layer fully-connected neural network written from first principles in pure NumPy** — no TensorFlow, no PyTorch.
Every piece of forward propagation, back-propagation and optimisation is implemented by hand to show a solid understanding of how deep learning works under the hood.

## ✨ What's implemented

| Component | Details |
|---|---|
| Architecture | Configurable depth — demo uses `[12288 → 20 → 7 → 5 → 1]` |
| Activations | ReLU (hidden), Sigmoid (output) + their derivatives |
| Initialisation | **He** (ReLU) and **Xavier** (tanh) |
| Forward / backward pass | Vectorised linear + activation steps with caches |
| Loss | Binary cross-entropy + **L2 regularisation** |
| Optimisation | **Mini-batch gradient descent** with shuffling |
| Monitoring | Training vs. validation cost curves |
| Analysis | Visualises mis-classified images |

## 📊 Task

Binary image classification (**cat vs. non-cat**, 64×64 RGB images flattened to 12,288 features).
The model fits the training set perfectly (100%) and reaches **72% test accuracy** on a deliberately tiny dataset (209 train / 50 test images) — the validation curves and L2 term are used to study over-fitting.

## 🚀 Run it

```bash
pip install numpy matplotlib h5py
jupyter notebook DNN.ipynb
```

## 📚 Concepts Practised

Vectorisation · chain rule & back-propagation · weight initialisation · regularisation · bias/variance trade-off

---
👤 **Muath Omarieh** — CS student @ University of Jordan · [GitHub](https://github.com/muathomarieh) · [LinkedIn](https://www.linkedin.com/in/muath-omarieh-550679297/)
