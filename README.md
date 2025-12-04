# MLP Depth Analysis on MNIST

This repository contains the code, tutorial, and results for a machine learning experiment studying how the depth of a Multilayer Perceptron (MLP) affects performance on the MNIST handwritten digits dataset.

The project forms part of the Machine Learning and Neural Networks course assignment.

---

## 📁 Repository Structure

<img width="276" height="478" alt="Screenshot 2025-12-04 at 10 27 42 PM" src="https://github.com/user-attachments/assets/24557a4d-755b-4414-b067-fc7d19266617" />

---

## 📌 Overview

The experiment compares three MLP architectures:

- **MLP_1_hidden** → 128 ReLU → Softmax  
- **MLP_2_hidden** → 128 ReLU → 64 ReLU → Softmax  
- **MLP_3_hidden** → 256 ReLU → 128 ReLU → 64 ReLU → Softmax  

Each model was trained for 20 epochs (batch size 128) using the Adam optimiser and categorical cross-entropy loss.

---

## 🔍 Key Findings

- The **1-hidden-layer MLP** achieved the highest accuracy (**0.911**).
- The **2-hidden-layer MLP** achieved moderate accuracy (**0.744**).
- The **3-hidden-layer MLP** failed to converge (**0.305** accuracy).
- Training time increased significantly with depth.
- Validation loss for the 3-layer MLP exploded, indicating unstable training.
- Confusion matrices confirmed that deeper models made many more errors.

Full visualisations and explanations are available in the tutorial PDF and notebook.

---

## ▶️ How to Run the Notebook

1. Install dependencies:
```bash
pip install tensorflow matplotlib numpy scikit-learn
