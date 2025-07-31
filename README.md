# TabularGAN: GANs and cGANs for Realistic Tabular Data Synthesis

This project investigates the use of **Generative Adversarial Networks (GANs)** and **Conditional GANs (cGANs)** for generating high-quality synthetic tabular data using the UCI Adult dataset. We evaluate the fidelity and utility of the generated data using multiple seeds, detection metrics, and efficacy on downstream classification tasks.

---

## 🧠 Project Objective

- Generate realistic synthetic data that mirrors the distribution and predictive power of the original tabular dataset.
- Use both GAN and cGAN architectures for unconditional and class-conditioned generation.
- Tune hyperparameters via **Optuna**, ensuring optimized training performance.
- Evaluate both detection (AUC of real vs. fake classification) and efficacy (performance on downstream tasks like income prediction).

---

## 📊 Dataset: UCI Adult Income

A tabular dataset containing demographic and work-related information to predict whether an individual earns over $50K/year.

- Features: mix of **categorical** and **numerical** variables
- Label: binary income class (≤50K, >50K)

---

## ⚙️ Architecture

### 🔁 GAN

- **Generator**: Accepts latent noise vector, transforms through fully connected layers with ReLU, BatchNorm, Dropout → output scaled via sigmoid.
- **Discriminator**: Takes real or generated samples, passes through FC layers with LeakyReLU → output logit.

### 🔄 Conditional GAN (cGAN)

- **Conditioning**: One-hot label is concatenated to inputs of both Generator and Discriminator.
- **Same core architecture**, but conditioned on class label for controllable generation.

---

## 🧪 Training Setup

- Optimized using **Optuna** over:
  - Learning rates (G, D)
  - Noise dimension
  - Hidden size
  - Dropout rate
  - Batch size
- Fixed training budget of 50 epochs per model
- Weight initialization and optimizers follow DCGAN best practices
- Loss: `BCEWithLogitsLoss` for stable adversarial training

---

## 🧮 Preprocessing

- Label-encoded categorical features
- MinMax scaling for numerical features
- Stratified 80/20 train-test split
- Missing values handled and pruned

---

## 📈 Evaluation Metrics

### 1. **Detection**
- Train a Random Forest to distinguish real from synthetic samples
- Report AUC (ideal: 0.5 = indistinguishable)

### 2. **Efficacy**
- Train two classifiers on real vs. synthetic data
- Compare downstream AUC on income prediction
- Report AUC ratio: `AUC_synth / AUC_real`

---

## 🔍 Experimental Setup

- Seeds used: `42`, `2`, `3` for reproducibility
- Each model trained and evaluated independently per seed

---

## 📊 Results Summary

| Model | Seed | Detection AUC | Efficacy AUC (Real) | Efficacy AUC (Synth) | AUC Ratio |
|-------|------|----------------|---------------------|----------------------|-----------|
| GAN   | 42   | 1.00           | 0.9110              | 0.4976               | 0.55      |
| cGAN  | 42   | 1.00           | 0.9110              | 0.9876               | **1.08**  |
| GAN   | 2    | 0.9999         | 0.9081              | 0.4981               | 0.55      |
| cGAN  | 2    | 1.00           | 0.9081              | 0.6601               | 0.73      |
| GAN   | 3    | 0.99995        | 0.9068              | 0.4927               | 0.54      |
| cGAN  | 3    | 1.00           | 0.9068              | 0.5129               | 0.56      |

> 💡 Only cGAN with seed **42** achieves both high fidelity and downstream task performance.

---

## 💾 File Structure

```bash
├── seed2.ipynb                # Notebook for seed 2 experiments
├── report.docx                # Full project report and results
├── models/                    # GAN and cGAN architecture definitions
├── utils/                     # Preprocessing, training, and eval scripts
├── data/                      # Processed Adult dataset (optional)

## 💾 File Structure
-Python 3.8+
-PyTorch
-Optuna
-scikit-learn
-pandas
-numpy
-matplotlib
-seaborn

