# CMPE 258 — Nonlinear Regression with 3-Layer Neural Nets (Numpy / PyTorch / TF)

This repo contains **five+ Colab notebooks** implementing the same **3-layer MLP** (input → hidden1 → hidden2 → output) for **nonlinear regression** across different stacks, following the homework instructions.

## What you must add (mandatory)
1. **Video walkthroughs** for *every* notebook (Colab run + code tour + output).  
2. Embed/link those videos **inside this README** (most important requirement).  
3. Ensure the repo is **public** and the Colab links work.

> Tip: easiest workflow is to record the video in one take per notebook:  
> open notebook → run all → explain main sections → show plots/output → done.

---

## Notebooks

### A) Numpy “from scratch” + **manual backprop** + **tf.einsum**
- File: `notebooks/A_numpy_manual_3layer_einsum.ipynb`
- Highlights:
  - Synthetic data from a **3-variable nonlinear equation**
  - **Manual chain-rule backprop** (no autograd)
  - Uses **`tf.einsum`** for matrix products (per requirement)
  - Loss curve + parity plot + **4D visualization via PCA→3D + color**

Open in Colab: https://colab.research.google.com/drive/1TLH5yyBgQzGPGxHdEcyQBpmrLjHXR19H?usp=sharing

### B) PyTorch “from scratch” (no `nn.Linear`)
- File: `notebooks/B_torch_scratch_3layer_no_layers.ipynb`
- Highlights:
  - Uses raw tensors for weights/biases
  - Forward uses `torch.einsum`
  - Backprop via autograd (but **no built-in layers**)

Open in Colab: https://colab.research.google.com/drive/1ya4saipOxs4Qp2soZgMQhe5rskE5NJfk?usp=sharing

### C) PyTorch class-based with `nn.Module`
- File: `notebooks/C_torch_module_3layer_builtin.ipynb`
- Highlights:
  - Proper `nn.Module` model
  - Standard optimizer + training loop

Open in Colab: https://colab.research.google.com/drive/1J_qYoB2mI9criFiKhjPNNK1dqm9dyvH6?usp=sharing

### D) PyTorch Lightning
- File: `notebooks/D_torch_lightning_3layer.ipynb`
- Highlights:
  - LightningModule + Trainer
  - Cleaner training logs

Open in Colab: https://colab.research.google.com/github/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME/blob/main/notebooks/D_torch_lightning_3layer.ipynb

### E) TensorFlow variants (4 notebooks)
- `notebooks/E1_tf_scratch_lowlevel.ipynb` — low-level variables + `GradientTape` + `tf.einsum`
- `notebooks/E2_tf_builtin_layers.ipynb` — `tf.keras.Sequential` + Dense layers
- `notebooks/E3_tf_functional_api.ipynb` — Functional API model
- `notebooks/E4_tf_highlevel_subclassing.ipynb` — subclassed `tf.keras.Model`

Open in Colab:
- https://colab.research.google.com/github/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME/blob/main/notebooks/E1_tf_scratch_lowlevel.ipynb  
- https://colab.research.google.com/github/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME/blob/main/notebooks/E2_tf_builtin_layers.ipynb  
- https://colab.research.google.com/github/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME/blob/main/notebooks/E3_tf_functional_api.ipynb  
- https://colab.research.google.com/github/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME/blob/main/notebooks/E4_tf_highlevel_subclassing.ipynb  

---

## Required videos (embed/link them here)

Create one video per notebook. Put files in `videos/` **or** upload to YouTube/Drive and paste links.

### Video A — Numpy + tf.einsum + manual backprop
- Link: **(ADD LINK HERE)**

### Video B — PyTorch scratch (no layers)
- Link: **(ADD LINK HERE)**

### Video C — PyTorch `nn.Module`
- Link: **(ADD LINK HERE)**

### Video D — PyTorch Lightning
- Link: **(ADD LINK HERE)**

### Video E1 — TF scratch (low-level)
- Link: **(ADD LINK HERE)**

### Video E2 — TF built-in layers
- Link: **(ADD LINK HERE)**

### Video E3 — TF Functional API
- Link: **(ADD LINK HERE)**

### Video E4 — TF high-level subclassing
- Link: **(ADD LINK HERE)**

---

## How to run
1. Open any notebook in Colab.
2. Runtime → **Run all**.
3. Confirm you see:
   - loss curve
   - predicted vs actual plot
   - PCA→3D “4D plot” with color = target
4. Record video walkthrough and add it above.

---

## File tree
