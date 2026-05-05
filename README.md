# MFVI: Model-Agnostic Mean-Field Variational Inference for PyTorch

Minimal, non-intrusive wrapper that converts **any PyTorch `nn.Module`** into a Bayesian neural network using **mean-field variational inference (MFVI)**.

Designed to work out-of-the-box with **arbitrary architectures**, including large-output models such as neural operators (e.g., FNO), where many Bayesian methods become impractical.

---

## Key idea

Transforms deterministic parameters ( \theta ) into variational parameters:

* ( \theta \rightarrow (\mu, \sigma) )
* Sampling via reparameterization:
  [
  \theta = \mu + \sigma \odot \epsilon,\quad \epsilon \sim \mathcal{N}(0, I)
  ]

Optimizes the standard ELBO:

* likelihood (NLL)
* * KL divergence to prior

No explicit Jacobians or curvature approximations required → scales naturally to large output dimensions.

---

## Features

* **Model-agnostic**: works with any `nn.Module`
* **Non-intrusive**: no layer rewriting required
* **Scales to large outputs** (e.g., PDE grids, neural operators)
* **Supports regression and classification**
* **Posterior sampling utilities included**
* **Built-in KL aggregation and dataset-size normalization**

---

## Minimal usage

```python
model = MyModel()

# convert to Bayesian model
bnn = model_agnostic_dnn_to_bnn(
    model,
    train_dataset_size=len(dataset)  # or pass DataLoader / Dataset
)

# training loop
for x, y in dataloader:
    y_pred = bnn(x)
    nll = nll_regression(y_pred, y)  # or nll_classification
    kl = bnn.get_kl_loss()
    loss = nll + kl
    loss.backward()
```

---

## Posterior sampling

```python
# wrap model for sampling
bnn = PredSamplingWrapper.wrap_VI_model(bnn)

# sample predictions
with PredSamplingWrapper.enable_sampling(n_samples=50, moments=True):
    mu, sigma = bnn(x)
```

Supports:

* predictive moments (mean, variance)
* full sample distributions
* mixture distributions via `torch.distributions`

---

## Notes

* KL is automatically normalized by dataset size if provided
* Uses softplus parameterization for stability
* Supports MOPED-style priors
* Handles complex tensors via real-view conversion
* Designed for stability in large models (e.g., avoids explicit curvature)

---

## Citation

If you use this code, please cite:

```bibtex
@misc{mfvi_model_agnostic_2026,
  title={MFVI: Model-Agnostic Mean-Field Variational Inference for PyTorch},
  author={Your Name},
  year={2026},
  note={Code release. Paper forthcoming.}
}
```

---

## Status

Research code. Actively used in internal and collaborative projects.
A formal paper describing the method and scaling properties is in preparation.
