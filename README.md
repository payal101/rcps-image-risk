# RCPS for image regression

A small, documented implementation of **risk-controlling prediction sets (RCPS)** for image-to-image regression. The notebook uses a small LoDoPaB-derived CT image subset and builds pixelwise intervals around a reconstruction model's prediction. Calibration controls the *expected fraction of pixels missed per image*.

The statistical unit is one image pair `(X_i, Y_i)`. Pixels within an image may be dependent; each image is reduced to one bounded loss in `[0, 1]`. This project does not claim simultaneous coverage of every pixel or whole-image coverage.

## What is in this repository

- `notebooks/lodopab_rcps.ipynb` — end-to-end Colab-friendly data, training, calibration, test diagnostics, and visualization.
- `src/rcps/` — reusable bounded-risk confidence bounds, monotone-family RCPS calibration, and image interval losses.
- `docs/method.md` — equations, assumptions, and interpretation of the output.

## Quick start

In Google Colab, open `notebooks/lodopab_rcps.ipynb` and run the cells in order. It downloads a small subset and simulates sparse-view noisy CT measurements, so no large LoDoPaB reconstruction download is needed. A GPU is optional.

To use the Python package locally:

```bash
pip install -e .
```

The core package needs NumPy only. Notebook dependencies are optional:

```bash
pip install -e ".[demo]"
```

## Statistical target

For image `i`, with `P` pixels and interval `T_lambda(X_i)`, define

```text
L_i(lambda) = (1/P) sum_p 1{Y_i,p is outside T_lambda(X_i),p}.
R(lambda)   = E[L_i(lambda)] for a fresh image pair.
```

The goal for a chosen risk target `alpha` and confidence error `delta` is

```text
P_calibration(R(lambda_hat) > alpha) <= delta.
```

The inner expectation is over a new image pair. The outer probability is over the random calibration data. The notebook reports test-set empirical pixel coverage as a diagnostic; test results do not choose the interval or create the certificate. See [the method notes](docs/method.md) before interpreting a result.

## Methods

The calibration supports three upper confidence bounds for `[0,1]` image losses:

- `hoeffding`: simple and distribution-free, often conservative at small `n`.
- `wsr`: the wealth-sequence risk bound from Bates et al., intended to improve efficiency for bounded nonbinary losses.
- `empirical_bernstein`: variance-adaptive bound included as a comparison.

The notebook compares Hoeffding and WSR at `alpha = 0.01, 0.05, 0.10, 0.20`, with `delta = 0.10`. This is a separate RCPS risk-control workflow; `alpha` is the maximum allowed *mean miss fraction*, not split-conformal miscoverage. Each alpha row has its own `1-delta` statement; the four rows are not jointly certified at once.

The Gaussian chi-square comparison is not applicable to this pixelwise bounded-loss RCPS objective. It belongs to a different construction that assumes a Gaussian residual vector and controls a Mahalanobis ellipsoid. Do not interpret it as an RCPS baseline for the same risk target.

## Data and limitations

The notebook uses LoDoPaB-derived target images from the `AmirEhsan1995/lodopab-ct-glimpse` Hugging Face dataset. Inputs are simulated from those targets with sparse-view Radon projections, added Gaussian noise, and filtered back projection. They are **not** the official LoDoPaB measurement inputs. The dataset mirror does not provide patient grouping metadata, so this experiment can support only an image-level IID/exchangeability working assumption, not a patient-level independence claim. It is a reproducible research stress test, not a clinical validation.

## Reference

Bates, S., Angelopoulos, A. N., Lei, L., Malik, J., & Jordan, M. I. (2021). *Distribution-Free, Risk-Controlling Prediction Sets*. Journal of the ACM, 68(6). [DOI](https://doi.org/10.1145/3478535) · [Paper](https://arxiv.org/abs/2101.02703) · [Authors' code](https://github.com/aangelopoulos/rcps).

This is an independent educational implementation, not the authors' official repository.
