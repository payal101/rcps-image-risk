# Method and guarantees

## Prediction set and image-level loss

The fitted image model is fixed before calibration. For an input image `X`, it returns a point reconstruction `f(X)`. For radius `lambda >= 0`, define the pixelwise interval

```text
T_lambda(X)_p = [max(0, f(X)_p - lambda), min(1, f(X)_p + lambda)].
```

For target image `Y` with `P = M*N` pixels, one calibration observation is the **fractional pixel miss loss**

```text
L(T_lambda(X), Y)
  = (1/P) sum_(p=1)^P 1{Y_p not in T_lambda(X)_p}.
```

It is a scalar in `[0,1]` for each image pair. The population risk is

```text
R(T_lambda) = E_(X,Y)~P [ L(T_lambda(X), Y) ].
```

This is the expected fraction of pixels missed on a new image drawn from the same population. It does not promise that all pixels in a given image are covered, nor does it give a separate guarantee for each pixel. Pixel dependence inside an image is allowed because pixels are aggregated into one image-level loss.

## RCPS statement

For target risk `alpha` and confidence error `delta`, the goal is

```text
P_calibration( R(T_lambda_hat) > alpha ) <= delta.
```

Equivalently, with probability at least `1-delta` over fresh calibration samples, the selected predictor's expected miss fraction is at most `alpha`. This is a high-probability certificate about expected risk, not a statement that `1-delta` of individual test images will be fully covered.

## Calibration procedure

1. Train the reconstruction model on the training split. Freeze it; do not fit or tune it on calibration or test data.
2. Before inspecting calibration outcomes, choose an increasing finite radius grid `lambda_1 < ... < lambda_K`. Larger radii make larger nested intervals and cannot increase any image's miss loss.
3. Compute each calibration image's loss at each radius, forming `L[i,k]`.
4. For every radius, compute a one-sided `(1-delta)` upper confidence bound `U_k` on the mean of its bounded image losses. The notebook includes Hoeffding and WSR bounds.
5. Select the smallest radius index `k` for which every larger radius also has `U_j < alpha`. If none passes, report **no certificate** for that risk target; do not silently claim success.
6. Evaluate the selected interval on untouched test images. Report the empirical mean miss fraction and pixel coverage as diagnostics only.

Requiring the suffix of larger nested sets to pass is the monotone-family RCPS rule. It allows the same calibration sample to estimate bounds over the radius family without a union bound over all radii. The model and parameter grid must be fixed independently of the calibration labels. If you use calibration results to change the network, grid, preprocessing, or loss, the stated certificate no longer follows from this workflow.

## Bounds used

For `n` image losses `L_1,...,L_n` with sample mean `Lbar`, Hoeffding gives

```text
U_H = min(1, Lbar + sqrt(log(1/delta)/(2n))).
```

This is transparent but may be too wide when `n` is small or `alpha` is stringent. The WSR bound in Bates et al. is obtained by inverting a wealth-sequence test for the bounded mean; it can adapt to observed loss variability. Neither bound removes the need for representative calibration images.

## Sampling assumptions and scope

The concentration bounds used here assume calibration image pairs are IID from the same distribution as future image pairs, conditional on the already-trained model and fixed procedure. The finite-sample guarantee is not justified by exchangeability alone for these Hoeffding/WSR mean-risk bounds. Dependence between pixels within one image is not the sampling assumption. Dependence between image examples (for example, multiple slices from one patient) can invalidate image-level IID sampling if the split leaks correlated groups. Patient IDs are not documented in the small mirror used by this example, so this notebook cannot certify patient-level generalization.

The notebook's targets are LoDoPaB-derived images, while inputs are simulated sparse-view noisy projections reconstructed by FBP. Conclusions are about this constructed image-pair task, not performance on official LoDoPaB measurements or clinical deployment.

## What the reported metrics mean

- **Calibration mean miss fraction:** empirical estimate of the controlled risk.
- **Risk UCB:** upper confidence bound; certification requires it to be below `alpha`.
- **Test pixel coverage:** fraction of all test pixels inside the intervals. Its complement equals the average fractional miss loss when images have equal size.
- **Test whole-image coverage:** fraction of images with no missed pixels. It is a different diagnostic and is not the controlled risk.
- **Mean interval width:** average pixel interval width, a measure of set size, not a guarantee.

The notebook compares bounds at four alpha values. Each is a separate risk-control claim at confidence `1-delta`; simultaneous confidence across all four targets is not claimed.
