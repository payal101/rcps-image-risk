# RCPS for image regression: a mathematical walkthrough

This tutorial explains the image-level risk-controlling prediction set (RCPS) experiment in this repository. It defines the prediction intervals, the loss, the calibration rule, and the guarantee, then explains what the test metrics mean.

The linked UQ tutorial is a useful reference for the style of explanation, but it uses a different procedure: it evaluates a frozen prediction rule and forms a lower confidence bound on coverage. Here we select an interval size using RCPS and form an upper confidence bound on expected miss risk.

## 1. Data and prediction model

Each example is an image pair

\[
(X_i,Y_i) \sim \mathcal{D}, \qquad i=1,\ldots,n,
\]

where \(X_i\) is an input reconstruction and \(Y_i\) is the target image. The notebook creates the input by simulating a sparse-view noisy CT measurement from a LoDoPaB-derived target and applying filtered back projection (FBP). A small convolutional neural network is trained to predict the target from the FBP input.

Write the trained predictor as

\[
f_{\theta}(X) \in [0,1]^{M\times N},
\]

where \(M\times N\) is the image size and \(\theta\) denotes the trained network parameters. The network is trained on the training split and then **frozen**: its parameters and the image preprocessing are not changed using calibration or test labels.

The notebook uses 700 training images, 300 calibration images, and 100 test images. The first split fits \(f_{\theta}\); the second chooses the interval size; the last is used only for evaluation diagnostics.

## 2. Define the prediction set

For a nonnegative radius \(\lambda\), the prediction set at pixel \(p\) is the interval

\[
T_{\lambda}(X)_p
=\left[\max\{0,f_{\theta}(X)_p-\lambda\},\;
        \min\{1,f_{\theta}(X)_p+\lambda\}\right].
\]

Here \(p\) indexes one of the \(P=MN\) pixels. The interval is clipped to the target intensity range \([0,1]\). Increasing \(\lambda\) makes the intervals wider, so it cannot turn a covered pixel into a missed pixel.

The set \(T_{\lambda}(X)\) is a collection of intervals, one per pixel. It is **not** one interval for the entire image, and this construction does not by itself give a guarantee that every pixel is covered simultaneously.

## 3. Reduce each image to one bounded loss

For image pair \((X_i,Y_i)\), define the loss

\[
L_{\lambda}(X_i,Y_i)
=\frac{1}{P}\sum_{p=1}^{P}
  \mathbf{1}\!\left\{Y_{i,p}\notin T_{\lambda}(X_i)_p\right\}.
\]

The indicator \(\mathbf{1}\{\cdot\}\) equals one when its condition is true and zero otherwise. Thus \(L_{\lambda}\) is the fraction of pixels missed in image \(i\), and always lies in \([0,1]\).

Although an image contains many pixels, the statistical observation used by RCPS is **one image-level loss**. Pixels inside an image may be dependent; the method does not assume pixel independence. The sampling assumption is about different image pairs.

For a fixed radius \(\lambda\), the population risk is

\[
R(\lambda)
=\mathbb{E}_{(X,Y)\sim\mathcal{D}}
  \left[L_{\lambda}(X,Y)\right].
\]

This expectation is over a fresh image pair from the target population. It is the long-run average fraction of missed pixels per image.

## 4. State the risk-control goal

Choose two numbers:

\[
\alpha\in(0,1) \quad\text{and}\quad \delta\in(0,1).
\]

- \(\alpha\) is the largest acceptable **expected fraction of pixels missed per image**.
- \(\delta\) is the probability that the calibration procedure fails to certify the target.

For the radius \(\widehat{\lambda}\) selected from calibration data, the RCPS objective is

\[
\Pr_{S_{\mathrm{cal}}}\!\left\{
  R(\widehat{\lambda})>\alpha
\right\}\leq\delta.
\]

The probability on the left is over the random calibration sample \(S_{\mathrm{cal}}\), with the trained network and pre-specified procedure held fixed. Equivalently, with probability at least \(1-\delta\) over calibration samples, the selected rule has expected miss risk at most \(\alpha\) on a new image pair.

For example, \(\alpha=0.10\) and \(\delta=0.10\) aim to certify that the **mean fraction of missed pixels** is at most 10%, with 90% confidence. They do not say that 90% of images have all pixels covered, or that every image misses at most 10% of its pixels.

## 5. Estimate the risk on calibration images

Let the calibration set contain \(n_{\mathrm{cal}}\) independent image pairs. At each pre-specified radius \(\lambda_k\), compute one loss per calibration image:

\[
L_{i,k}=L_{\lambda_k}(X_i,Y_i), \qquad
\widehat{R}_{\mathrm{cal}}(\lambda_k)
=\frac{1}{n_{\mathrm{cal}}}\sum_{i=1}^{n_{\mathrm{cal}}}L_{i,k}.
\]

The sample mean \(\widehat{R}_{\mathrm{cal}}\) is an estimate of population risk, but an estimate alone is not a risk certificate. RCPS also computes an upper confidence bound \(U_k\) satisfying, for each fixed \(\lambda_k\),

\[
\Pr\!\left\{R(\lambda_k)\leq U_k\right\}\geq 1-\delta.
\]

### Hoeffding upper bound

Because each image loss is in \([0,1]\), the notebook can use

\[
U_{k}^{\mathrm{H}}
=\min\!\left\{
1,\;
\widehat{R}_{\mathrm{cal}}(\lambda_k)
+\sqrt{\frac{\log(1/\delta)}{2n_{\mathrm{cal}}}}
\right\}.
\]

The square-root term is the finite-sample uncertainty allowance. It shrinks as the number of independent calibration images increases. The notebook also includes the WSR bound from Bates et al., which uses a wealth-sequence construction for bounded losses and can be more informative in some cases. Neither method can compensate for calibration images that are unrepresentative of future images.

## 6. Select a radius from the nested family

The radius grid is set before examining calibration labels:

\[
\lambda_1<\lambda_2<\cdots<\lambda_K.
\]

Because larger radii create larger intervals, the observed losses are nested and nonincreasing:

\[
L_{i,1}\geq L_{i,2}\geq\cdots\geq L_{i,K}
\quad\text{for every calibration image }i.
\]

Using one calibration set, RCPS selects the smallest grid radius for which the upper bounds at that radius and all larger radii are below the target:

\[
\widehat{k}
=\min\left\{
k:\;\max_{j\geq k} U_j < \alpha
\right\},
\qquad
\widehat{\lambda}=\lambda_{\widehat{k}}.
\]

The maximum over \(j\geq k\) means every larger set must pass too. This suffix check uses the nesting of the prediction sets to support selection over the radius grid without splitting calibration into separate search and risk-check subsets or applying a union bound over every radius. If the set in braces is empty, the algorithm reports **no certificate** for that \(\alpha\); it must not claim success.

The reusable implementation is:

```python
import numpy as np
from rcps import interval_loss_grid, calibrate_rcps

radii = np.linspace(0.0, 1.0, 201)  # choose before inspecting calibration labels
losses = interval_loss_grid(
    Y_cal[:, None, :, :],
    pred_cal.numpy(),
    radii,
    value_range=(0.0, 1.0),
)

result = calibrate_rcps(
    losses,
    radii,
    alpha=0.10,
    delta=0.10,
    method="wsr",  # alternatively: "hoeffding"
)

if result.certified:
    print("Selected lambda:", result.selected_parameter)
    print("Selected risk upper bound:", result.selected_ucb)
else:
    print("No radius on this grid was certified for alpha=0.10.")
```

The array `losses` has shape \((n_{\mathrm{cal}},K)\): each row is an image and each column a radius. The selected \(\lambda\) is an interval half-width in normalized image intensity units; it is not the split-conformal quantile \(q\) used in a different procedure.

## 7. Why a small calibration set may not certify a strict target

In the notebook, \(n_{\mathrm{cal}}=300\) and \(\delta=0.10\). The Hoeffding allowance is

\[
\sqrt{\frac{\log(1/0.10)}{2\cdot 300}}
\approx 0.062.
\]

Even if the observed calibration miss risk were zero, this bound could not fall below \(\alpha=0.01\) or \(\alpha=0.05\). This says the Hoeffding bound is too wide to certify those targets with this sample size; it does **not** show that the true risk is larger than those targets. WSR may give a different bound from the same losses, but certification still depends on the observed data and assumptions.

The notebook repeats the procedure for \(\alpha\in\{0.01,0.05,0.10,0.20\}\), using \(\delta=0.10\). Each row is an individual 90%-confidence risk statement. The four rows are not one simultaneous 90%-confidence statement.

## 8. Evaluate on the untouched test images

After calibration selects \(\widehat{\lambda}\), the notebook applies it to test predictions without changing the model or radius. It reports:

### Test pixel coverage

\[
\widehat{C}_{\mathrm{pixel}}
=\frac{1}{n_{\mathrm{test}}P}
  \sum_{i=1}^{n_{\mathrm{test}}}\sum_{p=1}^{P}
  \mathbf{1}\!\left\{
  Y_{i,p}\in T_{\widehat{\lambda}}(X_i)_p
  \right\}.
\]

This is the fraction of all test pixels inside their intervals. For equal-size images,
\(1-\widehat{C}_{\mathrm{pixel}}\) equals the average fractional-pixel miss loss across test images. It is a test-set diagnostic; the guarantee comes from calibration, not from this calculation.

### Test whole-image coverage

\[
\widehat{C}_{\mathrm{image}}
=\frac{1}{n_{\mathrm{test}}}\sum_{i=1}^{n_{\mathrm{test}}}
  \mathbf{1}\!\left\{
  Y_{i,p}\in T_{\widehat{\lambda}}(X_i)_p
  \text{ for every pixel }p
  \right\}.
\]

This is the fraction of test images with no missed pixels. It is a different and much stricter event. The RCPS loss defined above does not control this whole-image coverage quantity.

The notebook also reports mean interval width. Width describes how large the predicted intervals are; it is not itself a coverage or risk guarantee.

## 9. Visualize a test example

The final notebook cell displays the FBP input, target, reconstruction, interval-width map, and missed-pixel mask for one untouched test image. These pictures help diagnose where the reconstruction and intervals struggle. The guarantee remains an average across new image pairs: it does not promise equal risk at every pixel, region, or image.

## 10. Assumptions and limitations

The Hoeffding and WSR bounds require calibration image pairs to be independent and identically distributed with future image pairs, conditional on the already-trained predictor. Image pairs are the observations; pixels inside a single image may be dependent. If several images come from the same patient, image-level independence may fail unless the split accounts for patient groups. The small dataset mirror does not document patient IDs, so this experiment cannot claim patient-level generalization.

The targets are LoDoPaB-derived, but the inputs are simulated from sparse-view noisy measurements and reconstructed by FBP. The notebook therefore evaluates this constructed image-to-image task, not official LoDoPaB measurement inputs or a clinical system.

## Related reading

- [RCPS method notes](docs/method.md)
- [Runnable LoDoPaB notebook](notebooks/lodopab_rcps.ipynb)
- [Bates et al., *Distribution-Free, Risk-Controlling Prediction Sets*](https://doi.org/10.1145/3478535)
- [Tutorial that inspired this explanatory format](https://github.com/jatinbatra50/jatinbatra50/blob/provable-uq-tutorial/TUTORIAL.md)
