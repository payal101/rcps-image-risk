# Control missed-pixel risk. See what the guarantee means.

A reconstruction model returns one image for each input. We wrap each predicted pixel in an interval and want to control how many target pixels those intervals miss, on average across new images.

This tutorial follows the practical, step-by-step style of the linked UQ tutorial. The statistical procedure here is Bates et al.'s risk-controlling prediction sets (RCPS), which is related but not identical to that tutorial's validation-based coverage bound. RCPS chooses a set size using calibration data and bounds its expected loss.

## 1. Define the target before looking at calibration outcomes

The example uses LoDoPaB-derived CT images as targets. For each target, it simulates a sparse-view, noisy CT measurement and reconstructs a filtered back-projection (FBP) image. A small CNN learns to map the FBP input `X` to a target image `Y`.

For an image with `P = M*N` pixels, define a symmetric interval around the model prediction `f(X)`:

```text
T_lambda(X)_p = [max(0, f(X)_p - lambda), min(1, f(X)_p + lambda)].
```

Here `lambda` is the interval radius. A larger radius creates a wider set and cannot increase the number of missed pixels.

The loss for one image pair is the fraction of its pixels outside their intervals:

```text
L_i(lambda) = (1/P) sum_p 1{Y_i,p is outside T_lambda(X_i),p}.
```

This is one number in `[0, 1]` per image. Pixels within an image can be dependent; we do not treat them as separate calibration observations.

## 2. Fit the reconstruction rule, then freeze it

The notebook uses three disjoint image groups: 700 training images, 300 calibration images, and 100 test images. It trains the CNN using only the training group. Once training ends, the network and preprocessing are kept fixed. The calibration images choose the interval radius; the test images are saved for the final diagnostic only.

The model does not need to be a correct statistical model for the RCPS risk statement. Its quality affects how wide the intervals need to be to reach the chosen risk target.

## 3. Turn each calibration image into a bounded loss

Choose the radius grid in advance. For each calibration image and each radius, count the missed pixels and divide by the number of pixels. The package helper does this:

```python
import numpy as np
from rcps import interval_loss_grid

radii = np.linspace(0.0, 1.0, 201)
losses = interval_loss_grid(
    Y_cal[:, None, :, :],  # targets: (images, channels, height, width)
    pred_cal.numpy(),      # frozen-model predictions, same shape
    radii,
    value_range=(0.0, 1.0),
)
print(losses.shape)  # (number of calibration images, number of radii)
```

Each row belongs to one image, not one pixel. Every column uses the same calibration images at a different, pre-fixed interval radius.

## 4. Bound the expected miss fraction

For one fixed radius, let `L_1,...,L_n` be the calibration image losses. The Hoeffding upper confidence bound is

```text
U_H = min(1, mean(L_i) + sqrt(log(1/delta) / (2*n))).
```

The notebook also includes the WSR bound from Bates et al. WSR uses the bounded losses and their observed variability to form an upper bound; it can be less conservative, though it is not guaranteed to be smaller on every dataset.

For a target `alpha`, RCPS chooses the narrowest radius on the grid whose bound and the bounds for every larger radius are below `alpha`:

```python
from rcps import calibrate_rcps

result = calibrate_rcps(
    losses,
    radii,
    alpha=0.10,       # allowed expected fraction of missed pixels
    delta=0.10,       # confidence failure probability
    method="wsr",    # also try "hoeffding"
)

if result.certified:
    print("Chosen radius:", result.selected_parameter)
    print("Risk upper bound:", result.selected_ucb)
else:
    print("No radius on this grid was certified for this target.")
```

For this example, the intended finite-sample statement is

```text
P_calibration( R(lambda_hat) > alpha ) <= delta,
R(lambda) = E_(X,Y)[ L(T_lambda(X), Y) ].
```

In words: with probability at least `1-delta` over the calibration sample, the chosen interval rule's expected fraction of missed pixels on a fresh image pair is at most `alpha`. With `delta=0.10`, the confidence level is 90%. With `alpha=0.10`, the target is at most 10% missed pixels on average per new image. It does **not** mean that 90% of images have every pixel covered.

The nested intervals are important: larger radii give nonincreasing image losses. The calibration procedure checks all larger radii when choosing the smallest passing one. It does not spend half of calibration data on a separate radius search and half on a risk check. The model, loss, preprocessing, and radius grid must be fixed before inspecting calibration labels.

### Why small samples can give “no certificate”

The notebook has 300 calibration images and uses `delta=0.10`. Its Hoeffding additive margin is about `0.062`, even if the observed loss is zero. Thus Hoeffding cannot certify `alpha=0.01` or `0.05` with this sample size. That means the bound is too conservative to certify those targets; it does not prove the model's true risk is above them. WSR may give a tighter bound depending on the observed losses, but it still needs enough representative calibration images.

The notebook repeats calibration for `alpha = 0.01, 0.05, 0.10, 0.20`. Each row is a separate `1-delta` claim. It does not provide one simultaneous confidence statement for all four targets together.

## 5. Evaluate once on untouched test images

After calibration chooses a radius, apply that radius to the test predictions. The notebook reports:

- **Test pixel coverage:** fraction of all test pixels inside the intervals. With equal-size images, one minus this value is the test mean fractional-pixel miss loss.
- **Test whole-image coverage:** fraction of test images with every pixel inside. This is a separate diagnostic and is not controlled by this RCPS loss.
- **Mean interval width:** average interval width across pixels. Smaller is more concise, but width alone is not a guarantee.

The test results estimate how the method behaved on this particular untouched test sample. They neither choose the radius nor create the risk certificate. With only 100 test images, these empirical summaries can vary noticeably between samples.

## 6. Look at an example, not just the table

The final notebook cell shows an FBP input, target image, reconstruction, interval width map, and pixels missed by the interval for one test case. This helps identify where the model struggles. The risk guarantee is still an average over image pairs: some images or regions can have more misses than others.

## Assumptions and scope

The Hoeffding and WSR mean-risk bounds assume calibration image pairs are independent and identically distributed with future image pairs, conditional on the already-fitted model. Independence is required between the image-level observations; pixels inside one image may be dependent because they are aggregated into a single loss. If images include correlated slices from the same patient, a random image split may not provide patient-level independence. The small dataset mirror does not include documented patient grouping, so this experiment cannot establish patient-level guarantees.

Targets are LoDoPaB-derived, but inputs are simulated sparse-view noisy measurements and FBP reconstructions. This is a research stress test, not an evaluation on official LoDoPaB measurements or a clinical validation.

## How this differs from the linked tutorial

The linked tutorial freezes a prediction-interval rule, measures binary coverage on a held-out validation set, and uses a Hoeffding **lower** bound on coverage. This notebook instead searches a fixed nested family of intervals and uses an upper bound on the bounded **miss risk** to choose a radius. These are connected finite-sample UQ ideas, but their statistics and guarantees are not interchangeable.

For the equations and implementation details, see [`docs/method.md`](docs/method.md). The complete runnable experiment is [`notebooks/lodopab_rcps.ipynb`](notebooks/lodopab_rcps.ipynb).
