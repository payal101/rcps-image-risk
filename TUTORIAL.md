# RCPS for image regression: a readable walkthrough

This tutorial explains how the notebook uses risk-controlling prediction sets (RCPS) to control the average fraction of missed pixels in CT images. The equations are written in plain text so they remain readable even in viewers that do not render LaTeX.

## 1. What goes into the model?

Each example has two images:

- `X`: the input image, a sparse-view CT scan reconstructed using filtered back projection (FBP).
- `Y`: the target image, a LoDoPaB-derived CT image.

A small CNN learns a prediction rule called `f`. Given an input `X`, it produces a predicted image `f(X)`. The notebook trains `f` on 700 images, then freezes it. Calibration (300 images) chooses the interval size. The 100 test images are kept untouched until final evaluation.

## 2. How do we make an interval image?

For each pixel, the model's predicted intensity is the center. We add and subtract a chosen radius, `lambda`:

```text
lower pixel value = max(0, predicted pixel value - lambda)
upper pixel value = min(1, predicted pixel value + lambda)
```

Together, these lower and upper values form an interval at every pixel. Values are clipped to `[0, 1]`, the image intensity range. A larger `lambda` makes wider intervals.

In notation, for pixel `p`:

```text
T_lambda(X)[p] = [ lower value at p, upper value at p ]
```

`T_lambda(X)` means the whole collection of pixel intervals for input image `X`.

## 3. How do we score one image?

For one image, count how many target pixels fall outside their intervals, then divide by the total number of pixels:

```text
image loss = missed-pixel count / total-pixel count
```

For example, a loss of `0.03` means the intervals missed 3% of the pixels in that image.

Each image contributes **one loss number** between 0 and 1 to calibration. We do not pretend pixels inside an image are independent. Their misses are combined into one image-level number.

## 4. What risk are we trying to control?

Imagine repeatedly drawing a new image pair `(X, Y)` from the population of interest and calculating its image loss. The population risk is the average of those losses:

```text
R(lambda) = average image loss over new image pairs, using radius lambda
```

We choose:

- `alpha`: the largest acceptable average miss fraction per image.
- `delta`: the chance that calibration fails to certify the chosen radius.

The RCPS goal is:

```text
Probability over calibration data that R(chosen lambda) > alpha <= delta
```

In words: with confidence at least `1 - delta`, the chosen intervals have an expected miss fraction no greater than `alpha` on a new image pair.

Example: `alpha = 0.10` and `delta = 0.10` means the target average miss fraction is at most 10%, with a 90% confidence certificate—if calibration succeeds. This does **not** promise that every image misses at most 10% of its pixels, or that 90% of images have every pixel covered.

## 5. How does calibration choose the radius?

Before looking at calibration outcomes, the notebook sets a grid of possible radii. For every calibration image, it calculates the image loss at every radius.

For a given radius, the average calibration loss estimates the risk. RCPS also calculates an upper confidence bound (`UCB`) to account for finite calibration data. With the Hoeffding option, the formula is:

```text
UCB = min(1, average calibration loss
             + sqrt(log(1 / delta) / (2 * number of calibration images)))
```

Here:

- `average calibration loss` is the mean of the per-image losses at that radius.
- `number of calibration images` is 300 in this notebook.
- `delta` sets the confidence level; for example, `delta = 0.10` means 90% confidence.

The notebook also offers the WSR bound from Bates et al. It uses the same bounded, per-image losses but a different confidence-bound calculation. WSR can be tighter than Hoeffding in some cases, but not always.

The selection rule is:

1. Start with the smallest radius.
2. Check whether its risk upper bound is below `alpha`.
3. Also require the bounds for every larger radius to be below `alpha`.
4. Choose the first radius that passes.

Larger radii create wider intervals and cannot increase the missed-pixel loss. This nesting lets RCPS search the radius grid using the same calibration set. We do not split calibration into a radius-search part and a separate risk-check part. If no radius passes, the notebook reports **no certificate** for that `alpha`.

## 6. Why might calibration fail to certify a small `alpha`?

With 300 calibration images and `delta = 0.10`, the Hoeffding formula adds about `0.062` to the observed average loss. So even if calibration misses zero pixels, the resulting upper bound is about 6.2%. It cannot be below `alpha = 0.01` or `0.05` in this example.

That means the Hoeffding bound is too wide to certify those strict targets with this sample size. It does **not** prove the true risk is above the target. More representative calibration images can shrink the Hoeffding allowance. WSR may produce a different bound based on the observed losses.

The notebook repeats calibration for `alpha = 0.01, 0.05, 0.10, 0.20`, using `delta = 0.10`. Each row is a separate 90%-confidence claim; the four rows are not jointly certified at 90% confidence.

## 7. How do we evaluate on the test images?

After calibration chooses `lambda`, the notebook applies it to predictions for the untouched test images. Test results do not select the radius and do not create the certificate.

- **Test pixel coverage:** fraction of all test pixels inside their intervals. One minus this number is the average fractional-pixel miss loss on the test set.
- **Test whole-image coverage:** fraction of test images where every pixel is covered. This is a different, stricter quantity. RCPS here does not control it.
- **Mean interval width:** average size of the pixel intervals. It describes how wide the intervals are, not whether the risk guarantee holds.

The notebook also visualizes one test example: input, target, prediction, interval width, and missed pixels. This helps reveal where the model has difficulty. The formal risk target is still averaged over images; it does not promise the same error rate at every pixel or on every image.

## 8. Assumptions and limitations

The Hoeffding and WSR bounds assume image pairs in calibration are independent and identically distributed with future image pairs, conditional on the already-trained model. Pixels within an image can be dependent because we combine them into one loss. If multiple images come from the same patient, image-level independence may fail unless the split accounts for patient groups. The small dataset mirror does not document patient IDs, so this experiment cannot establish patient-level generalization.

The targets are LoDoPaB-derived, but the input CT scans are simulated using sparse views, noise, and FBP. This is a research stress test on a constructed image-to-image task, not a test on official LoDoPaB measurement inputs or a clinical validation.

## Related files

- [Runnable LoDoPaB notebook](notebooks/lodopab_rcps.ipynb)
- [More detailed method notes](docs/method.md)
- [Bates et al., *Distribution-Free, Risk-Controlling Prediction Sets*](https://doi.org/10.1145/3478535)
- [Tutorial that inspired this format](https://github.com/jatinbatra50/jatinbatra50/blob/provable-uq-tutorial/TUTORIAL.md)
