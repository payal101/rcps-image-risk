# RCPS for image regression: a readable walkthrough

This tutorial explains how the notebook uses risk-controlling prediction sets (RCPS) to control the average fraction of missed pixels in CT images. Equations use GitHub's `$$ ... $$` math formatting, with a plain-language explanation directly after each one.

## 1. What goes into the model?

Each example has two images:

- `X`: the input image, a sparse-view CT scan reconstructed using filtered back projection (FBP).
- `Y`: the target image, a LoDoPaB-derived CT image.

A small CNN learns a prediction rule called `f`. Given an input `X`, it produces a predicted image `f(X)`. The notebook trains `f` on 700 images, then freezes it. Calibration (300 images) chooses the interval size. The 100 test images are kept untouched until final evaluation.

## 2. How do we make an interval image?

For each pixel, the model's predicted intensity is the center. We add and subtract a chosen radius, \(\lambda\):

$$
T_{\lambda}(X)_p
=\left[\max\{0, f(X)_p-\lambda\},\;\min\{1, f(X)_p+\lambda\}\right].
$$

Here, \(p\) identifies a pixel and \(f(X)_p\) is the model's predicted value there. The endpoints are clipped to the target intensity range \([0,1]\). \(T_{\lambda}(X)\) is the collection of intervals for all pixels in image \(X\). A larger \(\lambda\) makes wider intervals.

## 3. How do we score one image?

For image \(i\), let \(P\) be its total number of pixels. Its loss is

$$
L_i(\lambda)
=\frac{1}{P}\sum_{p=1}^{P}
\mathbf{1}\!\left\{Y_{i,p}\notin T_{\lambda}(X_i)_p\right\}.
$$

The indicator \(\mathbf{1}\{\cdot\}\) equals 1 when the target pixel is outside its interval and 0 otherwise. Thus \(L_i(\lambda)\) is the fraction of missed pixels in image \(i\). For example, \(L_i(\lambda)=0.03\) means 3% of its pixels were missed.

Each image contributes **one loss number** between 0 and 1 to calibration. We do not pretend pixels inside an image are independent. Their misses are combined into one image-level number.

## 4. What risk are we trying to control?

Imagine repeatedly drawing a new image pair \((X,Y)\) from the population of interest and calculating its image loss. The population risk is

$$
R(\lambda)
=\mathbb{E}_{(X,Y)\sim\mathcal{D}}\!\left[L_{\lambda}(X,Y)\right].
$$

In words, \(R(\lambda)\) is the average fraction of pixels missed over new images. The expectation is over new image pairs, not over pixels treated as independent samples.

We choose:

- `alpha`: the largest acceptable average miss fraction per image.
- `delta`: the chance that calibration fails to certify the chosen radius.

The RCPS goal is

$$
\Pr_{S_{\mathrm{cal}}}\!\left\{R(\widehat{\lambda})>\alpha\right\}\leq\delta.
$$

Here, \(S_{\mathrm{cal}}\) is the random calibration sample and \(\widehat{\lambda}\) is the radius selected from it. In words: with confidence at least \(1-\delta\), the chosen intervals have expected miss fraction no greater than \(\alpha\) on a new image pair.

Example: `alpha = 0.10` and `delta = 0.10` means the target average miss fraction is at most 10%, with a 90% confidence certificate—if calibration succeeds. This does **not** promise that every image misses at most 10% of its pixels, or that 90% of images have every pixel covered.

## 5. How does calibration choose the radius?

Before looking at calibration outcomes, the notebook sets a grid of possible radii. For every calibration image, it calculates the image loss at every radius.

For a given radius, the average calibration loss estimates the risk. RCPS also calculates an upper confidence bound (`UCB`) to account for finite calibration data. With the Hoeffding option, the bound is

$$
U_{\mathrm{H}}(\lambda)
=\min\!\left\{1,\;
\widehat{R}_{\mathrm{cal}}(\lambda)
+\sqrt{\frac{\log(1/\delta)}{2n_{\mathrm{cal}}}}\right\}.
$$

Here:

- \(\widehat{R}_{\mathrm{cal}}(\lambda)\) is the average loss measured on calibration images at radius \(\lambda\).
- \(n_{\mathrm{cal}}\) is the number of calibration images (300 in this notebook).
- \(\delta\) sets the confidence level; for example, \(\delta=0.10\) means 90% confidence.

The notebook also offers the WSR bound from Bates et al. It uses the same bounded, per-image losses but a different confidence-bound calculation. WSR can be tighter than Hoeffding in some cases, but not always.

Let the tested radii be \(\lambda_1<\lambda_2<\cdots<\lambda_K\), and let \(U_j\) be the risk upper bound for radius \(\lambda_j\). The selection rule is

$$
\widehat{k}
=\min\left\{k:\;\max_{j\geq k}U_j<\alpha\right\},
\qquad \widehat{\lambda}=\lambda_{\widehat{k}}.
$$

The maximum over \(j\geq k\) is shorthand for checking the bound at radius \(k\) and every larger radius. In steps:

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
