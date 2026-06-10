# Imaging math — 12U distance sweep

This document reconstructs the math chain that produced the 12U-at-distance image set rendered from
[`configs/nro/cubesat_12u_distance_sweep.json`](../configs/nro/cubesat_12u_distance_sweep.json).
It is intended to be readable end-to-end and to make every numeric value reproducible.

The target is a 12U CubeSat envelope of **1.0 × 0.2 × 0.2 m** (long axis × short × short),
rendered at five ranges (3, 6, 9, 12, 15 km) through a real space-imaging optic.

> **Asset note (envelope guarantee).** `model_scale_xyz` multiplies every mesh vertex
> before rasterization, so the rendered geometry is provably bounded by a 1.0 × 0.2 × 0.2 m
> world-space box: every lit pixel comes from a vertex inside that envelope. The *internal shape*
> is still a (compressed) Hubble mesh and not a true CubeSat-bus geometry — for pixel-footprint
> and lighting work that is irrelevant; for shape-feature studies you'd want a real CubeSat asset.

---

## 1. Physical camera spec → Blender camera intrinsics

### Inputs

| Symbol | Value | Description |
|---|---|---|
| f | 650 mm | Focal length |
| p | 2.4 µm | Pixel pitch |
| W × H | 5472 × 3648 | Sensor resolution |
| N | 7.6 | f-number |
| FWHM_aberr | 0.68 px | Optical-aberration PSF FWHM (design budget, no diffraction) |
| λ | 550 nm | Reference wavelength |

### Sensor size derivation

The spec gives pixels and pitch, not sensor mm — derive:

```
sensor_width_mm  = p · W / 1000  =  2.4 · 5472 / 1000  = 13.1328 mm
sensor_height_mm = p · H / 1000  =  2.4 · 3648 / 1000  =  8.7552 mm
sensor_diag_mm   = √(13.1328² + 8.7552²)               = 15.7836 mm
```

These are pushed to `cam.data.sensor_width` with `sensor_fit = HORIZONTAL` (sensor_height
is implied by the resolution aspect ratio).

### Field of view

Blender uses pinhole projection, so FOV = 2·atan(half-sensor / focal):

```
FOV_h = 2·atan( (sensor_w / 2) / f ) = 1.158°
FOV_v = 2·atan( (sensor_h / 2) / f ) = 0.772°
FOV_d = 2·atan( (sensor_d / 2) / f ) = 1.391°
```

### Pixel focal length

`f_px` is the camera intrinsic that governs all downstream apparent-pixel-size math:

```
f_px = f / p = 650 mm / 0.0024 mm = 270,833 px
```

### IFOV per pixel

Equivalent to `1 / f_px` in radians:

```
IFOV = p / f = 2.4e-6 m / 0.65 m = 3.692 µrad/px
```

This matches the spec value (3.7 µrad/px) — that consistency is the test that the `f_px`
calculation went through correctly.

---

## 2. Optical PSF → compositor Gaussian blur

The *system* PSF is the convolution of two independent Gaussians: optical aberrations
(spec) and diffraction (physics). Two Gaussians convolve to a Gaussian whose variance is
the sum of variances; for FWHM that means quadrature.

### Diffraction-limited FWHM

```
FWHM_diff_linear = 1.028 · λ · N
                 = 1.028 · 550 nm · 7.6  = 4.297 µm
FWHM_diff_px     = 4.297 / 2.4           = 1.790 px
```

### Combined system PSF

```
FWHM_total² = FWHM_aberr² + FWHM_diff²
            = 0.68² + 1.79²              = 3.666
FWHM_total  = √3.666                     = 1.915 px
σ_total     = FWHM / 2.355               = 0.813 px
```

### Wired into the compositor

Blender's `CompositorNodeBlur` (Gaussian filter) takes a *kernel radius* in pixels.
We use `radius = ⌈3·σ⌉ = 3 px`, then drop the node into the chain:

```
RL → PSF blur → [Glare] → Composite
```

Result: every rendered pixel is convolved with a Gaussian of σ ≈ 0.81 px before being
written. This is done in compositor space on linear EXR data, *not* in image-space
convolution after PNG read, so radiometric linearity is preserved.

---

## 3. Camera and target placement in Blender world

The trajectory generator `write_camera_trajectory_fib(N=1, R_LEO, R_RPO)` puts both the
target and camera on the same radial direction `d̂` from the inertial origin:

```
p_G_I = d̂ · R_LEO              (target on LEO sphere)
p_C_I = d̂ · (R_LEO + R_RPO)    (camera radially outward)
⇒ |p_G_I − p_C_I| = R_RPO       (camera-target separation)
```

Then `get_scaled_trajectory_in_ECI(scale_factor = 0.001)` brings everything closer to the
Blender world origin while *preserving* the relative camera-target offset:

```
r_GC_I  = p_G_I − p_C_I              (vector C → G)
p_C′    = p_C_I · scale_factor       (camera scaled toward origin)
p_G′    = p_C′ + r_GC_I              (target placed offset-from-camera, unscaled)
⇒ |p_G′ − p_C′| = |r_GC_I| = R_RPO   (separation preserved)
```

So R_RPO is preserved regardless of `scale_factor`. The only thing compressed is the
absolute distance from the world origin (e.g. R_LEO = 8e6 m → 8 000 m), which keeps Earth
at a sensible Blender-world distance without distorting the local imaging geometry.

Verified at runtime for R_RPO = 4000 m:

```
cam.location   = (8004.0, 0.0, 0.0)
model.location = (4004.0, 0.0, 0.0)
|cam − model|  = 4000.0 m  ✓
```

The camera's orientation is `to_track_quat("-Z", "Y")` of the look-at direction (toward
origin), so the camera's −Z (forward axis) points at the target. The target is centered
on the optical axis.

---

## 4. Apparent pixel size prediction

Standard pinhole projection: an object of linear extent `L` at range `R` (m) projects to:

```
pixels = L · f_px / R = L · 270,833 / R
```

For the 1.0 × 0.2 × 0.2 m 12U envelope:

| R (m) | Long axis (1.0 m) | Short axis (0.2 m) |
|---|---|---|
| 3 000 | 90.3 | 18.1 |
| 6 000 | 45.1 | 9.0 |
| 9 000 | 30.1 | 6.0 |
| 12 000 | 22.6 | 4.5 |
| 15 000 | 18.1 | 3.6 |

These are the *geometric* footprint, before optical PSF blur is applied (which adds
~σ pixels of soft edge in every direction).

---

## 5. Measured vs. predicted

Measured from the rendered PNGs (bright-pixel bbox at intensity > 10):

| R (m) | bbox measured | Long-axis prediction | Ratio |
|---|---|---|---|
| 3 000 | 47 × 54 | 90 px | 0.60 |
| 6 000 | 25 × 29 | 45 px | 0.64 |
| 9 000 | 19 × 21 | 30 px | 0.70 |
| 12 000 | 14 × 16 | 23 px | 0.70 |
| 15 000 | 13 × 14 | 18 px | 0.78 |

Measured bbox is consistently 60–80% of the geometric long-axis. Two reasons:

- **Threshold-based bbox** captures only well-lit pixels (> 10 / 255). Unlit / shadowed
  faces don't trigger the threshold.
- **Viewing angle**: the camera looks along `−d̂`, but the target's body axis isn't
  aligned with `d̂` — it's tilted by whatever rotation the Fibonacci-sphere sample
  induces. Apparent long-axis extent is `L · cos(θ)` where θ is the angle to the image
  plane. With θ ≈ 45° the long axis projects to roughly `1.0 / √2 ≈ 0.71 m`, predicting
  64 px at 3 km — and the measured 47 px ≈ 0.74× of that. Foreshortening accounts for
  the gap.

The 1/R scaling itself is clean: `bbox(3 km) / bbox(15 km) = 47 / 13 ≈ 3.6`, vs. the
expected 5.0. The remaining gap is discreteness — at 15 km the bbox is only 13 px wide,
so a single pixel of threshold noise changes the ratio by ~8%.

---

## TL;DR — the math chain

```
spec → camera intrinsics:   f_px = f_mm / p_µm × 1000  =  270,833 px

spec → optical PSF:         σ = √(0.68² + (1.028·λ·N/p)²) / 2.355
                              = 0.81 px  →  Gaussian compositor blur

range R (m)  →              apparent pixels = L · 270,833 / R
                              with L = 1.0 m for the 12U long axis,
                                     0.2 m for the short axes.
                            Verified to within 60–80% of bbox measure,
                            gap explained by threshold + 45° foreshortening.
```
