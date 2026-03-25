---
title: "FP8 Quantization and Beyond: From First Principles to Intel Neural Compressor"
description: A deep dive into quantization for LLM inference -- covering INT, FP8, and MXFP4 formats, the science of calibration, and Intel Neural Compressor's quantization strategies on Gaudi accelerators.
date: 2026-03-25 00:00:00 +0530
author: skrohit
categories: [quantization, inference]
tags: [llms, fp8, mxfp4, quantization, neural-compressor, gaudi, deep-learning]
math: true
pin: false
---

# FP8 Quantization and Beyond: From First Principles to Intel Neural Compressor 🚀

Large language models are hungry -- hungry for memory, hungry for compute. A 70B parameter model in BF16 needs ~140 GB just to hold its weights. Quantization is the most practical technique to tame this appetite: represent the same numbers in fewer bits, and you slash both memory footprint and compute cost.

This post builds up quantization from first principles, walks through INT and FP8 formats, dives into the strategies used by Intel Neural Compressor (INC) for FP8 quantization on Gaudi accelerators, and finishes with the emerging MXFP4 format.

---

## 📐 1. What Is Quantization?

Quantization is the process of mapping values from a **high-precision** representation (e.g., FP32, BF16 -- 16 or 32 bits per value) to a **low-precision** representation (e.g., FP8, INT8, INT4 -- 4 or 8 bits per value). The goal is to reduce memory footprint and increase computation throughput, at the cost of some precision loss.

Every quantization scheme, regardless of the target precision, follows the same fundamental pattern:

**Quantize** (high precision → low precision):

$$x_q = \text{round}\!\left(\frac{x}{\text{scale}} + \text{zero\_point}\right)$$

**Dequantize** (low precision → high precision):

$$\hat{x} = (x_q - \text{zero\_point}) \times \text{scale}$$

Where:
- $x$ = original high-precision value
- $x_q$ = quantized low-precision value
- $\text{scale}$ = a scalar that maps the real-value range to the quantized range
- $\text{zero\_point}$ = an offset that aligns the zero of the real range with a quantized code
- $\hat{x}$ = reconstructed value (approximately equal to $x$, with some quantization error)

The **quantization error** is $\epsilon = x - \hat{x}$. This error is the price paid for compression. The entire art of quantization is choosing `scale` and `zero_point` to minimize this error across all elements.

---

## 🔢 2. Integer Quantization (INT8, INT4)

Integer formats have a **uniform, evenly-spaced** grid of representable values. INT8 can represent integers from $-128$ to $127$ (signed) or $0$ to $255$ (unsigned). INT4 has only 16 representable values ($-8$ to $7$).

### 2.1 Symmetric Quantization (zero_point = 0)

Most common for weights. The representable range is centered at zero:

$$\text{scale} = \frac{\max(\lvert x \rvert)}{q_{\max}}$$

where $q_{\max} = 127$ for INT8 or $7$ for INT4.

<details markdown="1">
<summary>💡 INT8 Symmetric Example</summary>

Given a weight tensor `x = [-0.8, 0.3, 0.5, -1.2]`:

```
scale = max(|-0.8|, |0.3|, |0.5|, |-1.2|) / 127 = 1.2 / 127 = 0.009449

Quantize:
  -0.8 / 0.009449 = -84.67  → round → -85
   0.3 / 0.009449 =  31.75  → round →  32
   0.5 / 0.009449 =  52.92  → round →  53
  -1.2 / 0.009449 = -127.0  → round → -127

x_q = [-85, 32, 53, -127]      (stored as 8-bit integers, 1 byte each)

Dequantize:
  -85  * 0.009449 = -0.8032
   32  * 0.009449 =  0.3024
   53  * 0.009449 =  0.5008
  -127 * 0.009449 = -1.2000

x_hat   = [-0.8032, 0.3024, 0.5008, -1.2000]
error   = [ 0.0032, 0.0024, 0.0008,  0.0000]
```

</details>

### 2.2 Asymmetric Quantization (non-zero zero_point)

Common for activations like ReLU outputs that are always $\geq 0$. Instead of wasting half the quantized range on negative values that never occur, the range is shifted:

$$\text{scale} = \frac{x_{\max} - x_{\min}}{q_{\max} - q_{\min}}, \qquad \text{zero\_point} = \text{round}\!\left(q_{\min} - \frac{x_{\min}}{\text{scale}}\right)$$

This allows mapping a range like $[0, 6]$ (ReLU6 output) to the full $[0, 255]$ unsigned INT8 range.

### 2.3 INT4 -- Fewer Bits, More Error

INT4 has the same formula, but with $q_{\max} = 7$. The spacing between representable values is much wider:

<details markdown="1">
<summary>💡 INT4 Symmetric Example</summary>

Given `x = [-0.8, 0.3, 0.5, -1.2]`:

```
scale = 1.2 / 7 = 0.1714

Quantize:
  -0.8 / 0.1714 = -4.67  → round → -5
   0.3 / 0.1714 =  1.75  → round →  2
   0.5 / 0.1714 =  2.92  → round →  3
  -1.2 / 0.1714 = -7.00  → round → -7

Dequantize:
  -5 * 0.1714 = -0.857
   2 * 0.1714 =  0.343
   3 * 0.1714 =  0.514
  -7 * 0.1714 = -1.200

error = [0.057, 0.043, 0.014, 0.000]   ← much larger errors than INT8
```

</details>

The fewer the bits, the wider the spacing between representable values, the larger the quantization error.

---

## 🧮 3. Floating-Point Representation

Before discussing FP8, we need to understand how floating-point numbers work. Every floating-point number has three fields:

```
| sign (1 bit) | exponent (E bits) | mantissa (M bits) |
```

The value is computed as:

$$\text{value} = (-1)^{\text{sign}} \times 2^{(\text{stored\_exponent} - \text{bias})} \times (1 + \text{mantissa} / 2^M)$$

### 3.1 What Is the Exponent Bias?

The `stored_exponent` field is an **unsigned integer** (always $\geq 0$). But we need to represent both very large values ($2^{10}$) and very small values ($2^{-10}$). The **bias** is an offset that shifts the exponent so it can be effectively negative:

$$\text{effective\_exponent} = \text{stored\_exponent} - \text{bias}$$

The standard formula for bias is: $\text{bias} = 2^{(E-1)} - 1$, where $E$ is the number of exponent bits.

For FP32 ($E = 8$): $\text{bias} = 127$. Stored exponent range $[1, 254]$ maps to effective range $[-126, +127]$.

### 3.2 Non-Uniform Spacing

The critical difference from integers: floating-point values are **logarithmically spaced**. Values near zero are densely packed; values at large magnitudes are sparsely spaced:

```
INT8 grid:   ... -3, -2, -1, 0, 1, 2, 3 ...     (uniform spacing of 1)

FP grid:     ... 0.125, 0.1875, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0 ...
                  ^^^^^^^^^^^^^^^^^^^^^           ^^^^^^^^^^^^^^^^^^
                  dense near zero                 sparse at large values
```

This is a natural fit for neural network weights and activations, which tend to follow bell-curve distributions centered near zero -- most values are small, and few are large.

---

## ⚡ 4. FP8 Formats: E4M3 and E5M2

FP8 is an 8-bit floating-point format. There are two standard variants:

| Format | Exponent Bits | Mantissa Bits | Bias | Max Value | Min Positive Subnormal | Values per Exponent Bucket |
|--------|:---:|:---:|:---:|---:|---:|:---:|
| **E4M3** | 4 | 3 | 7 | 448.0 | ~0.00195 | 8 |
| **E5M2** | 5 | 2 | 15 | 57,344.0 | ~0.0000153 | 4 |

**E4M3** trades range for precision: more mantissa bits mean finer resolution within each exponent bucket, making it the standard choice for **weights and forward-pass activations** in LLM inference.

**E5M2** trades precision for range: more exponent bits mean a wider representable range, making it suitable for **gradients during training** where the dynamic range can be very large.

### 4.1 E4M3 Exponent Bias in Detail

With 4 exponent bits and bias = 7:

| Stored Exponent | Effective Exponent | Multiplier |
|:---:|:---:|---:|
| 0 | (subnormal) | $2^{-6}$ |
| 1 | $1 - 7 = -6$ | $2^{-6} = 0.0156$ |
| 7 | $7 - 7 = 0$ | $2^0 = 1.0$ |
| 14 | $14 - 7 = 7$ | $2^7 = 128.0$ |
| 15 | reserved (NaN) | -- |

The 3 mantissa bits give 8 fractional steps per exponent bucket: $1.000, 1.125, 1.250, 1.375, 1.500, 1.625, 1.750, 1.875$.

So in the exponent-7 bucket ($2^7 = 128$), the representable values are:
$128 \times 1.000 = 128$, $128 \times 1.125 = 144$, ..., $128 \times 1.750 = 224$.

The spacing at this magnitude is $128 \times (1/8) = 16$. Compare this to the exponent-0 bucket where spacing is $1 \times (1/8) = 0.125$. Values near zero have 128x finer resolution than values near 224.

### 4.2 E5M2 Exponent Bias in Detail

With 5 exponent bits and bias = 15:

| Stored Exponent | Effective Exponent | Multiplier |
|:---:|:---:|---:|
| 0 | (subnormal) | $2^{-14}$ |
| 1 | $1 - 15 = -14$ | $2^{-14} = 0.0000610$ |
| 15 | $15 - 15 = 0$ | $2^0 = 1.0$ |
| 30 | $30 - 15 = 15$ | $2^{15} = 32768.0$ |
| 31 | reserved (inf/NaN) | -- |

Only 2 mantissa bits means only 4 values per bucket: $1.00, 1.25, 1.50, 1.75$. Much coarser than E4M3's 8 values per bucket, but the exponent range spans $2^{-14}$ to $2^{15}$ -- a dynamic range of $\sim 10^9$.

### 4.3 Configurable Bias on Gaudi

Gaudi accelerators allow **changing the exponent bias at runtime**. This is a hardware feature that effectively shifts the entire FP8 number line without any computation:

| | Supported Bias Values |
|---|---|
| **Gaudi2** (E4M3) | $[3, 7, 11, 15]$ -- only 4 choices |
| **Gaudi3** (E4M3) | $[0, 1, 2, ..., 62]$ -- 63 choices |

By changing the bias from the default 7 to, say, 11:

```
Default bias=7:   representable range ≈ [0.002, 448]
Bias=11:          range shifts toward smaller values: [0.0001, 28]
```

The hardware adjusts the exponent bias register, achieving the same effect as dividing by a scale factor -- but at **zero computational cost**. This is the foundation of INC's **HW-aligned scaling** strategy discussed later.

---

## 🔄 5. FP8 Quantization

Because FP8 is a floating-point format, the hardware can directly cast between FP8 and FP32/BF16. The role of the `scale` is to shift the tensor's dynamic range to align with FP8's representable range.

**Quantize:**

$$x_q = \text{cast\_to\_fp8}\!\left(\frac{x}{\text{scale}}\right)$$

**Dequantize:**

$$\hat{x} = \text{cast\_from\_fp8}(x_q) \times \text{scale}$$

There is **no zero_point** in FP8 quantization -- it is always symmetric around zero, because the floating-point format already has a sign bit.

<details markdown="1">
<summary>💡 FP8 E4M3 Worked Example</summary>

Suppose a tensor has values: `x = [-12.5, 0.03, 4.7, -0.001]`

```
Step 1: Compute scale
  maxabs   = max(|-12.5|, |0.03|, |4.7|, |-0.001|) = 12.5
  fullscale = 448.0  (E4M3 max)
  backoff   = 0.5
  scale     = 12.5 / (448.0 * 0.5) = 12.5 / 224.0 = 0.05580

Step 2: Quantize (divide by scale, then cast)
  -12.5 / 0.05580 = -224.01  → cast_to_fp8 → -224.0   (representable in E4M3)
   0.03 / 0.05580 =    0.538 → cast_to_fp8 →    0.5    (nearest E4M3 value)
   4.7  / 0.05580 =   84.23  → cast_to_fp8 →   80.0    (nearest E4M3 value)
  -0.001/ 0.05580 =   -0.018 → cast_to_fp8 →   -0.0195 (nearest E4M3 subnormal)

Step 3: Dequantize (cast back, multiply by scale)
  -224.0   * 0.05580 = -12.499
   0.5     * 0.05580 =   0.02790
   80.0    * 0.05580 =   4.464
  -0.0195  * 0.05580 =  -0.001090

x_hat = [-12.499, 0.02790, 4.464, -0.001090]
error = [ 0.001,  0.00210, 0.236,  0.000090]
```

Notice:
- `-12.5` (large value, near top of range) has tiny error
- `4.7` has larger error because at magnitude ~84 in the scaled domain, E4M3 values are spaced 4-8 apart
- `0.03` has small absolute error because near-zero values have dense FP8 representation

</details>

---

## ⚖️ 6. INT vs FP8 Quantization: A Comparison

| Property | INT8/INT4 | FP8 (E4M3/E5M2) |
|----------|-----------|------------------|
| **Value spacing** | Uniform (evenly spaced) | Logarithmic (dense near 0, sparse near max) |
| **Zero point** | Can be non-zero (asymmetric) | Always 0 (symmetric only) |
| **Rounding** | Round to nearest integer | Cast to nearest FP8 representable value |
| **Good at** | Uniform distributions | Bell-curve / Gaussian distributions |
| **Scale formula** | $\max(\lvert x \rvert) / q_{\max}$ | $\max(\lvert x \rvert) / (\text{fullscale} \times \text{backoff})$ |
| **HW operation** | Integer multiply-accumulate | FP8 multiply-accumulate (native on Gaudi, H100) |

For LLM inference, FP8 is generally preferred because neural network weight and activation distributions are approximately Gaussian (bell-shaped, centered near zero). FP8's logarithmic spacing naturally allocates more precision where most values lie.

---

## 🏭 7. How Quantized Inference Works: The Role of Accumulation

In a normal BF16 forward pass through a Linear layer:

```
output = input @ weight.T + bias        (all in BF16)
```

In FP8 quantized inference:

```
input_fp8  = cast_to_fp8(input / input_scale)
weight_fp8 = (pre-quantized at calibration time, already in FP8)

output_fp32 = fp8_matmul(input_fp8, weight_fp8.T)       # FP8 multiply, FP32 accumulate
output_bf16 = to_bf16(output_fp32 * input_scale * weight_scale) + bias
```

### 7.1 What "Accumulation" Means

A matrix multiply $C = A \times B$ where $A$ is $[M, K]$ and $B$ is $[K, N]$ computes each output element as a **dot product** -- the sum of $K$ individual products:

$$C[i][j] = \sum_{k=0}^{K-1} A[i][k] \times B[k][j]$$

For a typical LLM hidden dimension of $K = 4096$, each output element is the sum of 4096 individual products. The hardware computes this iteratively:

```
accumulator = 0                          # ← this register holds the running sum
for k in range(4096):
    product = A[i][k] * B[k][j]         # one FP8 × FP8 multiplication
    accumulator += product               # ← "accumulation": adding product to running sum
result = accumulator
```

### 7.2 Why Accumulation Must Be in FP32

**Problem 1 -- Range overflow.** Each FP8 E4M3 product has max value $448 \times 448 = 200{,}704$. After summing 4096 such products, the result could reach $\sim 8 \times 10^8$. FP8's max is 448. The accumulator would overflow after just a few additions.

**Problem 2 -- Precision loss from repeated addition (swamping).** When adding a small number to a large running sum in low precision, the small number gets rounded away:

```
In hypothetical low-precision float:
  accumulator = 100.0
  product     =   0.3
  100.0 + 0.3 = 100.3 → rounds to 100.0 (nearest representable)
  The 0.3 is simply LOST.

After 1000 such additions:
  Should be: 100 + 1000 × 0.3 = 400.0
  Actually:  100.0  (none of the small values accumulated)
```

With FP8's 3 mantissa bits, once the accumulator reaches ~16× the product magnitude, new products round away to nothing.

**FP32 accumulation solves both:** its max value ($3.4 \times 10^{38}$) eliminates overflow, and its 23 mantissa bits ($\sim 8$ million distinguishable levels per exponent) prevent swamping.

The hardware architecture:

```
      FP8       FP8
       |         |
       v         v
  ┌──────────────────┐
  │  FP8 multiplier  │  ← small, fast (2x throughput vs BF16)
  └────────┬─────────┘
           │ intermediate product (promoted to FP32)
           v
  ┌──────────────────┐
  │  FP32 adder      │  ← preserves numerical accuracy
  │  (accumulator)   │
  └────────┬─────────┘
           │ FP32 running sum (after K iterations)
           v
  ┌──────────────────┐
  │  Cast to BF16    │  ← final output
  └──────────────────┘
```

The FP8 multiplier array is half the size of BF16 (smaller operands → simpler circuits), giving ~2x throughput. The FP32 accumulator stays the same regardless -- it is the bottleneck for area, not speed.

---

## 🔬 8. Intel Neural Compressor (INC) FP8 Quantization

Intel Neural Compressor (INC) provides a comprehensive FP8 quantization framework designed for Gaudi accelerators. It implements a **two-phase** approach: MEASURE (calibration) then QUANTIZE.

INC's FP8 quantization has several configurable dimensions. Understanding these is essential for tuning quantization quality.

### 8.1 The Scale Formula

At the core, INC computes:

$$\text{scale} = \frac{\text{maxabs}}{\text{fullscale} \times \text{backoff}}$$

Where:
- **maxabs** = the maximum absolute value observed (during calibration, or from the weight tensor)
- **fullscale** = the maximum representable value in the target FP8 format (e.g., 448.0 for E4M3)
- **backoff** = a safety margin factor (e.g., 0.5 for weights, 0.25 for activations). A smaller backoff widens the quantization range, reducing clipping risk but increasing quantization noise

### 8.2 Scale Granularity

Granularity determines **how many elements share a single scale value**.

**Per-Tensor Scaling (PTS)** -- the default: one single scale for the **entire tensor**.

For a weight matrix of shape $[4096, 4096]$ (16M elements), PTS computes one scalar from $\max(\lvert \text{all 16M values} \rvert)$. Every element is divided by the same scale.

```
Weight [4 × 8]:

Channel 0:  [ 0.01,  0.02, -0.03,  0.01, ...]  ← range: [-0.03, 0.03]
Channel 1:  [ 1.20, -0.80,  1.50, -1.10, ...]  ← range: [-1.30, 1.50]
Channel 2:  [ 0.00,  0.00,  0.01,  0.00, ...]  ← range: [0.00, 0.01]
Channel 3:  [-5.00,  3.20, -4.80,  2.90, ...]  ← range: [-5.00, 4.10]

PTS: scale = max_over_all = 5.00 / (448 × 0.5) = 0.02232

Problem: Channel 2's tiny values (0.01) are crushed.
  0.01 / 0.02232 = 0.448 → cast_to_fp8 → 0.4375
  All distinctions within Channel 2 are nearly destroyed.
```

**Per-Channel Scaling (PCS)**: one scale **per output channel** (per row of the weight matrix).

```
Channel 0: scale = 0.03  / (448 × 0.5) = 0.000134
Channel 1: scale = 1.50  / (448 × 0.5) = 0.006696
Channel 2: scale = 0.01  / (448 × 0.5) = 0.0000446
Channel 3: scale = 5.00  / (448 × 0.5) = 0.02232

Now Channel 2: 0.01 / 0.0000446 = 224.0 → cast_to_fp8 → 224.0  (exact!)
Each channel uses FP8's full dynamic range independently.
```

PCS is more accurate but stores more scale values (one FP32 per channel vs one total). For a $[4096, 4096]$ matrix, that is 4096 × 4 = 16 KB of scale data -- still only 0.1% overhead.

For **activations**, "channel" means the last (feature) dimension. Per-channel activation scaling is mainly used in **dynamic quantization** mode, where scales are computed on-the-fly for each input.

### 8.3 Scale Value Type

The value type determines **how the scale is computed** from tensor data.

**MAXABS** (default): Scale derived from the maximum absolute value.

```
scale = max(|tensor|) / (fullscale × backoff)
```

Simple, fast, and effective. One `max(|x|)` reduction over the tensor. Works well when value distributions are roughly Gaussian. Sensitive to outliers -- a single extreme value forces the scale up, compressing precision for all other values.

**OPT (MMSE-optimized)**: Scale found by brute-force searching for the value that minimizes reconstruction error.

```
for each candidate_scale in [list of possible scales]:
    quantized    = cast_to_fp8(tensor / candidate_scale)
    reconstructed = cast_from_fp8(quantized) × candidate_scale
    error        = ||tensor - reconstructed||²
    if error < best_error:
        best_scale = candidate_scale
```

OPT tries every candidate scale (e.g., all HW-aligned values, or powers of 2 from $2^{-10}$ to $2^{9}$), quantizes and dequantizes the tensor, and picks the scale with the **lowest mean squared error**.

OPT is more accurate than MAXABS because it accounts for the actual error distribution rather than just the range extremes. If 99.9% of values cluster in $[-0.5, 0.5]$ but one outlier sits at 10.0, MAXABS maps FP8 to $[-10, 10]$ and wastes precision for the majority. OPT would find a tighter scale that sacrifices the outlier but greatly improves precision for the 99.9%.

The tradeoff: significantly more expensive (iterates over 20+ candidate scales, doing full quantize/dequantize each time). Practical only for **weights** (static tensors, computed once). Methods like `maxabs_hw_opt_weight` use OPT for weights and MAXABS for activations.

**FIXED_VALUE**: Scale is a predetermined constant, independent of tensor data.

With `SCALE_UNIT` rounding, the scale is always 1.0. With `HW_ALIGNED_FIXED`, it is the smallest HW-aligned scale for the device. No calibration needed, but worst accuracy since the scale bears no relation to actual data. Useful only for debugging or quick smoke tests.

**Accuracy ranking: OPT > MAXABS >> FIXED**
**Speed ranking: FIXED > MAXABS >> OPT**

### 8.4 Scale Rounding Method

After computing the scale, it can be **rounded** to constrain it to values that the hardware can apply efficiently.

**Arbitrary (IDENTITY)**: No rounding. The scale is used exactly as computed -- any floating-point value. Maximum accuracy, but the quantize/dequantize step requires a real multiplication/division at runtime.

**POW2**: Round up to the nearest power of 2:

$$\text{scale\_pow2} = 2^{\lceil \log_2(\text{scale}) \rceil}$$

Example: $\text{scale} = 3.5 \Rightarrow \log_2(3.5) \approx 1.81 \Rightarrow \lceil 1.81 \rceil = 2 \Rightarrow 2^2 = 4.0$

Multiplying/dividing by a power of 2 is equivalent to a bit-shift in the exponent field -- essentially free on hardware. Slight accuracy loss due to the ceiling rounding.

**HW_ALIGNED**: Round to the nearest hardware-supported exponent bias value. This is a stricter form of POW2 that further constrains the scale to values the Gaudi hardware can apply through its exponent bias register at **zero computational cost**.

On **Gaudi2**: only 4 supported exponent biases $[3, 7, 11, 15]$ for E4M3, so the scale factor jumps in steps of $2^4 = 16$. Very coarse.

On **Gaudi3**: supports biases 0-62, so HW_ALIGNED is essentially identical to POW2 -- any power of 2 works.

| Rounding | Accuracy | HW Speed | # Possible Scales |
|----------|----------|----------|-------------------|
| Arbitrary | Best | Slowest (real mul/div) | Infinite |
| POW2 | Good | Fast (exponent bit-shift) | ~20 |
| HW_ALIGNED | Coarse on Gaudi2, ~POW2 on Gaudi3 | Fastest (free, via exponent bias register) | 4 (Gaudi2), 63 (Gaudi3) |

### 8.5 Preset Scale Methods

INC bundles granularity, value type, rounding, and backoff into named **scale method presets**. The most important ones:

| Scale Method | Weights | Activations | Notes |
|---|---|---|---|
| **`maxabs_hw`** (default) | PTS, MAXABS, HW_ALIGNED, backoff=0.5 | PTS, MAXABS, HW_ALIGNED, backoff=0.25 | Best HW performance. Recommended starting point. |
| `maxabs_pow2` | PTS, MAXABS, POW2, backoff=0.5 | PTS, MAXABS, POW2, backoff=0.25 | More scale options than HW_ALIGNED. |
| `maxabs_arbitrary` | PTS, MAXABS, IDENTITY, backoff=0.5 | PTS, MAXABS, IDENTITY, backoff=0.25 | Best accuracy, no HW rounding. |
| `maxabs_hw_opt_weight` | PTS, **OPT**, HW_ALIGNED | PTS, MAXABS, HW_ALIGNED | Better weight accuracy via MMSE search. |
| `maxabs_pow2_opt_weight` | PTS, **OPT**, POW2 | PTS, MAXABS, POW2 | OPT weights + POW2 rounding. |
| `act_maxabs_hw_weights_pcs_maxabs_pow2` | **PCS**, MAXABS, POW2 | PTS, MAXABS, HW_ALIGNED | Per-channel weights for better accuracy. |
| `unit_scale` | FIXED (scale=1.0) | FIXED (scale=1.0) | Debug/baseline only. |

INC also supports per-node, per-layer-index, and per-layer-type overrides via a dictionary-based scale method config, allowing fine-grained control over individual modules in the model.

---

## 🧪 9. Calibration: Why It Is Required

FP8 has a very narrow representable range compared to BF16/FP32. Different layers and tensors in a model have vastly different value distributions -- one layer's weights might range in $[-0.5, 0.5]$ while its activations range in $[-100, 100]$.

Without calibration, you'd have to guess scales. A bad guess means either:

- **Clipping** (scale too small, range too narrow): values exceeding the FP8 range saturate to max/min, losing information about large values.
- **Underutilization** (scale too large, range too wide): small values collapse to zero or the same FP8 code, wasting precision.

**Calibration solves this** by running representative data through the model and observing actual value distributions, then computing a per-tensor (or per-channel) scale that maps each tensor's dynamic range to the FP8 representable range as efficiently as possible.

### 9.1 INC's Two-Phase Approach

**Phase 1: MEASURE (Calibration)**

Every non-blocklisted module (e.g., `nn.Linear`) is instrumented with observer hooks. The `MaxAbsObserver` tracks the running maximum absolute value across all calibration samples:

```
# Pseudocode for MaxAbsObserver
class MaxAbsObserver:
    state = 0

    def observe(x):
        state = max(state, max(|x|))
```

Each patched module's forward pass becomes:

```
# Pseudocode for measurement forward pass
def forward_measure(input):
    observe_input(input)           # record input maxabs
    output = original_forward(input)  # run the real computation in BF16
    observe_output(output)         # record output maxabs
    return output
```

**Weights are measured once** at preparation time (they are static). **Activations** are measured on every calibration batch. After calibration, maxabs values are saved to `.npz` files.

**Phase 2: QUANTIZE**

The saved measurements are loaded. For each module, scales are computed from the measurements using the chosen scale method (MAXABS, OPT, etc.) and rounding (HW_ALIGNED, POW2, etc.). Weights are then cast to FP8 in-place. Activation scales are stored for runtime use.

### 9.2 KV Cache Alignment (Post-Processing)

After the MEASURE phase, a post-processing step fixes a consistency issue between attention matmul operations and the KV cache. During calibration, the framework independently measures:

- `matmul_qk` (Q @ K^T) -- measures K as its 2nd input
- `k_cache` -- measures K as its input
- `matmul_av` (Attention @ V) -- measures V as its 2nd input
- `v_cache` -- measures V as its input

These can record **different maxabs values** for logically the same tensor (K or V), because K/V flow through different code paths during the calibration forward pass. At inference time, the matmuls always read K/V **from the cache**, not directly from the projection output. The post-processing step overwrites the matmul's input measurement with the cache's measurement, ensuring scale consistency:

```
# Pseudocode for KV cache alignment
for each layer:
    if matmul_av.input[1] != v_cache.input[0]:
        matmul_av.input[1] = v_cache.input[0]    # cache is source of truth
    if matmul_qk.input[1] != k_cache.input[0]:
        matmul_qk.input[1] = k_cache.input[0]
```

Without this fix, the matmul would use a different scale than what the cache was quantized with, causing systematic numerical errors.

---

## 🔀 10. Static vs Dynamic Quantization

### 10.1 Static Quantization (Default)

Scales for activations are computed **once during calibration** and reused for every inference call. The flow is:

```
Calibration:  Run model on sample data → observe ranges → save maxabs → compute scales
Inference:    Load pre-computed scales → apply same fixed scale to every input
```

The quantization operation at runtime is minimal -- just a cast with a pre-loaded scale:

```
# Pseudocode for static quantization forward
def forward(x):
    x_fp8 = cast_to_fp8(x, precomputed_scale_inv)   # scale never changes
    return x_fp8
```

### 10.2 Dynamic Quantization

Scales for activations are computed **on-the-fly from the actual input** at each inference call. No calibration data is needed:

```
# Pseudocode for dynamic quantization forward
def forward(x):
    scale = max(|x|) / (fullscale × 0.5)         # computed NOW, from THIS input
    scale = round_to_pow2(scale)
    x_fp8 = cast_to_fp8(x, 1/scale)
    return (x_fp8, scale)                          # scale must travel with the data
```

| Aspect | Static | Dynamic |
|--------|--------|---------|
| Calibration needed | Yes (MEASURE → QUANTIZE, multiple steps) | No (QUANTIZE only, no measurement files) |
| Activation scale accuracy | Approximate (based on calibration data) | Exact (computed from the actual input) |
| Extra compute per inference call | None | `max(\|x\|)` + scale computation |
| Risk of clipping unseen distributions | Yes (if inference data differs from calibration data) | No (adapts to each input) |
| Throughput | Higher (no per-call overhead) | Lower (scale computation adds latency) |

### 10.3 When to Use Which

| Scenario | Recommendation |
|----------|---------------|
| Production, maximum throughput | **Static** -- no per-call overhead |
| Diverse/unpredictable inputs, accuracy-critical | **Dynamic** -- adapts to each input |
| Quick experimentation, no calibration data available | **Dynamic** -- zero setup |
| Calibration is expensive (huge model, limited data) | **Dynamic** -- skips calibration entirely |

### 10.4 Enabling Dynamic Quantization in INC

Set `"dynamic_quantization": true` in the config JSON and use one of the two dynamic-only scale methods:

```json
{
    "mode": "QUANTIZE",
    "observer": "maxabs",
    "scale_format": "CONST",
    "scale_method": "act_maxabs_pcs_pow2_weight_maxabs_pts_pow2_hw",
    "dynamic_quantization": true,
    "blocklist": {
        "types": ["Matmul", "B2BMatmul", "KVCache", "VLLMKVCache"]
    },
    "dump_stats_path": ""
}
```

Dynamic quantization supports two scale methods:

| Method | Weights | Activations |
|--------|---------|-------------|
| `act_maxabs_pcs_pow2_weight_maxabs_pts_pow2_hw` | PTS, MAXABS, HW_ALIGNED | **PCS**, MAXABS, POW2 |
| `maxabs_pcs_pow2` | **PCS**, MAXABS, POW2 | **PCS**, MAXABS, POW2 |

Both use **per-channel scaling for activations** -- since scales are computed at runtime anyway, the additional cost of per-channel (vs per-tensor) is marginal, and the accuracy benefit is significant.

Constraints: `scale_format` must be `CONST`, HW_ALIGNED rounding is not allowed for activations, and no MEASURE phase is needed.

---

## 🚫 11. Layers Typically Excluded from Quantization

Not every layer in an LLM benefits from quantization. Certain layers are kept in BF16 because they are **numerically sensitive, make discrete decisions, or offer negligible compute savings**.

### 11.1 Language Model Head (`lm_head`)

The `lm_head` is the final linear layer projecting the hidden state to vocabulary logits (a vector of size `vocab_size`, e.g., 128,256). Token selection depends on **relative ordering and tiny magnitude differences** between logits:

```
Token "the":  logit = 5.213
Token "a":    logit = 5.198       ← difference is only 0.015
```

FP8 E4M3 at this magnitude has a step size of ~0.03. Both values could snap to the same FP8 code, flipping which token gets selected. Softmax further amplifies errors exponentially.

The `lm_head` is a single layer (out of, say, 80 transformer layers) that runs once per generated token -- quantizing it saves negligible time but risks significant accuracy degradation.

### 11.2 Attention Internals

| Layer | Why Excluded |
|-------|-------------|
| **Softmax** | Output is a probability distribution in $[0, 1]$. FP8 has very coarse resolution here (values near 1.0 are: 0.875, 1.0, 1.25). A probability of 0.95 snaps to 1.0, distorting attention patterns. |
| **matmul_qk** (Q @ K^T) | Feeds into softmax. Errors are amplified exponentially. |
| **matmul_av** (Attention @ V) | Second input comes from KV cache with its own quantization constraints. Scale mismatches cause systematic errors. |
| **fused_sdpa** | Fused scaled-dot-product attention -- same sensitivity as above. |

### 11.3 KV Cache

Cached K/V values are reused across many generation steps. Quantization error in the cache is **permanent** -- once a K/V entry is stored with error, every subsequent token that attends to it sees that error, compounding over the sequence. In contrast, quantization error in a linear layer only affects the current token.

### 11.4 MoE Gating (`mlp.gate`)

The gate decides which expert(s) process each token. Its output is a small vector (e.g., 8 values for 8 experts), and top-k selection determines routing. A tiny FP8 rounding error can flip which experts are chosen, causing the token to be processed by entirely wrong experts -- a catastrophic error.

### 11.5 Embedding and Normalization Layers

| Layer | Why Usually Kept in BF16 |
|-------|-------------------------|
| **Embedding layers** | Lookup tables, not matmuls. No FP8 throughput benefit. |
| **RMSNorm / LayerNorm** | Element-wise ops with few parameters. Normalize the distribution that all subsequent layers depend on -- small errors propagate everywhere. |

### 11.6 General Principle

Layers avoided from quantization share these traits:
1. **Small compute cost** relative to the model (not bottlenecks)
2. **High sensitivity** -- errors propagate or amplify downstream
3. **Decision boundaries** -- output determines discrete choices (token selection, expert routing, attention distribution)

The layers that benefit most are the **large linear projections** (q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, gate_proj) which are compute-bound, large, and relatively tolerant to quantization noise.

---

## 🧊 12. MXFP4: Microscaling FP4

MXFP4 (Microscaling FP4) is an emerging format from the **MX (Microscaling) specification**, published by a consortium including Microsoft, AMD, ARM, Intel, Meta, NVIDIA, and Qualcomm. It pushes quantization below 8 bits while maintaining usable accuracy through a novel scaling architecture.

### 12.1 The FP4 Element (E2M1)

Each element is a 4-bit floating-point number:

| | Sign | Exponent | Mantissa | Total |
|---|:---:|:---:|:---:|:---:|
| FP4 (E2M1) | 1 | 2 | 1 | 4 bits |
| FP8 (E4M3) | 1 | 4 | 3 | 8 bits |

With 2 exponent bits and 1 mantissa bit, FP4 can represent only ~16 distinct values. The positive representable values are roughly:

```
0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
```

On its own, this is far too coarse for neural networks. The key to MXFP4 is the **shared exponent**.

### 12.2 The Shared Exponent: What Makes MX Different

Instead of one scale per tensor (PTS) or per channel (PCS), MXFP4 uses **one scale per small block of elements** (typically 32):

```
Tensor:  [v0, v1, v2, ... v31] [v32, v33, ... v63] [v64, ...]
          \___ block 0 ___/     \___ block 1 ___/
          shared exponent s0    shared exponent s1
```

The shared exponent is stored in **E8M0 format** -- 8 exponent bits, 0 mantissa bits, no sign bit. It can only represent **exact powers of 2**:

$$\text{block\_scale} = 2^{(\text{stored\_exponent} - 127)}$$

### 12.3 Why E8M0 (No Mantissa) for the Shared Exponent?

The shared exponent only needs to represent the **order of magnitude** of the block, not a precise value. The individual FP4 elements handle fine-grained value representation within that magnitude range.

Think of it as a two-level system:

```
Shared exponent (E8M0):  "this block's values are in the ballpark of 2^-5 = 0.03125"
Individual FP4 (E2M1):   "within that ballpark, this element is 1.5x the ballpark"
                          → actual value = 1.5 × 0.03125 = 0.046875
```

Adding mantissa bits to the shared exponent would increase overhead while providing minimal benefit: FP4 elements are already so coarse (16 distinct values) that making the "ballpark" slightly more precise is not worth the extra bits. The maximum error from the shared exponent being a power of 2 (up to 2x off) only shifts the effective FP4 grid by about one step.

### 12.4 How the Shared Exponent Is Calculated

```
Given a block of 32 elements: [v0, v1, ..., v31]

Step 1: Find max absolute value
  block_max = max(|v0|, |v1|, ..., |v31|)

Step 2: Extract its exponent (floor of log2)
  shared_exponent = floor(log2(block_max))

Step 3: Encode as E8M0
  stored_value = shared_exponent + 127    (add bias, same as FP32's convention)

Step 4: Block scale is
  block_scale = 2^shared_exponent
```

<details markdown="1">
<summary>💡 MXFP4 Worked Example</summary>

```
Block = [0.03, -0.015, 0.042, 0.008, ...]
block_max = 0.042

shared_exponent = floor(log2(0.042)) = floor(-4.57) = -5
block_scale = 2^-5 = 0.03125
stored E8M0 value = -5 + 127 = 122

Quantize each element:
  0.03   / 0.03125 = 0.96  → cast_to_fp4 → 1.0
  -0.015 / 0.03125 = -0.48 → cast_to_fp4 → -0.5
  0.042  / 0.03125 = 1.344 → cast_to_fp4 → 1.5
  0.008  / 0.03125 = 0.256 → cast_to_fp4 → 0.25

Dequantize:
  1.0  × 0.03125 = 0.03125   (original: 0.03,   error: 0.00125)
  -0.5 × 0.03125 = -0.015625 (original: -0.015, error: 0.000625)
  1.5  × 0.03125 = 0.046875  (original: 0.042,  error: 0.004875)
  0.25 × 0.03125 = 0.0078125 (original: 0.008,  error: 0.0001875)
```

</details>

### 12.5 Memory Layout and Effective Bits

```
Each block:  32 × 4-bit elements  +  1 × 8-bit shared exponent
           = 128 bits             +  8 bits
           = 136 bits total

Effective bits per element: 136 / 32 = 4.25 bits
```

Compare:

| Format | Bits per Element | Memory vs BF16 |
|--------|:---:|:---:|
| BF16 | 16 | 1x (baseline) |
| FP8 | 8 | 2x compression |
| MXFP4 | 4.25 | ~3.8x compression |

### 12.6 Why Block-Level Scaling Rescues FP4's Coarseness

Consider 32 weight values ranging from $-0.03$ to $0.05$. Without scaling, FP4's smallest representable non-zero value is $0.5$ -- everything would round to $0$. With a block scale of $0.03125$ ($2^{-5}$):

```
0.03 / 0.03125 = 0.96 → FP4: 1.0 → dequant: 0.03125  ✓
```

The shared exponent shifts FP4's representable range to match each block's local distribution. Since nearby elements in a weight matrix tend to have similar magnitudes, 32-element blocks are small enough that one shared exponent works well.

---

## 🔍 13. MXFP4 vs FP8: The Shared Exponent Difference

This is the most fundamental architectural distinction between the two formats.

### 13.1 FP8: Each Element Is Self-Contained

Every FP8 element carries its **own exponent** (4 or 5 bits). It can independently represent its magnitude:

```
FP8 E4M3 element:  [sign=0 | exp=0110 | mantissa=101]
                     1 bit    4 bits     3 bits = 8 bits

Value = 2^(6-7) × (1 + 5/8) = 2^-1 × 1.625 = 0.8125
```

No neighboring elements are involved. The external scale (per-tensor or per-channel) exists solely because the 4-bit exponent can only span a limited range. If a tensor's values fall outside that range, the external scale shifts everything into the representable window.

**The external scale is metadata about the tensor. It is NOT part of the data format.**

### 13.2 MXFP4: Elements Are Incomplete Without the Shared Exponent

Each FP4 element has only 2 exponent bits. It can only distinguish values in a tiny range (roughly $[0.5, 6.0]$). Without the shared exponent, FP4 is non-functional for real neural network values:

```
FP4 E2M1 element:  [sign=0 | exp=10 | mantissa=1]
                     1 bit    2 bits   1 bit = 4 bits

Local value = 2^(2-1) × (1 + 1/2) = 3.0

But the ACTUAL value depends on the shared exponent:
  If shared_exp = 2^-7:  actual = 3.0 × 0.0078125 = 0.0234375
  If shared_exp = 2^3:   actual = 3.0 × 8.0       = 24.0
```

**The shared exponent IS part of the data encoding. Without it, values are uninterpretable.**

### 13.3 Side-by-Side Comparison

Storing the value $0.0234375$:

```
FP8 E4M3 (self-contained):
┌─────────────────────────┐
│ 0 | 0010 | 100          │  8 bits total
│ s   exp    mantissa      │  Fully self-contained.
│ = 2^(2-7) × 1.5 = 0.047 │  (nearest representable)
└─────────────────────────┘

MXFP4 E2M1 (needs shared exponent):
┌───────────┐  ┌───────────┐
│ 01111000  │  │ 0 | 10 | 1│  4 bits + 8/32 = 4.25 bits effective
│ shared exp│  │ s  ex  man│
│ = 2^-7    │  │ = 3.0     │
└───────────┘  └───────────┘
Actual value = 3.0 × 2^-7 = 0.0234375  (exact!)
```

| Property | FP8 | MXFP4 |
|----------|-----|-------|
| Exponent per element | 4-5 bits (self-contained) | 2 bits (insufficient alone) |
| Shared exponent | None. External scale is optional metadata. | **Required.** Part of the data format. |
| Scale granularity | Per-tensor (millions share 1 scale) or per-channel | **Per-block (every 32 elements)** |
| Scale format | FP32 or BF16 (any float) | E8M0 (powers of 2 only) |
| Scale memory overhead | Negligible (~0.002%) | Structural (6.25% of data) |
| Can you omit the scale? | Yes (if values fit in FP8's native range) | No (format is non-functional without it) |
| HW reads scale | Once, at start of operation | Continuously, every 32 elements |

### 13.4 Why the Different Designs

**FP8 has enough exponent bits to be self-sufficient.** 4 exponent bits span a $2^{14} \approx 16{,}000\times$ dynamic range, covering most neural network tensors without external help.

**FP4 has too few exponent bits to stand alone.** 2 exponent bits span only a $4\times$ dynamic range. Neural network values routinely span $1000\times$ or more. The shared exponent is architecturally mandatory. Making it per-block (32 elements) rather than per-tensor is the core insight of the MX format -- it gives fine-grained dynamic range adaptation at modest overhead.

### 13.5 Why We Don't Count Scale Overhead for FP8

In FP8 with per-tensor scaling, there is **one 32-bit float** per entire tensor. For a $[4096, 4096]$ weight matrix:

```
Weight data:    4096 × 4096 × 1 byte  = 16,777,216 bytes
Scale:          1 × 4 bytes            =           4 bytes
Overhead:       4 / 16,777,216         = 0.00002%
```

Even with per-channel scaling (4096 channels): $16{,}384 / 16{,}777{,}216 = 0.098\%$. The scale is loaded once and broadcast -- architecturally negligible.

In MXFP4, the shared exponent is **woven into the data layout** -- one 8-bit exponent per 32 elements, read continuously by the hardware. It is a structural part of the format, not a side-table, which is why it is counted in the effective bits per element ($4.25$).

---

## 📚 References

- [FP8 Formats for Deep Learning (NVIDIA/ARM/Intel)](https://arxiv.org/abs/2209.05433)
- [OCP Microscaling Formats (MX) Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [Intel Neural Compressor Documentation](https://intel.github.io/neural-compressor/)
- [Habana Gaudi Documentation](https://docs.habana.ai/)
- [vLLM-Gaudi FP8 Calibration Guide](https://vllm-gaudi.readthedocs.io/en/latest/)
- [A Survey of Quantization Methods for Efficient Neural Network Inference](https://arxiv.org/abs/2103.13630)

---

*"Quantization is not about losing precision -- it is about spending precision where it matters most."*
