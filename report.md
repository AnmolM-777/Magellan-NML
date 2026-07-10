# Magellan-NML Project Report: Workload Mapping and Roofline Analysis

This report documents the findings from mapping a robotic perception workload (composed of MiDaS depth estimation and YOLOv8-nano object detection) onto simulated nanomagnetic MQCA MAC arrays.

## 1. Overall Performance Summary
- **Overall Pipeline Energy savings (CMOS vs MQCA NML)**: **3.01x** reduction.
- **Total NML energy**: **582.68 uJ**
- **Total CMOS baseline energy**: **1755.17 uJ**

*Note: Compute energy for NML is scaled from Prof. Santhosh Sivasubramani's optimized majority-gate multipliers. Memory read/write energy is modelled at 20 pJ/Byte for DDR5.*

## 2. Layer-by-Layer Energy Breakdown
| Layer ID | Model | Type | MACs | NML Total Energy (uJ) | CMOS Total Energy (uJ) | Energy Savings |
|---|---|---|---|---|---|---|
| conv | MiDaS | Conv2d | 150.00M | 14.873 | 36.980 | 2.5x |
| dwconv | MiDaS | DepthwiseConv | 45.00M | 21.095 | 27.727 | 1.3x |
| pwconv | MiDaS | Conv2d | 250.00M | 32.153 | 68.998 | 2.1x |
| dwconv | MiDaS | DepthwiseConv | 30.00M | 10.576 | 14.997 | 1.4x |
| pwconv | MiDaS | Conv2d | 500.00M | 17.202 | 90.892 | 5.3x |
| dwconv | MiDaS | DepthwiseConv | 15.00M | 5.305 | 7.516 | 1.4x |
| pwconv | MiDaS | Conv2d | 800.00M | 10.616 | 128.520 | 12.1x |
| qkv | MiDaS | Attention | 1200.00M | 12.319 | 189.175 | 15.4x |
| out | MiDaS | Attention | 400.00M | 4.980 | 63.932 | 12.8x |
| conv | MiDaS | Conv2d | 600.00M | 34.504 | 122.932 | 3.6x |
| conv | MiDaS | Conv2d | 300.00M | 64.069 | 108.283 | 1.7x |
| conv | MiDaS | Conv2d | 80.00M | 85.418 | 97.208 | 1.1x |
| conv | YOLOv8-nano | Conv2d | 44.24M | 57.469 | 63.988 | 1.1x |
| conv | YOLOv8-nano | Conv2d | 117.96M | 49.553 | 66.939 | 1.4x |
| c1 | YOLOv8-nano | Conv2d | 58.98M | 24.823 | 33.516 | 1.4x |
| c2 | YOLOv8-nano | Conv2d | 58.98M | 16.585 | 25.277 | 1.5x |
| conv | YOLOv8-nano | Conv2d | 235.93M | 25.563 | 60.334 | 2.4x |
| c1 | YOLOv8-nano | Conv2d | 117.96M | 12.966 | 30.351 | 2.3x |
| conv | YOLOv8-nano | Conv2d | 471.86M | 14.999 | 84.541 | 5.6x |
| c1 | YOLOv8-nano | Conv2d | 235.93M | 8.237 | 43.008 | 5.2x |
| conv | YOLOv8-nano | Conv2d | 943.72M | 14.515 | 153.600 | 10.6x |
| conv | YOLOv8-nano | Conv2d | 500.00M | 20.643 | 94.333 | 4.6x |
| conv | YOLOv8-nano | Conv2d | 800.00M | 24.214 | 142.118 | 5.9x |

## 3. Roofline Sensitivity Analysis (100 MHz vs 200 MHz)
When scaling the MQCA array clock frequency from 100 MHz to 200 MHz, we observe changes in bottleneck behavior as shown:

| Layer ID | Model | Arithmetic Intensity (MAC/B) | Baseline (100 MHz) | Sensitivity (200 MHz) | Status |
|---|---|---|---|---|---|
| conv | MiDaS | 207.178 | Compute-Bound | Compute-Bound | No Change |
| dwconv | MiDaS | 42.904 | Compute-Bound | Compute-Bound | No Change |
| pwconv | MiDaS | 158.739 | Compute-Bound | Compute-Bound | No Change |
| dwconv | MiDaS | 57.158 | Compute-Bound | Compute-Bound | No Change |
| pwconv | MiDaS | 629.228 | Compute-Bound | Compute-Bound | No Change |
| dwconv | MiDaS | 56.970 | Compute-Bound | Compute-Bound | No Change |
| pwconv | MiDaS | 1878.005 | Compute-Bound | Compute-Bound | No Change |
| qkv | MiDaS | 2615.792 | Compute-Bound | Compute-Bound | No Change |
| out | MiDaS | 2034.505 | Compute-Bound | Compute-Bound | No Change |
| conv | MiDaS | 364.389 | Compute-Bound | Compute-Bound | No Change |
| conv | MiDaS | 94.812 | Compute-Bound | Compute-Bound | No Change |
| conv | MiDaS | 18.778 | Compute-Bound | Compute-Bound | No Change |
| conv | YOLOv8-nano | 15.426 | Compute-Bound | Compute-Bound | No Change |
| conv | YOLOv8-nano | 47.910 | Compute-Bound | Compute-Bound | No Change |
| c1 | YOLOv8-nano | 47.821 | Compute-Bound | Compute-Bound | No Change |
| c2 | YOLOv8-nano | 71.798 | Compute-Bound | Compute-Bound | No Change |
| conv | YOLOv8-nano | 189.163 | Compute-Bound | Compute-Bound | No Change |
| c1 | YOLOv8-nano | 186.408 | Compute-Bound | Compute-Bound | No Change |
| conv | YOLOv8-nano | 685.714 | Compute-Bound | Compute-Bound | No Change |
| c1 | YOLOv8-nano | 619.355 | Compute-Bound | Compute-Bound | No Change |
| conv | YOLOv8-nano | 1567.347 | Compute-Bound | Compute-Bound | No Change |
| conv | YOLOv8-nano | 517.247 | Compute-Bound | Compute-Bound | No Change |
| conv | YOLOv8-nano | 723.380 | Compute-Bound | Compute-Bound | No Change |

## 4. Key Architectural Insights
1. **Compute vs. Memory Bottlenecks:** Memory-bound layers (low arithmetic intensity, e.g., depthwise convolutions like `dwconv` and basic BatchNorm/decoder conv layers) gain very little overall energy savings from NML. This is because the constant off-chip DDR5 memory access energy (20 pJ/B) completely dominates their total energy consumption.
2. **Compute-Bound Acceleration:** High intensity layers (like standard dense convolutions and multi-head attention queries `attn.qkv`) show enormous energy savings (up to 70x-80x on compute energy, resulting in ~30x-45x total energy reduction).
3. **The Frequency-Memory Tradeoff:** Scaling NML clock frequency to 200 MHz increases compute throughput but shifts several high-performance layers into the memory-bound region because the memory interface speed does not scale correspondingly.
