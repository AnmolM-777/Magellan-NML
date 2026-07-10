# Magellan-NML: Roofline-Guided Workload Mapping of Robotic Perception onto Nanomagnetic MAC Arrays

This repository houses the simulation suite and research codebase for mapping robotic perception pipelines onto beyond-CMOS nanomagnetic Majority Gate (MQCA) MAC arrays. It implements an analytical simulation layer, a workload profiler for depth estimation and object detection pipelines, and a roofline-based bottleneck analysis tool.

---

## 1. Project Proposal & Background

### Project Overview
This project builds the first roofline characterisation of a real robotic perception pipeline mapped onto nanomagnetic MAC arrays. It sits at the intersection of three research threads:
1. **Prof. Santhosh Sivasubramani's** published MQCA convolution and arithmetic primitives.
2. Standard ML accelerator workload profiling methodology.
3. A concrete robotic perception workload (depth estimation + object detection).

The output is a layer-level energy breakdown and roofline plot that identifies exactly which parts of the pipeline benefit most from beyond-CMOS nanomagnetic compute — and which remain memory-bound regardless.

### Target Alignment
* **Lab:** INTRINSIC Lab, IIT Delhi (Prof. Santhosh Sivasubramani)
* **Author:** Anmol Mishra, B.Tech CSE, IIT Jodhpur

---

## 2. Research Core & Reference Parameters

### Nanomagnetic Logic (NML) / MQCA Multiplier Model
Calibrated against Sivasubramani et al. (*Nano Express 2021*): **"Area efficient in-plane nanomagnetic multiplier and convolution architecture design."**

Key physical and layout parameters:
* **Switching Energy ($E_{switch}$):** $2.0 \text{ aJ}$ per nanomagnet at room temperature ($2.0 \times 10^{-18} \text{ Joules}$).
* **Switching Activity Factor ($\alpha$):** 0.5 (average percentage of magnets switching state per cycle).
* **2-Bit Multiplier Baseline:** 120 magnets, 12 majority gates, 3.0 clock cycles latency.
* **1-Bit Full Adder Baseline:** 35 magnets, 4 majority gates, 1.5 clock cycles latency.
* **DRAM Memory Interface:** DDR5 interface running at $51.2 \text{ GB/s}$ bandwidth with a power footprint of $20.0 \text{ pJ/Byte}$.

### Mathematical Scaling Formulations
To simulate arbitrary integer bit-widths (e.g., INT4, INT8), the multiplier and adder logic scale as follows:

1. **Multiplier Magnets:**
   $$M_{mult}(N) = 30 \times N^2$$
2. **Multiplier Latency (cycles):**
   $$L_{mult}(N) = 1.5 \times N$$
3. **Adder Magnets:**
   $$M_{add}(N) = 35 \times N$$
4. **Adder Latency (cycles):**
   $$L_{add}(N) = 0.5 \times N + 1.0$$
5. **MAC Unit Resources ($N$-bit Multiply-Accumulate):**
   $$M_{mac}(N) = M_{mult}(N) + M_{add}(2N + 4)$$
   $$L_{mac}(N) = L_{mult}(N) + L_{add}(2N + 4)$$
   $$E_{mac}(N) = M_{mac}(N) \times \alpha \times E_{switch}$$

---

## 3. Workload Description
The robotic perception workload simulates a dual-task pipeline:
1. **MiDaS (Depth Estimation):** Composed of MobileNet-based encoder backbones, multi-head attention queries (transformer bottlenecks), and fusion decoders (~10 GFLOPs total).
2. **YOLOv8-nano (Object Detection):** Composed of CSPDarknet convolutions, C2f modules, and decoupled classification/bounding-box heads (~8.7 GFLOPs total).

---

## 4. Codebase Architecture

The project consists of the following modular Python scripts:
* **[simulator.py](file:///Users/akmishra/.gemini/antigravity/scratch/Magellan-NML/simulator.py):** Implements the analytical MQCA resource modeling equations and the 28nm CMOS MAC baseline reference energy values (calibrated from Mark Horowitz's energy scaling data).
* **[profiler.py](file:///Users/akmishra/.gemini/antigravity/scratch/Magellan-NML/profiler.py):** Extracts workload specifications layer-by-layer (MACs, input/output tensors, weights, and memory bytes) for MiDaS and YOLOv8-nano.
* **[mapper.py](file:///Users/akmishra/.gemini/antigravity/scratch/Magellan-NML/mapper.py):** Computes per-layer performance statistics (total energy, latency, execution time) when mapped onto parallel processing arrays.
* **[roofline.py](file:///Users/akmishra/.gemini/antigravity/scratch/Magellan-NML/roofline.py):** Calculates platform roofline boundaries and plots performance visualization figures.
* **[main.py](file:///Users/akmishra/.gemini/antigravity/scratch/Magellan-NML/main.py):** The master orchestrator script executing the end-to-end simulation flow and sensitivity analysis.
* **[test_simulation.py](file:///Users/akmishra/.gemini/antigravity/scratch/Magellan-NML/test_simulation.py):** Unit test suite verifying simulator scaling correctness.

---

## 5. Getting Started & Running

### Requirements
Install dependencies using pip:
```bash
pip install -r requirements.txt
```

### Execution
Run the end-to-end pipeline:
```bash
python3 main.py
```
This runs the simulation, outputs detailed terminal report tables, creates the `plots/` directory with visualization graphs, and writes `report.md`.

### Running Unit Tests
Validate that the simulation models are correct:
```bash
pytest test_simulation.py
```

---

## 6. References
1. S. Sivasubramani et al., **"Area efficient in-plane nanomagnetic multiplier and convolution architecture design,"** *Nano Express*, vol. 2, no. 1, 2021.
2. J. Bandyopadhyay et al., **"Energy dissipation in nanomagnetic logic gates under realistic parameters,"** *IEEE Transactions on Nanotechnology*, 2019.
3. M. Horowitz, **"Computing's Energy Problem (and What We Can Do About It),"** *IEEE International Solid-State Circuits Conference (ISSCC)*, 2014.
