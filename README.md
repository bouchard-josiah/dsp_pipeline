# RF DSP Pipeline Prototypes (OFDM BPSK, Channelization, Detection)

This repository contains a focused set of RF digital signal processing prototypes implemented in Python. The code is organized as modular, testable building blocks that reflect how wideband RF data processing pipelines can be structured in practice: from waveform generation, through filtering and detection, toward channelized outputs of detected channels and maximum likelihood based synchronization.

## Motivation

This project is meant to explore DSP workflows with modern Python programming techniques and how the fundamentals of digital communications can be applied to build practical, modular, and reusable algorithms.

The repository prioritizes reusable library code that can later be visualized via notebooks

---

## Current scope

### Implemented functionality

**Waveform generation**
- **OFDM BPSK signal generation**
  - Configurable OFDM parameters (FFT size, subcarriers, etc...)
  - Produces complex baseband IQ suitable for downstream DSP and detection work

**Core DSP building blocks**
- **FIR filtering**
  - Windowed-sinc style FIR filter design and application
  - Intended for both standalone filtering and channelization workflows

- **Energy detection**
  - Noise floor estimation and threshold-based detection
  - Designed to operate on FFT / spectrogram-style representations

**Wideband processing**
- **Channelizer**
  - Utilizes an energy detection mask to locate likely transmissions in a wideband received signal
  - Custom and configurable grouping algorithm for signal detection
  - Generates a spectrogram for a chosen sub-band of a detected signal

### In progress

- **Maximum Likelihood (ML) detection**
  - Initial class and file scaffolding are in place
  - Intended to support likelihood-based detection / synchronization as a complement to energy detection
  - Intended to use 3GPP standard M-sequence sychronization techniques
  - Dependency requirements, comments, and file headers need to be added for clarity and reproducabality

---

## Project Notes

- Algorithms and source files are exercised through pytest-based test functions
- Some test files (e.g., channelizer and FIR filtering tests) are written but have not yet been executed
- Jupyter notebooks for visualization are **planned but not yet added**
- The focus so far has been on:
  - Utilizing fundamentals to build intuitive and correct algorithms
  - Keeping functionality modular and lightweight
  - Pipeline-oriented design

This repo reflects an active development state and not a finished product.

---


---

## Repository layout

'''text
.
├── notebooks
├── pyproject.toml
├── README.md
├── results  
├── scripts
├── src
│   ├── dsp_demo
│   │   ├── Channelizer.py
│   │   ├── Energy_Detection.py
│   │   ├── FIR_Filter.py
│   │   ├── __init__.py
│   │   ├── Max_Likely_Detect.py
│   │   ├── OFDM_Gen.py
│   │   └── __pycache__
├── tests
│   ├── __init__.py
│   ├── __pycache__
│   ├── test_channelizer.py
│   ├── test_energy_detect.py
│   └── test_OFDM_gen.py
