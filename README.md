# DSP Analysis Pipeline Prototypes (OFDM BPSK, Channelization, Detection)

This repository contains a focused set of RF digital signal processing prototypes implemented in Python. The code is organized as modular and testable building blocks that reflect how wideband RF data processing pipelines can be structured in practice. The algorithms in this repo are focused on waveform generation, filtering, energy detection, channelized outputs of detected signals, and maximum likelihood based synchronization.

## Motivation

This project is meant to explore DSP workflows with modern Python programming techniques and how the fundamentals of digital communications can be applied to build practical, modular, and reusable algorithms.

The repository prioritizes reusable library code that can later be visualized via notebooks

---
## Run The Demo
Here is a link to Google Colab that will open the most recent version of the "Signal Of Interest Pipeline Demonstration."

This notebook demonstrates how some of the source code functionality can be used together to make an end-to-end signal analysis pipeline from signal generation to filtering and visualization.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/bouchard-josiah/dsp_pipeline/blob/main/notebooks/demo.ipynb
)

## Current scope

### Implemented functionality

**Waveform generation**
- **OFDM BPSK signal generation**
  - Configurable OFDM parameters (FFT size, subcarriers, etc...)
  - Produces complex baseband IQ suitable for downstream DSP and detection work
  - Currently implemented and working as intended in all test files except the in-progress tests/test_channelizer.py

**Core DSP building blocks**
- **FIR filtering**
  - Windowed-sinc style FIR filter design and application
  - Intended for both standalone filtering and channelization workflows
  - FIR filtering functionality is currently fully implemented with intended results in tests/test_FIR_filter.py

- **Energy detection**
  - Noise floor estimation and threshold-based detection
  - Designed to operate on FFT / spectrogram-style representations
  - Currently working and implemented as intended in tests/test_energy_detect.py

**Wideband processing**
- **Channelizer**
  - Utilizes an energy detection mask to locate likely transmissions in a wideband received signal
  - Custom and configurable grouping algorithm for signal detection
  - Generates a spectrogram for a chosen sub-band of a detected signal
  - Functionality currently in testing/debug stage

### In progress

- **Channelizer**
  - Source code functionality and test file are currently in debug/testing phase

- **Maximum Likelihood (ML) detection**
  - Initial class and file scaffolding are in place
  - Intended to support likelihood-based detection / synchronization as a complement to energy detection
  - Intended to use 3GPP standard M-sequence sychronization techniques

- **Organization**
  - Dependency requirements, comments, and file headers need to be added/modified for clarity and reproducabality

---

## Project Notes

- Algorithms and source files are exercised through pytest-based test functions
- Channelizer test file is written but has not yet been executed
- The first Jupyter notebook example is **added and in progress**
- The focus so far has been on:
  - Utilizing fundamentals to build intuitive and correct algorithms
  - Keeping functionality modular and lightweight
  - Pipeline-oriented design

This repo reflects an active development state and not a finished product.

---


---

## Repository layout

```text
.
├── notebooks
│   ├── Signal_Of_Interest_demo.ipynb
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
