# Digital Communications Simulator

`digital-comms` is a Python implementation of a digital communications
physical layer. It builds the link in stages, beginning with uncoded QPSK over
additive white Gaussian noise (AWGN) and progressing to coded OFDM over a
static multipath channel. The project also trains and evaluates a neural
decoder against a conventional soft-input Viterbi decoder.

The repository is intended for experimentation and learning: each phase has a
small runner or plotting script, while the reusable signal-processing blocks
live under `src/core`.

## System overview

The complete simulated link is:

```text
text / random bits
        |
        v
convolutional encoder (rate 1/2) -> block interleaver -> QPSK mapper
        |
        v
OFDM resource grid -> IFFT + cyclic prefix
        |
        v
static multipath channel -> AWGN channel
        |
        v
remove cyclic prefix + FFT -> perfect-CSI equalizer -> QPSK demapper
        |
        v
hard Viterbi / soft Viterbi / neural decoder
        |
        v
recovered bits or text + BER, EVM, success-rate, and latency metrics
```

The default OFDM configuration is inspired by the LTE 1.4 MHz physical layer:

| Parameter | Default |
| --- | ---: |
| FFT size | 128 |
| Resource blocks | 6 |
| Active subcarriers | 72 |
| OFDM symbols per subframe | 14 |
| Subcarrier spacing | 15 kHz |
| Sample rate | 1.92 MHz |
| QPSK symbols per subframe | 1,008 |
| Coded bits per subframe | 2,016 |
| Cyclic prefix | 10 samples for the long CP, 9 for the short CP |

## Development phases

### Phase 1: QPSK over AWGN

- Generates random information bits.
- Maps pairs of bits to normalized QPSK symbols.
- Adds complex AWGN for a selected $E_b/N_0$.
- Hard-demaps the received symbols.
- Measures bit error rate (BER), error-vector magnitude (EVM), symbol energy,
  and noise variance.
- Includes constellation and simulated-versus-theoretical BER plots.

### Phase 2: QPSK-OFDM over AWGN

- Maps QPSK symbols onto the 72 active subcarriers of a 128-bin grid.
- Performs OFDM modulation and demodulation with cyclic prefixes.
- Adds noise in the time domain using waveform bit energy.
- Produces resource-grid, waveform, spectrum, constellation, and BER results.
- Accounts for cyclic-prefix overhead when comparing with theoretical BER.

### Phase 3: Static multipath and equalization

- Passes the OFDM waveform through a deterministic three-path channel.
- Checks whether the cyclic prefix covers the channel delay spread.
- Uses the known channel response as perfect channel-state information.
- Compares no equalization, zero-forcing (ZF), and minimum mean-square error
  (MMSE) equalization.
- Reports BER and EVM and can plot the impulse response, frequency response,
  constellations, and equalizer BER curves.

### Phase 4: Channel coding

- Adds a rate-1/2, constraint-length-3 convolutional encoder with trellis
  termination.
- Implements hard-input and soft-input Viterbi decoders.
- Adds deterministic block interleaving for the OFDM link.
- Compares pre-decoder BER with hard- and soft-decoded payload BER.
- Covers coding loopback, coded QPSK/AWGN, and the complete coded
  QPSK-OFDM/multipath/AWGN chain.

### Neural decoding and text evaluation

- Generates convolutionally coded log-likelihood-ratio (LLR) training data
  across a range of $E_b/N_0$ values.
- Trains a sequence-to-sequence 1D convolutional network that receives two
  coded-bit LLRs per trellis step and predicts one information-bit logit.
- Wraps saved Keras models behind the same decoding-style interface used by
  the classical receiver.
- Compares neural decoding with soft Viterbi by BER and decoding latency.
- Supports Monte Carlo transmission of framed UTF-8 text, including exact-text
  success rate and frame error rate.

## Repository layout

```text
.
|-- main.py                         # Default Phase 2 demonstration
|-- pyproject.toml                  # Dependencies and tool configuration
|-- uv.lock                         # Reproducible dependency lockfile
|-- src/
|   |-- core/
|   |   |-- channel/                # AWGN and static multipath models
|   |   |-- coding/                 # Encoder, interleaver, Viterbi decoders
|   |   |-- metrics/                # BER, EVM, and theoretical curves
|   |   |-- modulation/             # QPSK mapping and hard/soft demapping
|   |   |-- ofdm/                   # Grid mapping, OFDM modulation, config
|   |   |-- receiver/               # ZF and MMSE equalizers
|   |   |-- source/                 # Text codec and bit framing
|   |   `-- system/phase1..phase4/  # End-to-end experiments and plots
|   |-- ml/
|   |   |-- datasets/               # Synthetic coded-LLR data generation
|   |   |-- models/                 # 1D-CNN and inference service
|   |   |-- training/               # Neural-decoder training entry point
|   |   `-- evaluation/             # BER, latency, and text comparisons
|   |-- exception/                  # Project exception wrapper
|   `-- logging/                    # Loguru configuration
`-- results/                        # Generated plots and trained models
```

`results/` is created by experiment scripts as needed and is not required in a
fresh checkout.

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/) for the documented installation commands
- Enough memory and disk space for TensorFlow if neural training is used

The principal dependencies are NumPy, SciPy, Matplotlib, TensorFlow, Loguru,
and PySide6. Pytest and Ruff are included in the development dependency group.

## Installation

Clone the repository, enter its directory, and create the locked environment:

```powershell
uv sync
```

Run commands from the repository root so that `src` imports and relative
`results/` paths resolve correctly.

## Quick start

The default entry point runs a single Phase 2 QPSK-OFDM/AWGN frame and a
500-subframe BER experiment at 6 dB:

```powershell
uv run python main.py
```

It prints the OFDM dimensions, waveform energy, noise information, BER, and
frequency-domain EVM.

## Experiments

All examples below should be launched from the repository root.

### OFDM and channel experiments

```powershell
# Phase 2: constellation and BER curve
uv run python -m src.core.system.phase2.phase2_constellation
uv run python -m src.core.system.phase2.phase2_ber_curve

# Phase 3: noiseless multipath, then noisy ZF/MMSE comparison and BER curve
uv run python -m src.core.system.phase3.run_phase3a
uv run python -m src.core.system.phase3.run_phase3b

# Phase 3 channel and constellation plots
uv run python -m src.core.system.phase3.phase3_plots
```

Phase 1 plotting functions are callable directly:

```powershell
uv run python -c "from src.core.system.phase1.phase1_constellation import plot_constellation; plot_constellation()"
uv run python -c "from src.core.system.phase1.phase1_ber_curve import run_ber_sweep; run_ber_sweep()"
```

### Coding experiments

```powershell
# Encoder-to-decoder loopback
uv run python -m src.core.system.phase4.phase4a_coding_loopback

# Coded QPSK over AWGN
uv run python -m src.core.system.phase4.phase4b_runner
uv run python -m src.core.system.phase4.phase4b_ber_curve

# Coded QPSK-OFDM over static multipath and AWGN
uv run python -m src.core.system.phase4.phase4c_runner
uv run python -m src.core.system.phase4.phase4c_ber_curve
```

The BER sweeps run many frames and may take substantially longer than the
single-frame runners.

### Train the neural decoder

```powershell
uv run python -m src.ml.training.train_neural_decoder
```

The training configuration currently uses:

- 1,006 information bits per frame plus two termination bits
- batches of 32 frames
- training SNR sampled from -2 dB through 8 dB
- up to 30 epochs with early stopping and learning-rate reduction
- 100 training and 20 validation steps per epoch

The best validation checkpoint and final model are written to:

```text
results/models/best_decoder.keras
results/models/final_decoder.keras
```

### Compare Viterbi and neural decoding

Train the model first, then run either evaluation:

```powershell
# Random-bit BER and decoder-latency comparison
uv run python -m src.ml.evaluation.compare_decoders

# Interactive Monte Carlo text-payload comparison
uv run python -m src.ml.evaluation.runner
```

The text runner prompts for a message and evaluates it from 3 dB through 8 dB
over 200 trials per SNR point. It reports where each decoder first reaches a
90% exact-text success rate.

## Generated results

Depending on the experiment, plots are saved below:

```text
results/
results/phase_2/
results/phase_3/
results/phase_4/
results/phase_5/
```

Matplotlib scripts also call `plt.show()`, so they normally open interactive
plot windows in a desktop environment. Use a non-interactive Matplotlib
backend when running on a headless machine.

## Using the simulators from Python

The system components can also be composed programmatically:

```python
from src.core.system.phase4.phase4c_coded_ofdm import Phase4CSimulator

simulator = Phase4CSimulator()
result = simulator.run(ebn0_db=10.0, random_seed=42)

print("Pre-decoder BER:", result.pre_decoder_ber)
print("Hard Viterbi BER:", result.hard_payload_ber)
print("Soft Viterbi BER:", result.soft_payload_ber)
```

Configuration objects and simulation result objects use dataclasses, making
parameters explicit and results easy to inspect without parsing console output.

## Metrics and assumptions

- **BER** is the fraction of transmitted bits recovered incorrectly.
- **EVM** measures RMS symbol error relative to the transmitted constellation.
- **LLRs** carry soft evidence for coded bits into the soft Viterbi and neural
  decoders.
- Random seeds are exposed by the runners to make experiments reproducible.
- The multipath receiver assumes perfect channel-state information; channel
  estimation, synchronization, carrier-frequency offset, and timing recovery
  are outside the current model.
- The channel is simulated in baseband. This project does not control radio
  hardware or transmit over the air.

## Tests and code quality

The project declares Pytest and Ruff as development tools:

```powershell
uv run pytest
uv run ruff check .
```

Pytest is configured to search `src/tests`. The repository currently contains
no committed test suite, so correctness is primarily exercised through the
phase runners, assertions, and numerical comparisons.

## Current project status

This is an evolving educational and experimental codebase rather than a
stable published library. In particular:

- `main.py` demonstrates Phase 2 even though later phases are available as
  module entry points.
- Neural evaluation requires a locally trained Keras model.
- Some experiments have fixed simulation sizes and SNR ranges in their runner
  files rather than command-line options.
- The current wheel configuration packages `src/core`; running from the
  repository through `uv run` is the supported workflow for the complete
  project, including `src/ml`, logging, and exception modules.

## License

No license file is currently included. Unless a license is added, reuse and
redistribution rights are not granted by default.
