"""Visualize the Phase 2 OFDM loopback."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from src.core.system.phase2.phase2_simulator import Phase2Simulator
from src.exception.exception import CustomException

# Define function to plot the resource grid
def plot_resource_grid(grid: np.ndarray, output_directory: Path = "results") -> None:
  # Plot the magnitude of the resource grid
  plt.figure(figsize=(11, 5))
  plt.imshow(np.abs(grid), aspect="auto", origin="lower")
  plt.xlabel("FFT bin")
  plt.ylabel("OFDM symbol")
  plt.title("LTE-Inspired OFDM Resource Grid Magnitude")
  plt.colorbar(label="Magnitude")
  plt.tight_layout()

  # Save the plot to the output directory
  plt.savefig(output_directory / "phase2_resource_grid.png", dpi=300)
  plt.show()

# Define function to plot the waveform
def plot_waveform(waveform: np.ndarray, output_directory: Path = "results") -> None:
  # Determine the number of samples to plot (up to 500)
  sample_count = min(500, waveform.size)
  sample_indices = np.arange(sample_count)

  # Plot the real and imaginary components of the waveform
  plt.figure(figsize=(11, 5))
  plt.plot(sample_indices, waveform[:sample_count].real, label="Real component")
  plt.plot(sample_indices, waveform[:sample_count].imag, label="Imaginary component")
  plt.xlabel("Sample index")
  plt.ylabel("Amplitude")
  plt.title("Time-Domain OFDM Waveform")
  plt.grid(True)
  plt.legend()
  plt.tight_layout()

  # Save the plot to the output directory
  plt.savefig(output_directory / "phase2_ofdm_waveform.png", dpi=300)
  plt.show()

# Define function to plot the spectrum of the waveform
def plot_spectrum(waveform: np.ndarray, sample_rate_hz: float, output_directory: Path = "results") -> None:
  # Compute the FFT of the waveform and shift it to center the zero frequency component
  spectrum = np.fft.fftshift(np.fft.fft(waveform, norm="ortho"))

  # Compute the corresponding frequency bins in Hz
  frequencies_hz = np.fft.fftshift(np.fft.fftfreq(waveform.size, d=1.0 / sample_rate_hz))

  # Compute the magnitude in dB, ensuring no log of zero by using a small epsilon
  magnitude_db = 20.0 * np.log10(np.maximum(np.abs(spectrum), 1e-12))

  # Plot the spectrum
  plt.figure(figsize=(11, 5))
  plt.plot(frequencies_hz / 1e6, magnitude_db)
  plt.xlabel("Frequency (MHz)")
  plt.ylabel("Magnitude (dB)")
  plt.title("OFDM Waveform Spectrum")
  plt.grid(True)
  plt.tight_layout()

  # Save the plot to the output directory
  plt.savefig(output_directory / "phase2_ofdm_spectrum.png", dpi=300)
  plt.show()

# Define function to plot the symbol reconstruction
def plot_symbol_reconstruction(transmitted_symbols: np.ndarray, recovered_symbols: np.ndarray, output_directory: Path = "results") -> None:
  # Determine the number of points to plot (up to 500)
  point_count = min(500, transmitted_symbols.size)

  # Create a scatter plot of the transmitted and recovered symbols
  plt.figure(figsize=(7, 7))
  plt.scatter(transmitted_symbols[:point_count].real, transmitted_symbols[:point_count].imag, marker="x", s=100, label="Transmitted symbols")
  plt.scatter(
    recovered_symbols[:point_count].real,
    recovered_symbols[:point_count].imag,
    s=20,
    alpha=0.5,
    label="Recovered symbols",
  )
  plt.axhline(0.0)
  plt.axvline(0.0)
  plt.xlabel("In-phase component")
  plt.ylabel("Quadrature component")
  plt.title("Noiseless OFDM Symbol Reconstruction")
  plt.axis("equal")
  plt.grid(True)
  plt.legend()
  plt.tight_layout()

  plt.savefig(output_directory / "phase2_symbol_reconstruction.png", dpi=300)
  plt.show()


def plot_and_run() -> None:
  try:
    # Run the Phase 2 OFDM loopback simulation and visualize the results
    simulator = Phase2Simulator()
    result = simulator.run(random_seed=42)
    config = simulator.config

    # Create the output directory if it doesn't exist
    output_directory = Path("results")
    output_directory.mkdir(exist_ok=True)

    # Print the simulation results
    print("Phase 2 OFDM Loopback and Simulation Results")
    print("-" * 40)
    print(f"FFT size                  : {config.fft_size}")
    print(f"Active subcarriers        : {config.active_subcarriers}")
    print(f"OFDM symbols/subframe     : {config.symbols_per_subframe}")
    print(f"QPSK symbols/subframe     : {config.qpsk_symbols_per_subframe}")
    print(f"BER                       : {result.ber:.6e}")
    print(f"Maximum grid error        : {result.maximum_grid_error:.6e}")
    print(f"Waveform length           : {result.waveform_length}")

    # Plot the resource grid, waveform, spectrum, and symbol reconstruction
    plot_resource_grid(result.transmitted_grid, output_directory)
    plot_waveform(result.transmitted_waveform, output_directory)
    plot_spectrum(result.transmitted_waveform, simulator.config.sample_rate_hz, output_directory)
    plot_symbol_reconstruction(result.transmitted_symbols, result.recovered_symbols, output_directory)
  except Exception as e:
    raise CustomException(e, sys)