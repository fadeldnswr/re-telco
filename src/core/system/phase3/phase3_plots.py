'''
Generate plots for Phase 3 of the digital communications system simulation.
This module provides functions to visualize the results of the Phase 3 simulations,
'''
"""
Plots for Phase 3 static multipath experiments.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from src.core.system.phase3.phase3_simulator import EqualizerType, Phase3Simulator
from src.exception.exception import CustomException

# Define type alias for complex number array
ComplexArray = NDArray[np.complex128]

# Define function to plot channel response
def plot_channel_response(impulse_response: ComplexArray, frequency_response_shifted: ComplexArray, sample_rate_hz: float, output_directory: Path) -> None:
  # Create output directory if it does not exist
  output_directory.mkdir(parents=True, exist_ok=True)

  # Convert impulse response and frequency response to numpy arrays
  impulse_response = np.asarray(impulse_response, dtype=np.complex128)

  # Convert frequency response to numpy array
  frequency_response_shifted = np.asarray(frequency_response_shifted, dtype=np.complex128)

  # Generate impulse response plot
  delays = np.arange(impulse_response.size)

  # Plot impulse response
  plt.figure(figsize=(9, 5))
  plt.stem(delays,np.abs(impulse_response))
  plt.xlabel("Delay (samples)")
  plt.ylabel("Path magnitude")
  plt.title("Static Multipath Channel Impulse Response")
  plt.grid(True)
  plt.tight_layout()

  # Save the impulse response plot to the output directory
  plt.savefig(output_directory / "phase3_channel_impulse_response.png", dpi=300)
  plt.show()

  # Generate frequency response plot
  frequencies_hz = np.fft.fftshift(
    np.fft.fftfreq(
      frequency_response_shifted.size,
      d=1.0 / sample_rate_hz,
    )
  )

  # Calculate magnitude in dB and unwrapped phase in radians
  magnitude_db = 20.0 * np.log10(
    np.maximum(
      np.abs(frequency_response_shifted),
      1e-12,
    )
  )

  # Unwrap the phase to avoid discontinuities
  phase_rad = np.unwrap(
    np.angle(
      frequency_response_shifted
    )
  )

  # Plot frequency response magnitude and phase
  plt.figure(figsize=(10, 5))
  plt.plot(frequencies_hz / 1e6, magnitude_db)
  plt.xlabel("Frequency (MHz)")
  plt.ylabel("Magnitude (dB)")
  plt.title("Static Multipath Channel Frequency Response")
  plt.grid(True)
  plt.tight_layout()

  plt.savefig(output_directory / "phase3_channel_magnitude_response.png", dpi=300)
  plt.show()

  # Plot phase response
  plt.figure(figsize=(10, 5))
  plt.plot(frequencies_hz / 1e6, phase_rad)
  plt.xlabel("Frequency (MHz)")
  plt.ylabel("Unwrapped phase (radians)")
  plt.title("Static Multipath Channel Phase Response")
  plt.grid(True)
  plt.tight_layout()

  plt.savefig(output_directory / "phase3_channel_phase_response.png", dpi=300)
  plt.show()

# Define function for ideal constellation points
def _ideal_constellation_points() -> ComplexArray:
  return np.array(
    [
      (1 + 1j) / np.sqrt(2.0),
      (1 - 1j) / np.sqrt(2.0),
      (-1 + 1j) / np.sqrt(2.0),
      (-1 - 1j) / np.sqrt(2.0),
    ],
    dtype=np.complex128
  )

# Define function to plot constellation diagram
def plot_constellation(symbols: ComplexArray, title: str, filename: str, output_directory: Path, maximum_points: int = 3_000) -> None:
  # Create output directory if it does not exist
  output_directory.mkdir(parents=True, exist_ok=True)

  # Convert symbols to numpy array of complex numbers
  symbols = np.asarray(symbols, dtype=np.complex128)
  display_symbols = symbols[: min(maximum_points, symbols.size)]
  ideal_points = _ideal_constellation_points()

  # Plot received symbols and ideal QPSK points
  plt.figure(figsize=(7, 7))
  plt.scatter(
    display_symbols.real,
    display_symbols.imag,
    s=12,
    alpha=0.35,
    label="Received symbols",
  )
  plt.scatter(
    ideal_points.real,
    ideal_points.imag,
    marker="x",
    s=160,
    linewidths=3,
    label="Ideal QPSK points",
  )
  plt.axhline(0.0)
  plt.axvline(0.0)
  plt.xlabel("In-phase component")
  plt.ylabel("Quadrature component")
  plt.title(title)
  plt.axis("equal")
  plt.grid(True)
  plt.legend()
  plt.tight_layout()

  plt.savefig(output_directory / filename, dpi=300)
  plt.show()

def plot_phase3_constellations(unequalized_symbols: ComplexArray, zf_symbols: ComplexArray, mmse_symbols: ComplexArray, ebn0_db: float, output_directory: Path) -> None:
  # Plot constellation for unequalized symbols
  plot_constellation(
    symbols=unequalized_symbols,
    title=(
      "QPSK-OFDM after Static Multipath\n"
      f"No Equalization, "
      rf"$E_b/N_0={ebn0_db:.1f}$ dB"
    ),
    filename=(
      "phase3_constellation_no_equalizer.png"
    ),
    output_directory=output_directory,
  )

  # Plot constellation for ZF equalized symbols
  plot_constellation(
    symbols=zf_symbols,
    title=(
      "QPSK-OFDM after Static Multipath\n"
      f"Perfect-CSI ZF, "
      rf"$E_b/N_0={ebn0_db:.1f}$ dB"
    ),
    filename="phase3_constellation_zf.png",
    output_directory=output_directory,
  )

  # Plot constellation for MMSE equalized symbols
  plot_constellation(
    symbols=mmse_symbols,
    title=(
      "QPSK-OFDM after Static Multipath\n"
      f"Perfect-CSI MMSE, "
      rf"$E_b/N_0={ebn0_db:.1f}$ dB"
    ),
    filename="phase3_constellation_mmse.png",
    output_directory=output_directory,
  )

# Define main function to run the Phase 3 plots
def main() -> None:
  try:
    ebn0_db = 10.0
    random_seed = 42
    simulator = Phase3Simulator()
    
    # Define common arguments for the simulator run
    common_arguments = {
      "ebn0_db": ebn0_db,
      "random_seed": random_seed,
    }

    # Run the same channel realization with no equalization.
    no_equalizer_result = simulator.run(
      **common_arguments,
      equalizer_type=EqualizerType.NONE,
    )

    # Run the same channel realization with ZF.
    zf_result = simulator.run(
      **common_arguments,
      equalizer_type=EqualizerType.ZF,
    )

    # Run the same channel realization with MMSE.
    mmse_result = simulator.run(
      **common_arguments,
      equalizer_type=EqualizerType.MMSE,
    )

    # Define output directory for saving the plots
    output_directory = Path("results/phase_3")

    # Plot channel impulse, magnitude, and phase responses.
    plot_channel_response(
      impulse_response=zf_result.impulse_response,
      frequency_response_shifted=(
        zf_result.channel_frequency_response
      ),
      sample_rate_hz=simulator.config.sample_rate_hz,
      output_directory=output_directory,
    )

    # Plot unequalized, ZF, and MMSE constellations.
    plot_phase3_constellations(
      unequalized_symbols=(
        no_equalizer_result.unequalized_symbols
      ),
      zf_symbols=zf_result.equalized_symbols,
      mmse_symbols=mmse_result.equalized_symbols,
      ebn0_db=ebn0_db,
      output_directory=output_directory,
    )
    # Print the output directory where the plots are saved
    print("Plots saved to:", output_directory)
  except Exception as e:
    raise CustomException(e, sys)

# Run the main function if this script is executed directly
if __name__ == "__main__":
  main()