'''
Constellation module for Phase 2 of the digital communications system.
This module provides functionality to define and manipulate constellations for modulation schemes.
'''
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np

from src.core.system.phase2.phase2_simulator import Phase2Simulator
from src.exception.exception import CustomException

# Define function to plot the constellation of the recovered symbols
if __name__ == "__main__":
  try:
    # Run the Phase 2 OFDM loopback simulation and visualization
    simulator = Phase2Simulator()
    result = simulator.run(ebn0_db=6.0, random_seed=42)

    # Run a batch of simulations to compute the BER over multiple subframes
    batch_result = (simulator.run_ber_experiment(
      ebn0_db=6.0,
      number_of_subframes=500,
      random_seed=42,
      )
    )

    # Define the output directory for saving the constellation plot
    output_directory = Path("results/phase_2")
    output_directory.mkdir(exist_ok=True)

    # Define the ideal QPSK constellation points
    ideal_points = np.array([
        (1 + 1j) / np.sqrt(2.0),
        (1 - 1j) / np.sqrt(2.0),
        (-1 + 1j) / np.sqrt(2.0),
        (-1 - 1j) / np.sqrt(2.0),
      ],
    dtype=np.complex128,
  )

    # Create a scatter plot of the recovered symbols and ideal QPSK points
    plt.figure(figsize=(7, 7))
    plt.scatter(
      result.recovered_symbols.real,
      result.recovered_symbols.imag,
      s=12,
      alpha=0.4,
      label="Recovered OFDM symbols",
    )
    plt.scatter(
      ideal_points.real,
      ideal_points.imag,
      marker="x",
      s=150,
      linewidths=3,
      label="Ideal QPSK points",
    )
    plt.axhline(0.0)
    plt.axvline(0.0)
    plt.xlabel("In-phase component")
    plt.ylabel("Quadrature component")
    plt.title("QPSK-OFDM Constellation over AWGN\n" r"$E_b/N_0=6$ dB")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save the constellation plot to the output directory
    output_path = (output_directory / "phase2b_ofdm_constellation.png")
    plt.savefig(output_path, dpi=200)
    plt.show()
  except Exception as e:
    raise CustomException(e, sys)