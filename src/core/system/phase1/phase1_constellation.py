"""
Plot the transmitted and received QPSK constellations.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sys

from src.core.system.config import SimulationConfig
from src.core.system.phase1.phase1_simulator import Phase1Simulator
from src.exception.exception import CustomException

# Define function to plot the QPSK constellation
def plot_constellation() -> None:
  try:
    # Define simulation configuration for constellation plotting
    config = SimulationConfig(
      ebn0_db=6.0,
      number_of_bits=20_000,
      random_seed=42,
    )

    # Initialize the Phase 1 simulator and run the simulation
    simulator = Phase1Simulator()
    result = simulator.run(config)

    # Determine the number of points to plot (limit to 2000 for clarity)
    number_of_points = min(
      2_000,
      result.received_symbols.size,
    )

    # Extract the transmitted and received symbols for plotting
    tx_symbols = result.transmitted_symbols[:number_of_points]

    # Extract the received symbols for plotting
    rx_symbols = result.received_symbols[:number_of_points]

    # Create a directory to save the constellation plot
    result_directory = Path("results")
    result_directory.mkdir(exist_ok=True)
    plt.figure(figsize=(7, 7))
    plt.scatter(
      rx_symbols.real,
      rx_symbols.imag,
      s=10,
      alpha=0.35,
      label="Received symbols",
    )

    # Plot the ideal QPSK constellation points for reference
    ideal_symbols = np.array([
        (1 + 1j) / np.sqrt(2.0),
        (1 - 1j) / np.sqrt(2.0),
        (-1 + 1j) / np.sqrt(2.0),
        (-1 - 1j) / np.sqrt(2.0),
      ],
      dtype=np.complex128,
    )

    # Plot the ideal QPSK constellation points with distinct markers
    plt.scatter(
      ideal_symbols.real,
      ideal_symbols.imag,
      marker="x",
      s=150,
      linewidths=3,
      label="Ideal QPSK points",
    )

    # Decision boundaries
    plt.axhline(0.0, linewidth=1)
    plt.axvline(0.0, linewidth=1)

    # Set plot labels, title, and grid
    plt.xlabel("In-phase component")
    plt.ylabel("Quadrature component")
    plt.title(f"QPSK Constellation at $E_b/N_0={config.ebn0_db:.1f}$ dB")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save the constellation plot to a file
    output_path = (result_directory/ "phase1_qpsk_constellation.png")
    plt.savefig(output_path, dpi=300)
    plt.show()

    # Print the simulation results
    print(f"BER: {result.ber:.6e}")
    print(f"Constellation saved to: {output_path}")
  except Exception as e:
    raise CustomException(e, sys)