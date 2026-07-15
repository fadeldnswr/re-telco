'''
Simulation of QPSK-OFDM over an AWGN channel with a batch BER experiment.
'''
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

from src.core.metrics.communication_metrics import theoritical_qpsk_ber, theoritical_qpsk_ofdm_ber
from src.core.system.phase2.phase2_simulator import Phase2Simulator
from src.exception.exception import CustomException

# Define function to plot the BER curve
if __name__ == "__main__":
  try:
    # Run the Phase 2 OFDM loopback simulation and visualize the results
    simulator = Phase2Simulator()
    ebn0_values_db = np.arange(-2.0, 11.0, 1.0)

    # Run a batch of simulations to compute the BER over multiple subframes
    simulated_ber: list[float] = []
    for index, ebn0_db in enumerate(ebn0_values_db):
      result = (
        simulator.run_ber_experiment(
          ebn0_db=float(ebn0_db),
          number_of_subframes=500,
          random_seed=(10_000 * index + 42),
        )
      )

      # Append the simulated BER to the list for plotting
      simulated_ber.append(result.ber)
      print(f"Eb/N0={ebn0_db:5.1f} dB | Bits={result.total_bits:9d} | Errors={result.total_bit_errors:7d} | BER={result.ber:.6e}")

    # Calculate the theoretical BER for QPSK modulation over AWGN
    theoretical_ber = (theoritical_qpsk_ber(ebn0_values_db))

    # Calculate the theoretical BER for QPSK-OFDM considering cyclic prefix overhead
    theoretical_ber_ofdm = theoritical_qpsk_ofdm_ber(
      ebn0_db=ebn0_values_db,
      useful_samples=1792,
      total_samples=1920,
    )

    # Create the output directory for saving the BER curve plot
    output_directory = Path("results/phase_2")
    output_directory.mkdir(exist_ok=True)

    plt.figure(figsize=(9, 6))

    plt.semilogy(ebn0_values_db, theoretical_ber, label="Theoretical - No CP Overhead")
    plt.semilogy(ebn0_values_db, theoretical_ber_ofdm, label="Theoretical - With CP Overhead")
    plt.semilogy(ebn0_values_db, simulated_ber, marker="o", linestyle="none", label="Simulation")
    plt.xlabel(r"$E_b/N_0$ (dB)")
    plt.ylabel("Bit Error Rate")
    plt.title("QPSK-OFDM BER over AWGN Channel")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()

    # Save the plot to the output directory
    output_path = (output_directory / "phase2b_ofdm_ber_curve.png")
    plt.savefig(output_path, dpi=200)
    plt.show()
  except Exception as e:
    raise CustomException(e, sys)