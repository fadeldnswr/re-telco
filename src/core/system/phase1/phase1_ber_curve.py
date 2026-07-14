'''
Compare simulated and theoretical BER curves for a QPSK system over an AWGN channel.
'''
import numpy as np
import matplotlib.pyplot as plt
import sys

from pathlib import Path
from src.core.metrics.communication_metrics import theoritical_qpsk_ber, calculate_ber
from src.core.system.phase1.phase1_simulator import Phase1Simulator
from src.core.system.config import SimulationConfig
from src.exception.exception import CustomException

# Define function to run BER sweep
def run_ber_sweep() -> None:
  try:
    # Define Eb/N0 range for the sweep
    ebn0_db_values = np.arange(-2.0, 11.0, 1.0)  # Eb/N0 from -2 dB to 10 dB

    # Define simulator and lists to store results
    simulator = Phase1Simulator()
    simulated_ber: list[float] = []
    bit_error_counts: list[int] = []

    # Run simulations for each Eb/N0 value
    for index, ebn0_db in enumerate(ebn0_db_values):
      config = SimulationConfig(
        ebn0_db=float(ebn0_db),
        number_of_bits=1_000_000,
        random_seed=42 + index,  # Different seed for each simulation
      )

      # Run the simulation and collect results
      result = simulator.run(config)
      simulated_ber.append(result.ber)
      bit_error_counts.append(result.number_of_bit_errors)

      # Log the simulation results for each Eb/N0 value
      print(f"Eb/N0: {ebn0_db:.2f} dB | Bit Errors: {result.number_of_bit_errors:7d} | BER: {result.ber:.6e}")

    # Define the theoretical BER for QPSK
    theoretical_ber = theoritical_qpsk_ber(ebn0_db_values)

    # Define the directory to save the BER curve plot
    result_directory = Path("results")
    result_directory.mkdir(exist_ok=True)

    # Plot the simulated and theoretical BER curves
    plt.figure(figsize=(8, 6))
    plt.semilogy(ebn0_db_values, theoretical_ber, label="Theoretical QPSK")
    plt.semilogy(ebn0_db_values, simulated_ber, label="Simulation", marker='o', linestyle='none')
    plt.xlabel(r"$E_b/N_0$ (dB)")
    plt.ylabel("Bit Error Rate")
    plt.title("QPSK BER Performance over AWGN Channel")
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()

    # Define output path
    output_path = result_directory / "phase1_qpsk_ber_curve.png"

    # Save the BER curve plot to a file
    plt.savefig(output_path, dpi=300)
    plt.show()

    print(f"BER curve saved to: {output_path}")
  except Exception as e:
    raise CustomException(e, sys)