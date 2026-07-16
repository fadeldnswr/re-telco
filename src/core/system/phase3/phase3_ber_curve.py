'''
Run BER curve experiment for Phase 3B static multipath channel with and without equalization.
'''
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sys

from src.core.system.phase3.phase3_simulator import EqualizerType, Phase3Simulator
from src.exception.exception import CustomException

# Define main function to run the Phase 3B BER curve experiment
def run_ber_curve() -> None:
  try:
    # Define the simulator
    simulator = Phase3Simulator()

    # Define Eb/N0 range in dB
    ebn0_values_db = np.arange(0.0, 21.0, 2.0)

    # Define number of subframes and base seed
    number_of_subframes = 500
    base_seed = 42

    # Define lists to store BER results for each equalization type
    no_eq_ber: list[float] = []
    zf_ber: list[float] = []
    mmse_ber: list[float] = []

    # Iterate over Eb/N0 values to run the BER experiment for each equalization type
    for snr_index, ebn0_db in enumerate(ebn0_values_db):
      # Define error variable to track errors during simulation
      no_eq_error = 0
      zf_error = 0
      mmse_error = 0
      total_bits = 0

      # Iterate over the number of subframes to simulate
      for frame_index in range(number_of_subframes):
        # Define random seed for reproducibility
        frame_seed = (
          base_seed + snr_index * number_of_subframes + frame_index
        )

        # Define common args
        common_args = {
          "ebn0_db": ebn0_db,
          "random_seed": frame_seed,
        }

        # Run simulation without equalization
        no_eq_result = simulator.run(
          **common_args, 
          equalizer_type=EqualizerType.NONE
        )

        # Run simulation with perfect-CSI ZF equalization
        zf_result = simulator.run(
          **common_args, 
          equalizer_type=EqualizerType.ZF
        )

        # Run simulation with MMSE equalization
        mmse_result = simulator.run(
          **common_args, 
          equalizer_type=EqualizerType.MMSE
        )

        # Add the number of bit errors and total bits for each equalization type
        total_bits += no_eq_result.transmitted_bits.size
        no_eq_error += no_eq_result.bit_errors
        zf_error += zf_result.bit_errors
        mmse_error += mmse_result.bit_errors
      
      # Calculate BER for each equalization type and append to the respective lists
      no_eq_ber_value = no_eq_error / total_bits
      zf_ber_value = zf_error / total_bits
      mmse_ber_value = mmse_error / total_bits

      # Append the calculated BER values to the respective lists
      no_eq_ber.append(no_eq_ber_value)
      zf_ber.append(zf_ber_value)
      mmse_ber.append(mmse_ber_value)

      # Print the result
      print(
        f"Eb/N0: {ebn0_db:.1f} dB | "
        f"No Equalization BER: {no_eq_ber_value:.6e} | "
        f"ZF BER: {zf_ber_value:.6e} | "
        f"MMSE BER: {mmse_ber_value:.6e}"
      )
    
    # Define output directory for saving the BER curve plot
    output_directory = Path("results/phase_3")
    output_directory.mkdir(parents=True, exist_ok=True)

    # Plot the BER curve for each equalization type
    plt.figure(figsize=(10, 6))
    plt.semilogy(ebn0_values_db, no_eq_ber, marker="o", label="No equalization")
    plt.semilogy(ebn0_values_db, zf_ber, marker="s", label="Perfect-CSI ZF")
    plt.semilogy(ebn0_values_db, mmse_ber, marker="^", label="Perfect-CSI MMSE")
    plt.xlabel(r"$E_b/N_0$ (dB)")
    plt.ylabel("Bit Error Rate")
    plt.title("QPSK-OFDM BER over Static Multipath")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()

    # Save the BER curve plot to the output directory
    output_path = (output_directory / "phase3_ber_equalizer_comparison.png")
    plt.savefig(output_path, dpi=200)
    plt.show()
    print(f"BER plot saved to: {output_path}")
  except Exception as e:
    raise CustomException(e, sys)