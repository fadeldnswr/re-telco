'''
Phase 4C BER Curve Simulation: convolutionally coded QPSK-OFDM over static multipath and AWGN.
'''
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

from src.core.system.phase4.phase4c_coded_ofdm import Phase4CSimulator
from src.exception.exception import CustomException


def main() -> None:
  try:
    # Create an instance of the Phase4CSimulator
    simulator = Phase4CSimulator()

    # Define Eb/N0 values in dB for the simulation
    ebn0_values_db = np.arange(0.0, 21.0, 2.0)  
    frames_per_snr = 500

    # Define lists to store BER results for pre-decoder, hard Viterbi, and soft Viterbi
    pre_decoder_ber: list[float] = []
    hard_payload_ber: list[float] = []
    soft_payload_ber: list[float] = []

    for snr_index, ebn0_db in enumerate(ebn0_values_db):
      # Initialize counters for total bits and errors for the current SNR point
      total_coded_bits = 0
      total_information_bits = 0
      total_pre_decoder_errors = 0
      total_hard_errors = 0
      total_soft_errors = 0

      # Loop over the number of frames for the current SNR point
      for frame_index in range(frames_per_snr):
        # Generate a unique random seed for each frame based on the SNR index and frame index
        seed = (100_000 * snr_index + frame_index)
        result = simulator.run(ebn0_db=float(ebn0_db), random_seed=seed)

        # Accumulate the total bits and errors for BER calculation
        total_coded_bits += result.encoded_bits.size
        total_information_bits += result.information_bits.size
        total_pre_decoder_errors += result.pre_decoder_bit_errors
        total_hard_errors += result.hard_payload_bit_errors
        total_soft_errors += result.soft_payload_bit_errors

        # Calculate the current BER values for pre-decoder, hard Viterbi, and soft Viterbi decoding
        current_pre_decoder_ber = (total_pre_decoder_errors / total_coded_bits)
        current_hard_ber = (total_hard_errors / total_information_bits)
        current_soft_ber = (total_soft_errors / total_information_bits)

      # Append the current BER values to the respective lists for plotting
      pre_decoder_ber.append(current_pre_decoder_ber)
      hard_payload_ber.append(current_hard_ber)
      soft_payload_ber.append(current_soft_ber)

      # Print the current BER values for the current Eb/N0 point
      print(f"Eb/N0={ebn0_db:5.1f} dB | Before={current_pre_decoder_ber:.6e} | Hard={current_hard_ber:.6e} | Soft={current_soft_ber:.6e}")

    # Create output directory for saving the plot
    output_directory = Path("results/phase_4")
    output_directory.mkdir(parents=True, exist_ok=True)

    # Plot the BER curves for pre-decoder, hard Viterbi, and soft Viterbi decoding
    plt.figure(figsize=(10, 6))
    plt.semilogy(ebn0_values_db, pre_decoder_ber, marker="o", label="Before Viterbi")
    plt.semilogy(ebn0_values_db, hard_payload_ber, marker="s", label="Hard-input Viterbi")
    plt.semilogy(ebn0_values_db, soft_payload_ber, marker="^", label="Soft-input Viterbi")
    plt.xlabel(r"$E_b/N_0$ (dB)")
    plt.ylabel("Bit Error Rate")
    plt.title("Coded QPSK-OFDM over Static Multipath")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()

    # Save the plot to a file in the output directory
    output_path = ( output_directory / "phase4c_coded_ofdm_ber.png")
    plt.savefig( output_path, dpi=300,)
    plt.show()
    print(f"Plot saved to: {output_path}")
  except CustomException as e:
    raise CustomException(e, sys)

# Run the main function if this script is executed directly
if __name__ == "__main__":
  main()