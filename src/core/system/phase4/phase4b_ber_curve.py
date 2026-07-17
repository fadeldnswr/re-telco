'''
BER curve for Phase 4B: convolutionally coded QPSK over AWGN.
'''
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
from src.core.system.phase4.phase4b_coded_qpsk_awgn import Phase4BSimulator
from src.exception.exception import CustomException

# Define main function to run the Phase 4B BER curve simulation
def main() -> None:
  try:
    # Define simulator parameters
    simulator = Phase4BSimulator()

    # Define eb/n0 range in dB for the simulation
    ebn0_db_range = np.arange(-2, 7, 1)  # Eb/N0 from -2 dB to 6 dB
    framer_per_snr = 100  # Number of frames per SNR point
    information_bits_per_frame = 5000  # Number of information bits per frame

    # Define lists to store results for hard and soft Viterbi decoding
    coded_channel_ber: list[float] = []
    hard_payload_ber: list[float] = []
    soft_payload_ber: list[float] = []

    # Loop over each Eb/N0 value in the defined range
    for snr_index, ebn0_db in enumerate(ebn0_db_range):
      # Initialize counters for total bits and errors
      total_coded_bits = 0
      total_information_bits = 0
      total_coded_errors = 0
      total_hard_errors = 0
      total_soft_errors = 0

      # Loop over the number of frames for the current SNR point
      for frame_index in range(framer_per_snr):
        seed = (
          100_000 * snr_index + frame_index
        )

        # Run the Phase 4B simulation for the current frame
        result = simulator.run(
          number_of_information_bits=information_bits_per_frame,
          ebn0_db=float(ebn0_db),
          random_seed=seed,
        )

        # Accumulate the total bits and errors for BER calculation
        total_coded_bits += result.encoded_bits.size
        total_information_bits += result.information_bits.size
        total_coded_errors += result.coded_channel_bit_errors
        total_hard_errors += result.hard_payload_bit_errors
        total_soft_errors += result.soft_payload_bit_errors

      # Calculate coded, hard, and soft BER for the current SNR point
      coded_ber_value = total_coded_errors / total_coded_bits
      hard_ber_value = total_hard_errors / total_information_bits
      soft_ber_value = total_soft_errors / total_information_bits

      # Append the calculated BER values to the respective lists
      coded_channel_ber.append(coded_ber_value)
      hard_payload_ber.append(hard_ber_value)
      soft_payload_ber.append(soft_ber_value)

      # Print the results for the current SNR point
      print(f"Eb/N0={ebn0_db:5.1f} dB | Coded bits={coded_ber_value:.6e} | Hard={hard_ber_value:.6e} | Soft={soft_ber_value:.6e}")
    
    # Define output path
    output_directory = Path("results/phase_4")
    output_directory.mkdir(parents=True, exist_ok=True)

    # Plot the BER curves for coded, hard, and soft Viterbi decoding
    plt.figure(figsize=(10, 6))
    plt.semilogy(ebn0_db_range, coded_channel_ber, marker="o", label="Before Viterbi")
    plt.semilogy(ebn0_db_range, hard_payload_ber, marker="s", label="Hard-input Viterbi")
    plt.semilogy(ebn0_db_range, soft_payload_ber, marker="^", label="Soft-input Viterbi")
    plt.xlabel(r"$E_b/N_0$ (dB)")
    plt.ylabel("Bit Error Rate")
    plt.title("Convolutionally Coded QPSK over AWGN")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()

    # Save the plot to the output directory
    output_path = (output_directory / "phase4b_hard_soft_viterbi_ber.png")
    plt.savefig(output_path, dpi=300)
    plt.show()
    print(f"Plot saved to: {output_path}")
  except CustomException as e:
    raise CustomException(e, sys)

# Run the main function if this script is executed directly
if __name__ == "__main__":
  main()