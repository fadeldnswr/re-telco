'''
Run Phase 4B simulation: convolutionally coded QPSK over AWGN.
'''
import sys

from src.core.system.phase4.phase4b_coded_qpsk_awgn import Phase4BSimulator
from src.exception.exception import CustomException

# Define main function to run the Phase 4B simulation
def main() -> None:
  try:
    # Create an instance of the Phase4BSimulator
    simulator = Phase4BSimulator()

    # Run the simulation with specified parameters
    result = simulator.run(
      number_of_information_bits=10_000,
      ebn0_db=3.0,
      random_seed=42,
    )

    # Print the simulation results
    print("Phase 4B: Coded QPSK over AWGN")
    print("-" * 52)

    print(f"Information bits          : {result.information_bits.size}")
    print(f"Encoded bits              : {result.encoded_bits.size}")
    print(f"Effective code rate       : {result.effective_code_rate:.6f}")
    print(f"Noise variance            : {result.noise_variance:.6e}")
    print()
    print("Before Viterbi")
    print(f"Coded-bit errors          : {result.coded_channel_bit_errors}")
    print(f"Coded-channel BER         : {result.coded_channel_ber:.6e}")
    print()
    print("Hard Viterbi")
    print(f"Payload bit errors        : {result.hard_payload_bit_errors}")
    print(f"Payload BER               : {result.hard_payload_ber:.6e}")
    print()
    print("Soft Viterbi")
    print(f"Payload bit errors        : {result.soft_payload_bit_errors}")
    print(f"Payload BER               : {result.soft_payload_ber:.6e}")
  except CustomException as e:
    raise CustomException(e, sys)

# Run the main function if this script is executed directly
if __name__ == "__main__":
  main()