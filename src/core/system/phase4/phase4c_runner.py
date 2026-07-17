'''
Phase 4C: convolutionally coded QPSK-OFDM over static multipath and AWGN.
'''
from src.core.system.phase4.phase4c_coded_ofdm import Phase4CSimulator
from src.exception.exception import CustomException
import sys

# Define main function to run the Phase 4C simulation
def main() -> None:
  try:
    # Create an instance of the Phase4CSimulator
    simulator = Phase4CSimulator()
    result = simulator.run(ebn0_db=10.0, random_seed=42)

    # Print the simulation results for Phase 4C
    print("Phase 4C: Coded QPSK-OFDM over Static Multipath + AWGN")
    print("-" * 64)

    # Print the number of information bits, encoded bits, maximum channel delay,
    # minimum CP length, CP condition satisfaction, and noise variance
    print(f"Information bits          : {result.information_bits.size}")
    print(f"Encoded bits              : {result.encoded_bits.size}")
    print(f"Maximum channel delay     : {result.maximum_channel_delay}")
    print(f"Minimum CP length         : {result.minimum_cp_length}")
    print(f"CP condition satisfied    : {result.cp_condition_satisfied}")
    print(f"Noise variance            : {result.noise_variance:.6e}")
    print()
    print("Before Viterbi")
    print(f"Coded-bit errors          : {result.pre_decoder_bit_errors}")
    print(f"Coded-bit BER             : {result.pre_decoder_ber:.6e}")
    print()
    print("Hard Viterbi")
    print(f"Payload errors            : {result.hard_payload_bit_errors}")
    print(f"Payload BER               : {result.hard_payload_ber:.6e}")
    print()
    print("Soft Viterbi")
    print(f"Payload errors            : {result.soft_payload_bit_errors}")
    print(f"Payload BER               : {result.soft_payload_ber:.6e}")
  except CustomException as e:
    raise CustomException(e, sys)

#   Run the main function if this script is executed directly
if __name__ == "__main__":
  main()