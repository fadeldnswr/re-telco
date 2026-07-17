"""
Phase 4A: convolutional encoder and hard Viterbi decoder.
"""

import numpy as np
import sys

from src.core.coding.convolutional import ConvolutionalEncoder
from src.core.coding.viterbi import HardViterbiDecoder
from src.exception.exception import CustomException

# Define main function to run the Phase 4A simulation
def main() -> None:
  try:
    # Define random number generator with a fixed seed for reproducibility
    rng = np.random.default_rng(42)
    number_of_information_bits = 1_000

    # Define random information bits to be encoded and decoded
    information_bits = rng.integers(0, 2, size=number_of_information_bits, dtype=np.uint8)

    # Initialize the convolutional encoder and hard Viterbi decoder
    encoder = ConvolutionalEncoder()
    decoder = HardViterbiDecoder()

    # Encode the information bits and then decode the encoded bits
    encoded_bits = encoder.encode(information_bits, terminate=True)
    decoded_bits = decoder.decode(encoded_bits, terminated=True)

    # Calculate the number of bit errors between the original information bits and the decoded bits
    bit_errors = int(np.count_nonzero(information_bits != decoded_bits))

    # Calculate the effective code rate based on the number of information bits and encoded bits
    effective_code_rate = (information_bits.size / encoded_bits.size)
    
    # Print the simulation results, including the number of information bits, 
    # encoded bits, nominal and effective code rates, decoded bits, and bit errors
    print("Phase 4A: Convolutional Coding Loopback")
    print("-" * 52)
    print(f"Information bits        : {information_bits.size}")
    print(f"Encoded bits            : {encoded_bits.size}")
    print(f"Nominal code rate       : 1/2")
    print(f"Effective code rate     : {effective_code_rate:.6f}")
    print(f"Decoded bits            : {decoded_bits.size}")
    print(f"Bit errors              : {bit_errors}")
  except CustomException as e:
    raise CustomException(e, sys)

# Run the main function if this script is executed directly
if __name__ == "__main__":
  main()