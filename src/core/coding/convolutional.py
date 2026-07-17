'''
Convolutional coding module for digital communications.
This module provides functionality for convolutional encoding and decoding of binary data.
'''
import numpy as np
from numpy.typing import NDArray

# Define type alias for binary array
BitArray = NDArray[np.uint8]

# Define class for convolutional encoder
class ConvolutionalEncoder:
  constraint_length: int = 3
  memory: int = constraint_length - 1
  code_rate: float = 0.5

  # Define method to encode input bits using convolutional coding
  def encode(self, input_bits: BitArray, terminate: bool = True) -> BitArray:
    # Define input bits as an array of binary values
    bits = np.asarray(input_bits, dtype=np.uint8)

    # Check if the dimension of input bits is not equal to 1
    if bits.ndim != 1:
      raise ValueError("Input bits must be a 1D array.")
    
    # Check if the input bits size is not equal to 0
    if bits.size == 0:
      raise ValueError("Input bits array cannot be empty.")
    
    # Check if the input bits contain only binary values (0 or 1)
    if not np.all((bits == 0) | (bits == 1)):
      raise ValueError("Input bits must contain only binary values (0 or 1).")
    
    # Check if the terminate flag is set to True and append zeros to terminate the trellis
    if terminate:
      input_bits = np.concatenate(
        [bits, np.zeros(self.memory, dtype=np.uint8)]
      )
    else:
      input_bits = bits
    
    # Define state transition and output matrices for the convolutional encoder
    state_1 = 0
    state_2 = 0

    # Perform encoding using the convolutional coding algorithm
    encoded = np.empty(input_bits.size * 2, dtype=np.uint8)

    # Iterate through each input bit and compute the encoded output bits based on the current state
    for index, current_bit in enumerate(input_bits):
      # Convert the current input bit to an integer (0 or 1)
      bit = int(current_bit)

      # Compute the output bits based on the current state and input bit
      output_0 = bit ^ state_1 ^ state_2
      output_1 = bit ^ state_2

      # Update the encoded output bits based on the current state and input bit
      encoded[2 * index] = output_0
      encoded[2 * index + 1] = output_1

      # Update the state variables for the next iteration
      state_2 = state_1
      state_1 = bit
    
    # Return the encoded output bits as a binary array
    return encoded