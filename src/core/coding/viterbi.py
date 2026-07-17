"""
Hard-input Viterbi decoder for a rate-1/2 convolutional code.

Code configuration:
  Constraint length: K = 3
  Generator polynomials: (7, 5) octal
"""

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

# Define type alias for binary array
BitArray = NDArray[np.uint8]
IntegerArray = NDArray[np.int64]

# Define trellis structure for the convolutional code
@dataclass(frozen=True, slots=True)
class Trellis:
  next_state: IntegerArray
  output_bits: NDArray[np.uint8]

# Define method to create a trellis for a half k3 convolutional code
def build_rate_half_k3_trellis() -> Trellis:
  '''
  Build the trellis for the rate-1/2, K=3, (7,5) code.
    State encoding:
      state bit 1 = u[i-1]
      state bit 0 = u[i-2]
    next_state = [current_input, previous u[i-1]]
  '''
  # Define the next state and output bits for each state and input
  number_of_states = 4
  number_of_inputs = 2

  # Define the next state and output bits arrays
  next_state = np.empty((
    number_of_states, number_of_inputs
  ), dtype=np.int64)
  output_bits = np.empty((
    number_of_states, number_of_inputs, 2
  ), dtype=np.uint8)

  # Fill the next state and output bits arrays based on the convolutional code
  for state in range(number_of_states):
    # Extract previous bits from the current state
    previous_bit_1 = (state >> 1) & 1
    previous_bit_0 = state & 1

    # Iterate through each possible input bit (0 or 1)
    for input_bit in range(number_of_inputs):
      output_0 = input_bit ^ previous_bit_1 ^ previous_bit_0
      output_1 = input_bit ^ previous_bit_0

      # Define new state based on the current input and previous bits
      new_state = (input_bit << 1) | previous_bit_1

      # Store the next state and output bits in the respective arrays
      next_state[state, input_bit] = new_state
      output_bits[state, input_bit] = np.array(
        [output_0, output_1], dtype=np.uint8
      ) 
  
  # Return the trellis structure containing next state and output bits
  return Trellis(next_state=next_state, output_bits=output_bits)

# Define function to calculate hamming distance
def hamming_distance(received_pair: BitArray, expected_pair: BitArray) -> int:
  '''Calculate the Hamming distance between two pairs of bits.'''
  return int(
    np.count_nonzero(
      received_pair != expected_pair
    )
  )

# Implement the hard viterbi decoder
class HardViterbiDecoder:
  number_of_states: int = 4
  coded_bits_per_input: int = 2
  encoder_memory: int = 2

  # Construct the trellis for the convolutional code
  def __init__(self, trellis: Trellis | None = None) -> None:
    self._trellis = trellis if trellis is not None else build_rate_half_k3_trellis()
  
  # Define static method to validate input bits
  @staticmethod
  def _validate_input(received_bits: BitArray) -> None:
    # Check if the dimension of received bits is not equal to 1
    if received_bits.ndim != 1:
      raise ValueError("Received bits must be a 1D array.")
    
    # Check if the size of received bits is not equal to 0
    if received_bits.size == 0:
      raise ValueError("Received bits array cannot be empty.")
    
    # Check if the size of received bits is not a multiple of 2
    if received_bits.size % 2 != 0:
      raise ValueError("Received bits array size must be a multiple of 2.")
    
    # Check if the received bits contain only binary values (0 or 1)
    if not np.all((received_bits == 0) | (received_bits == 1)):
      raise ValueError("Received bits must contain only binary values (0 or 1).")
  
  # Define method to decode received bits using the Viterbi algorithm
  def decode(self, received_bits: BitArray, terminated: bool = True) -> BitArray:
    """
      Decode a hard-decision convolutionally encoded bit sequence.
      Args:
        received_bits:
          One-dimensional array of coded bits.
        terminated:
          When True, the encoder is assumed to have appended
          two zero tail bits and ended in state zero.
      Returns:
        Decoded information bits. Tail bits are removed when
        terminated=True.
    """
    # Define received bits as an array of binary values
    received_bits = np.asarray(received_bits, dtype=np.uint8)

    # Validate the received bits input
    self._validate_input(received_bits)

    # Define the number of received pairs based on the length of received bits
    received_pairs = received_bits.reshape(-1, 2)
    number_of_steps = received_pairs.shape[0]

    # Inifinity value for path metrics
    infinity = np.iinfo(np.int64).max // 4

    # Metric for reaching every state at each step, initialized to infinity
    previous_metrics = np.full(
      self.number_of_states,
      infinity,
      dtype=np.int64
    )

    # Start in state 0 with metric 0
    previous_metrics[0] = 0

    # Store survivor information
    survivor_previous_state = np.full((
      number_of_steps, self.number_of_states
    ), -1, dtype=np.int64)
    survivor_input_bit = np.full((
      number_of_steps, self.number_of_states
    ), -1, dtype=np.int8)

    # Iterate through each step in the received pairs to compute path metrics and survivors
    for time_index in range(number_of_steps):
      # Define received pair for the current time index
      received_pair = received_pairs[time_index]

      # Define current metrics for the current time index, initialized to infinity
      current_metrics = np.full(
        self.number_of_states, infinity, dtype=np.int64
      )

      # Iterate through each state to compute the path metrics and update survivors
      for current_state in range(self.number_of_states):
        # Define current path metric for the current state
        current_path_metric = previous_metrics[current_state]
        if current_path_metric >= infinity: # Skip if the current path metric is infinity
          continue

        # Iterate through each possible input bit (0 or 1) to compute the next state and output bits
        for input_bit in (0, 1):
          # Define next state and expected output bits based on the current state and input bit
          next_state = self._trellis.next_state[current_state, input_bit]

          # Define expected output bits based on the current state and input bit
          expected_pair = self._trellis.output_bits[current_state, input_bit]

          # Calculate the Hamming distance between the received pair and expected output bits
          branch_metric = hamming_distance(received_pair, expected_pair)

          # Calculate the new path metric for the next state
          candidate_metric = current_path_metric + branch_metric

          # Check if the candidate metric is less than the current metric for the next state
          if (candidate_metric < current_metrics[next_state]):
            # Update the current metric for the next state and store the survivor information
            current_metrics[next_state] = candidate_metric

            # Store the previous state and input bit that led to the next state in the survivor arrays
            survivor_previous_state[time_index, next_state] = current_state
            survivor_input_bit[time_index, next_state] = input_bit
      
      # Update the previous metrics for the next iteration
      previous_metrics = current_metrics
    
    # Check if terminated
    if terminated:
      # Ensure that the final state is zero for terminated decoding
      final_state = 0
    else:
      # Choose the state with the minimum metric for non-terminated decoding
      final_state = int(np.argmin(previous_metrics))
    
    # Backtrack through the survivor information to reconstruct the decoded bits
    decoded_with_tail = self._traceback(survivor_previous_state, survivor_input_bit, final_state)

    # Check if terminated and remove tail bits from the decoded output
    if terminated:
      if (decoded_with_tail.size < self.encoder_memory):
        raise ValueError("Decoded output is smaller than the encoder memory.")
      return decoded_with_tail[:-self.encoder_memory]
    return decoded_with_tail
  
  # Define method to perform traceback through the survivor information to reconstruct the decoded bits
  def _traceback(self, survivor_previous_state: IntegerArray, survivor_input_bit: IntegerArray, final_state: int) -> BitArray:
    '''
    Recover the maximum likelihood sequence
    '''
    # Define the number of steps based on the survivor previous state array
    number_of_steps = survivor_previous_state.shape[0]

    # Initialize the decoded bits array with -1 values
    decoded_bits = np.empty(number_of_steps, dtype=np.int8)
    state = final_state

    # Backtrack through the survivor information to reconstruct the decoded bits
    for time_index in range(number_of_steps - 1, -1, -1):
      # Retrieve the input bit and previous state from the survivor arrays for the current time index and state
      input_bit = int(survivor_input_bit[time_index, state])
      previous_state = int(survivor_previous_state[time_index, state])

      # Check if the input bit or previous state is invalid (-1)
      if input_bit < 0 or previous_state < 0:
        raise ValueError("Invalid survivor information during traceback.")
      
      # Store the input bit in the decoded bits array and update the state for the next iteration
      decoded_bits[time_index] = input_bit
      state = previous_state
    
    # Return the reconstructed decoded bits as a binary array
    return decoded_bits
