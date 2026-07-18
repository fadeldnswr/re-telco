'''
Soft viterbi algorithm implementation.
This module provides functions to perform soft decision Viterbi decoding for convolutional codes.
Code:
  Constraint length K = 3
  Generator polynomials = (7, 5) octal

LLR convention:
  positive LLR -> bit 0 is more likely
  negative LLR -> bit 1 is more likely
'''
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# Define type aliases for clarity
FloatArray = NDArray[np.float64]
BitArray = NDArray[np.uint8]
IntegerArray = NDArray[np.int64]

# Define class for soft trellis
@dataclass(frozen=True, slots=True)
class SoftTrellis:
  next_state: IntegerArray  # Next state for each state and input
  # Shape : (num_states, num_of_inputs, coded_bits_per_input)
  output_bits: NDArray[np.uint8]  # Output bits for each state and input

# Define function to build soft trellis for viterbi decoding
def build_soft_viterbi_trellis() -> SoftTrellis:
  """
    Build a trellis matching a K=3, rate-1/2, (7,5) encoder.
    State representation:
      state bit 1 = u[i-1]
      state bit 0 = u[i-2]
  """
  # Define number of states and inputs
  number_of_states = 4  # 2^(K-1) = 2^(3-1) = 4
  number_of_inputs = 2  # Binary input (0 or 1)

  # Define next state and output bits arrays
  next_state = np.empty((
    number_of_states, 
    number_of_inputs,
  ), dtype=np.int64)

  # Define output bits array with shape (num_states, num_of_inputs, coded_bits_per_input)
  output_bits = np.empty((
    number_of_states,
    number_of_inputs,
    2,  # Coded bits per input (rate 1/2)
  ), dtype=np.uint8)

  # Iterate over each state and input to fill next_state and output_bits
  for current_state in range(number_of_states):
    # Define previous bits based on current state
    previous_bit_1 = (current_state >> 1) & 1 # u[i-1]
    previous_bit_2 = current_state & 1        # u[i-2]

    # Iterate over each possible input (0 or 1)
    for input_bit in (0, 1):
      # Define output bits based on the convolutional encoder's generator polynomials
      output_0 = (
        input_bit ^ previous_bit_1 ^ previous_bit_2  # Output bit 0 (7 in octal)
      )
      output_1 = (
        input_bit ^ previous_bit_2  # Output bit 1 (5 in octal)
      )

      # Calculate next state based on current state and input
      new_state = ((input_bit << 1) | previous_bit_1)  # Keep only the last 2 bits
      next_state[current_state, input_bit] = new_state

      # Store output bits in the output_bits array
      output_bits[current_state, input_bit] = np.array([
        output_0, output_1,
      ], dtype=np.uint8)

  # Return the constructed SoftTrellis object
  return SoftTrellis(
    next_state=next_state,
    output_bits=output_bits
  )

# Define function to perform llr branch metric calculation
def llr_branch_metric(received_llrs: FloatArray, expected_bits: BitArray) -> float:
  """
  Calculate a max-log soft branch metric.
  LLR > 0 means bit 0 is preferred.
  LLR < 0 means bit 1 is preferred.
  For expected bit b:
    bipolar expected value = 1 - 2b
  A lower metric means a more likely trellis branch.
  """
  # Define received llrs as an array of float64
  received_llrs = np.asarray(received_llrs, dtype=np.float64)

  # Define expected bits as an array of uint8
  expected_bits = np.asarray(expected_bits, dtype=np.uint8)

  # Check if both dimensions match
  if received_llrs.shape != expected_bits.shape:
    raise ValueError("Received LLRs and expected bits must have the same shape.")
  
  # Expected bipolar values for the expected bits
  expected_bipolar = 1.0 - 2.0 * expected_bits.astype(np.float64)

  # Correct the received LLRs based on the expected bipolar values
  return float(
    -np.sum(
      received_llrs * expected_bipolar
    )
  )

# Define class for soft viterbi decoder
class SoftViterbiDecoder:
  number_of_states: int = 4
  coded_bits_per_input: int = 2
  encoder_memory: int = 2

  # Define the constructor for the SoftViterbiDecoder class
  def __init__(self, trellis: SoftTrellis | None = None):
    self._trellis = trellis if trellis is not None else build_soft_viterbi_trellis()

  # Define static method to validate input bits
  @staticmethod
  def _validate_input(received_llrs: BitArray) -> None:
    # Check if the dimension of received llrs is not equal to 1
    if received_llrs.ndim != 1:
      raise ValueError("Received LLRs must be a 1D array.")
    
    # Check if the size of received llrs is not equal to 0
    if received_llrs.size == 0:
      raise ValueError("Received LLRs array cannot be empty.")
    
    # Check if the size of received llrs is not a multiple of 2
    if received_llrs.size % 2 != 0:
      raise ValueError("Received LLRs array size must be a multiple of 2.")
    
    # Check if the received llrs contain only binary values (0 or 1)
    if not np.all(np.isfinite(received_llrs)):
      raise ValueError("Received LLRs must contain only binary values (0 or 1).")
  
  # Define method to perform traceback through the survivor information to reconstruct the decoded bits
  def _traceback(self, survivor_previous_state: IntegerArray, survivor_input_bit: NDArray[np.int8], final_state: int) -> BitArray:
    '''
    Recover the maximum likelihood sequence
    '''
    # Define the number of steps based on the survivor previous state array
    number_of_steps = survivor_previous_state.shape[0]

    # Initialize the decoded bits array with -1 values
    decoded_bits = np.empty(number_of_steps, dtype=np.uint8)
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
  
  # Define method to decode
  def decode(self, received_llrs: FloatArray, terminated: bool = True) -> BitArray:
    """
    Decode a sequence of coded-bit LLRs.
    Args:
      received_llrs:
        One LLR per received coded bit.
      terminated:
        True when two zero tail bits were added by the encoder.
    Returns:
      Decoded information bits with termination bits removed.
    """
    # Define received llrs as an array of float64
    received_llrs = np.asarray(received_llrs, dtype=np.float64)

    # Validate the input received llrs
    self._validate_input(received_llrs)

    # Define received pairs
    received_pairs = received_llrs.reshape(-1, self.coded_bits_per_input)

    # Define the number of time steps based on the received pairs
    number_of_steps = received_pairs.shape[0]

    # Initialize the previous metrics with infinity for all states except the initial state (state 0)
    previous_metrics = np.full(
      self.number_of_states,
      np.inf,
      dtype=np.float64
    )

    # Encoder starts in state 0, so set the metric for state 0 to 0
    previous_metrics[0] = 0.0

    # Initialize survivor arrays to store the previous state and input bit for each state at each time step
    survivor_previous_state = np.full((
      number_of_steps, self.number_of_states
    ), -1, dtype=np.int64)
    survivor_input_bit = np.full((
      number_of_steps, self.number_of_states
    ), -1, dtype=np.int8)

    # Iterate through each time step to compute the path metrics and update the survivor information
    for time_index in range(number_of_steps):
      # Define the received pair for the current time index
      received_pair = received_pairs[time_index]

      # Initialize the current metrics for the current time index with infinity for all states
      current_metrics = np.full(
        self.number_of_states, np.inf, dtype=np.float64
      )
      
      # Iterate through each state to compute the path metrics and update the survivor information
      for current_state in range(self.number_of_states):
        # Define the current path metric for the current state
        current_path_metric = previous_metrics[current_state]

        # Check if the current path metric is infinity, and skip if it is
        if not np.isfinite(current_path_metric):  # Skip if the current path metric is infinity
          continue

        # Iterate through each possible input (0 or 1) to compute the next state and output bits
        for input_bit in (0, 1):
          next_state = int(
            self._trellis.next_state[current_state, input_bit]
          )
          
          # Expected output bits based on the current state and input bit
          expected_pair = self._trellis.output_bits[current_state, input_bit]

          # Calculate the branch metric using the received pair and expected output bits
          branch_metric = llr_branch_metric(received_pair, expected_pair)

          # Calculate the candidate metric for the next state
          candidate_metric = current_path_metric + branch_metric

          # Check if the candidate metric is less than the current metric for the next state
          if candidate_metric < current_metrics[next_state]:
            # Update the current metric for the next state and store the survivor information
            current_metrics[next_state] = candidate_metric
            survivor_previous_state[time_index, next_state] = current_state
            survivor_input_bit[time_index, next_state] = input_bit
      
      # Metric normalization to prevent numerical issues by subtracting the minimum metric from all current metrics
      minimum_metric = np.min(current_metrics[np.isfinite(current_metrics)])
      current_metrics = current_metrics - minimum_metric
      previous_metrics = current_metrics
    
    # Determine the final state for traceback based on whether the encoder is terminated or not
    final_state = 0 if terminated else int(np.argmin(previous_metrics))

    # Perform traceback through the survivor information to reconstruct the decoded bits
    decoded_with_tail = self._traceback(survivor_previous_state, survivor_input_bit, final_state)

    # Remove the tail bits if the encoder is terminated
    if terminated:
      if (decoded_with_tail.size < self.encoder_memory):
        raise ValueError("Decoded bits are fewer than the encoder memory; cannot remove tail bits.")
      return decoded_with_tail[:-self.encoder_memory]  # Remove tail bits
    return decoded_with_tail  # Return decoded bits without removing tail bits for non-terminated case