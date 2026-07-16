'''
Frequency domain for OFDM signals
'''
from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np

# Define complex number array type for better type hinting
ComplexArray = NDArray[np.complex128]

# Define function to validate equalizer input
def validate_equalizer_input(receiver_grid: ComplexArray, channel_response_shifted: ComplexArray) -> tuple[ComplexArray, ComplexArray]:
  # Define receiver grid as an array of complex numbers
  receiver_grid = np.asarray(receiver_grid, dtype=np.complex128)

  # Define channel response shifted as an array of complex numbers
  channel_response_shifted = np.asarray(channel_response_shifted, dtype=np.complex128)

  # Check if the dimension of receiver grid and channel response shifted are not equal to 1
  if receiver_grid.ndim != 2:
    raise ValueError("Receiver grid must be a 2D array.")
  
  # Check if the dimension of channel response shifted is not equal to 1
  if channel_response_shifted.ndim != 1:
    raise ValueError("Channel response shifted must be a 1D array.")
  
  # Check if receiver grid and channel response shifted have the same size
  if receiver_grid.shape[1] != channel_response_shifted.size:
    raise ValueError("Receiver grid and channel response shifted must have the same size.")
  
  # Return validated receiver grid and channel response shifted
  return receiver_grid, channel_response_shifted

# Define class for Zero Forcing Equalizer algorithm
@dataclass(frozen=True, slots=True)
class ZeroForcingEqualizer:
  epsilon: float = 1e-12

  # Define method to equalize received symbols using Zero Forcing Equalizer algorithm
  def equalize(self, receiver_grid: ComplexArray, channel_response_shifted: ComplexArray) -> ComplexArray:
    # Validate input parameters
    receiver_grid, channel_response = validate_equalizer_input(
      receiver_grid, channel_response_shifted
    )

    # Define safe response by adding epsilon to channel response shifted to avoid division by zero
    safe_response = np.where(
      np.abs(channel_response_shifted) > self.epsilon,
      channel_response, self.epsilon + 0j
    )

    # Return equalized symbols by dividing receiver grid by safe response
    return (
      receiver_grid / safe_response[np.newaxis, :]
    )

# Define class for Minimum Mean Square Error Equalizer algorithm
@dataclass(frozen=True, slots=True)
class MMSEEqualizer:
  epsilon: float = 1e-12

  # Define method to equalize received symbols using Minimum Mean Square Error Equalizer algorithm
  def equalize(self, receiver_grid: ComplexArray, channel_response_shifted: ComplexArray, noise_variance: float) -> ComplexArray:
    # Validate input parameters
    receiver_grid, channel_response = validate_equalizer_input(
      receiver_grid, channel_response_shifted
    )

    # Check if noise variance is negative
    if noise_variance < 0.0:
      raise ValueError("Noise variance must be non-negative.")
    
    # Define denominator for MMSE equalization by adding noise variance to the squared magnitude of channel response shifted
    denominator = (
      np.abs(channel_response) ** 2 + noise_variance
      + self.epsilon
    )

    # Define weights for MMSE equalization by dividing channel response shifted by the denominator
    weights = np.conjugate(channel_response) / denominator

    # Return equalized symbols by multiplying receiver grid by weights
    return (
      receiver_grid * weights[np.newaxis, :]
    )