'''
Static multipath channel implementation.
This module defines a static multipath channel model that simulates the effect of multipath propagation on a transmitted signal.
'''
from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np

# Define complex number array type for better type hinting
ComplexArray = NDArray[np.complex128]
IntegerArray = NDArray[np.int64]

# Define dataclass for multipath channel result
@dataclass(frozen=True, slots=True)
class StaticMultipathResult:
  received_signal: ComplexArray
  impulse_response: ComplexArray
  frequency_response: ComplexArray

# Define static multipath channel class
@dataclass(frozen=True, slots=True)
class StaticMultipathChannel:
  '''
  Time-invariant tapped-delay-line channel.

  delays:
    Integer sample delays for each propagation path.

  gains:
    Complex path gains corresponding to the delays.
  '''
  delays: IntegerArray
  gains: ComplexArray
  normalize_power: bool = True

  # Define post init method to validate input parameters
  def __post_init__(self) -> None:
    # Define delay as an array of integers
    delays = np.asarray(self.delays, dtype=np.int64)

    # Define gains as an array of complex numbers
    gains = np.asarray(self.gains, dtype=np.complex128)

    # Check if the dimension of delays and gains are not equal to 1
    if delays.ndim != 1:
      raise ValueError("Delays must be a 1D array.")
    if gains.ndim != 1:
      raise ValueError("Gains must be a 1D array.")
    
    # Check if delay array size is not equal to 0
    if delays.size == 0:
      raise ValueError("At least one channel path must be specified.")
    
    # Check if the size of delays and gains are not equal
    if delays.size != gains.size:
      raise ValueError("Delays and gains must have the same size.")
    
    # Check if the delay values are negative
    if np.any(delays < 0):
      raise ValueError("Delays must be non-negative integers.")
    
    # Check if delays are not unique
    if np.unique(delays).size != delays.size:
      raise ValueError("Delays must be unique.")
    
    # Check if the gains are all zeros
    if np.all(np.abs(gains) == 0.0):
      raise ValueError("At least one channel gain must be non-zero.")
    
    # Set attribute for delays and gains
    object.__setattr__(self, "delays", delays)
    object.__setattr__(self, "gains", gains)
  
  # Define property for maximum delay samples
  @property
  def maximum_delay_samples(self) -> int:
    return int(np.max(self.delays))
  
  # Define property for impulse response
  @property
  def impulse_response(self) -> ComplexArray:
    # Define impulse response as an array of zeros with length equal to maximum delay + 1
    impulse_response = np.zeros(self.maximum_delay_samples + 1, dtype=np.complex128)

    # Copy the gains into the impulse response at the specified delays
    gains = self.gains.copy()
    if self.normalize_power:
      total_power = np.sum(np.abs(gains) ** 2)
      gains /= np.sqrt(total_power)
    
    # Set the impulse response at the specified delays to the corresponding gains
    impulse_response[self.delays] = gains
    return impulse_response
  
  # Define method for frequency response
  def frequency_response(self, fft_size: int) -> ComplexArray:
    '''Calculate H[k] in normal NumPy FFT ordering.
      Do not use norm='ortho' here. With an orthonormal OFDM FFT/IFFT
      pair, the subcarrier relation is still Y[k] = H[k] X[k], where
      H[k] is the ordinary FFT of the channel impulse response.
    '''
    # Check if fft_size is less than or equal to maximum delay samples
    if fft_size <= self.maximum_delay_samples:
      raise ValueError("FFT size must be greater than the maximum delay samples.")
    
    # Calculate the fft of the impulse response with the specified fft_size
    return np.fft.fft(self.impulse_response, n=fft_size).astype(np.complex128)
  
  # Define method to apply the channel to a transmitted signal
  def transmit(self, signal: ComplexArray, fft_size: int) -> StaticMultipathResult:
    # Define signal as an array of complex numbers
    signal = np.asarray(signal, dtype=np.complex128)

    # Check if the signal dimension is not equal to 1
    if signal.ndim != 1:
      raise ValueError("Signal must be a 1D array.")
    
    # Check if the signal size is less than or equal to 0
    if signal.size == 0:
      raise ValueError("Signal must have at least one sample.")
    
    # Define impulse response
    impulse_response = self.impulse_response

    # Calculate full output
    full_output = np.convolve(signal, impulse_response, mode="full")

    # Preserve the original waveform duration
    received_signal = full_output[:signal.size]

    # Return static multipath result with received signal, impulse response, and frequency response
    return StaticMultipathResult(
      received_signal=received_signal,
      impulse_response=impulse_response,
      frequency_response=self.frequency_response(fft_size=fft_size)
    )