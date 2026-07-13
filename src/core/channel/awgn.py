'''
Additive White Gaussian Noise (AWGN) channel model implementation.
'''
from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np 

# Define complex number array
ComplexArray = NDArray[np.complex128]

# Define a dataclass for the AWGN channel model
@dataclass(frozen=True, slots=True)
class AWGNChannel:
  '''
  Additive White Gaussian Noise (AWGN) channel model.
  This class simulates the effect of an AWGN channel on a transmitted signal.
  '''
  ebn0_db: float  # Energy per bit to noise power spectral density ratio in dB
  bits_per_symbol: int = 2  # Number of bits per symbol for modulation
  symbol_energy: float = 1.0  # Energy of the transmitted symbol

  # Define a method to add AWGN noise to the transmitted signal
  def __post_init__(self) -> None:
    if self.bits_per_symbol <= 0:
      raise ValueError("Bits per symbol must be a positive integer.")
    
    if self.symbol_energy <= 0.0:
      raise ValueError("Symbol energy must be a positive value.")
  
  # Define property for energy per bit to noise power spectral density ratio in linear scale
  @property
  def ebn0_linear(self) -> float:
    '''Convert Eb/N0 from dB to linear scale.'''
    return float(10 ** (self.ebn0_db / 10))
  
  # Define property for bit energy based on symbol energy and bits per symbol
  @property
  def bit_energy(self) -> float:
    '''Return energy per bit E_b.'''
    return self.symbol_energy / self.bits_per_symbol
  
  # Define property for noise variance based on Eb/N0 and symbol energy
  @property
  def noise_variance(self) -> float:
    '''Total variance of complex AWGN: E[|n|²] = N0 = Eb / (Eb/N0).'''
    return self.bit_energy / self.ebn0_linear
  
  # Define method to transmit a signal through the AWGN channel
  def transmit(self, signal: ComplexArray, rng: np.random.Generator) -> ComplexArray:
    '''Add AWGN noise to the transmitted signal.'''
    signal = np.asarray(signal, dtype=np.complex128)

    # Check the signal dimensions
    if signal.ndim != 1:
      raise ValueError("Input signal must be a 1D array of complex symbols.")
    
    # Check the signal size
    if signal.size == 0:
      raise ValueError("Input signal must not be empty.")
    
    # Define standard deviation of the noise based on noise variance
    std_dev = float(np.sqrt(self.noise_variance / 2.0))

    # Define noise as a complex Gaussian random variable
    noise = std_dev * (rng.standard_normal(signal.shape) + 1j * rng.standard_normal(signal.shape))

    # Return the transmitted signal with added noise
    return signal + noise