'''
Additive White Gaussian Noise (AWGN) channel model implementation.
'''
from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np 

# Define complex number array
ComplexArray = NDArray[np.complex128]

# Define dataclass for AWGN Waveform result
@dataclass(frozen=True, slots=True)
class AWGNWaveformResult:
  received_signal: ComplexArray
  noise: ComplexArray
  total_signal_energy: float
  bit_energy: float
  noise_variance: float
  noise_std_dev: float

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

  # Define method to transmit a signal and return detailed results
  def transmit_waveform(self, waveform: ComplexArray, number_of_information_bits: int, rng: np.random.Generator, energy_reference: ComplexArray | None = None) -> AWGNWaveformResult:
    '''Transmit a waveform through the AWGN channel and return detailed results.'''
    # Define waveform
    waveform = self._validate_signal(waveform)

    # Check if the number of information bits is positive
    if number_of_information_bits <= 0:
      raise ValueError("Number of information bits must be a positive integer.")

    # Check if the energy reference is provided and validate it
    if energy_reference is None:
      reference = waveform
    else:
      reference = self._validate_signal(energy_reference)
    
    # Calculate total signal energy
    total_signal_energy = float(
      np.sum(np.abs(reference) ** 2)
    )

    # Calculate waveform energy per bit and noise variance
    waveform_bit_energy = total_signal_energy / number_of_information_bits
    waveform_noise_variance = waveform_bit_energy / self.ebn0_linear

    # Calculate standard deviation of the noise based on noise variance
    standard_deviation = float(
      np.sqrt(waveform_noise_variance / 2.0)
    )

    # Define noise as a complex Gaussian random variable
    noise = standard_deviation * (
      rng.standard_normal(waveform.shape)
      + 1j * rng.standard_normal(waveform.shape)
    )

    return AWGNWaveformResult(
      received_signal=waveform + noise,
      noise=noise,
      total_signal_energy=total_signal_energy,
      bit_energy=waveform_bit_energy,
      noise_variance=waveform_noise_variance,
      noise_std_dev=standard_deviation
    )
  
  # Define static method to validate signal
  @staticmethod
  def _validate_signal(signal: ComplexArray) -> ComplexArray:
    # Define signal as a numpy array of complex numbers
    signal = np.asarray(signal, dtype=np.complex128)

    # Check the signal dimensions and size
    if signal.ndim != 1:
      raise ValueError("Input signal must be a 1D array of complex symbols.")
    if signal.size == 0:
      raise ValueError("Input signal must not be empty.")

    # Return the validated signal
    return signal