'''
OFDM modulation using IFFT and cyclie prefix addition
'''
import numpy as np
from numpy.typing import NDArray
from src.core.ofdm.config import OFDMConfig

# Define complex array type for better type hinting
ComplexArray = NDArray[np.complex128]

# Define OFDM modulator class
class OFDMModulator:
  def __init__(self, config: OFDMConfig):
    self._config = config
  
  # Define method to perform OFDM modulation
  def modulate(self, resource_grid: ComplexArray) -> ComplexArray:
    '''Apply IFFT and cyclic prefix to the resource grid to generate time-domain OFDM symbols.'''
    # Define resource grid as a numpy array of complex numbers
    resource_grid = np.asarray(resource_grid, dtype=np.complex128)

    # Define expected shape of the resource grid
    expected_shape = (
      self._config.symbols_per_subframe,
      self._config.fft_size
    )

    # Check if the resource grid shape is correct
    if resource_grid.shape != expected_shape:
      raise ValueError(f"Resource grid must have shape {expected_shape}, but got {resource_grid.shape}.")
    
    # Define waveform parts
    waveform_parts: list[ComplexArray] = []

    # Loop through each symbol in the resource grid
    for symbol_index in range(self._config.symbols_per_subframe):
      # Shift the symbol to center the DC component
      shifted_frequency_symbol = resource_grid[symbol_index]

      # Convert the frequency-domain symbol to time-domain using IFFT
      frequency_symbol = np.fft.ifftshift(shifted_frequency_symbol)

      # Define time-domain symbol using IFFT
      time_domain_symbol = np.fft.ifft(frequency_symbol, norm="ortho")

      # Define cp length
      cp_length = self._config.cyclic_prefix_length(symbol_index)
      cyclic_prefix = time_domain_symbol[-cp_length:]

      # Append the cyclic prefix and time-domain symbol to the waveform parts
      waveform_parts.append(
        np.concatenate(
          [cyclic_prefix, time_domain_symbol]
        )
      )
    
    # Define the final waveform by concatenating all waveform parts
    waveform = np.concatenate(waveform_parts)

    # Check if the waveform length is correct
    if waveform.size != self._config.samples_per_subframe:
      raise ValueError(f"Waveform length must be {self._config.samples_per_subframe}, but got {waveform.size}.")
    
    return waveform.astype(np.complex128)