'''
OFDM demodulation using FFT and cyclic prefix removal
'''
import numpy as np
from numpy.typing import NDArray
from src.core.ofdm.config import OFDMConfig

# Define complex array type for better type hinting
ComplexArray = NDArray[np.complex128]

# Define OFDM demodulator class
class OFDMDemodulator:
  def __init__(self, config: OFDMConfig):
    self._config = config

  # Define method to perform OFDM demodulation
  def demodulate(self, waveform: ComplexArray) -> ComplexArray:
    '''Remove cyclic prefix and apply FFT to the time-domain waveform to recover the resource grid.'''
    # Define waveform as a numpy array of complex numbers
    waveform = np.asarray(waveform, dtype=np.complex128)

    # Check if the waveform dimension is 1D
    if waveform.ndim != 1:
      raise ValueError("Waveform must be a 1D array.")
    
    # Check if the waveform size is correct
    if waveform.size != self._config.samples_per_subframe:
      raise ValueError(f"Waveform length must be {self._config.samples_per_subframe}, but got {waveform.size}.")
    
    # Define resource grid parts
    resource_grids = np.zeros(
      (
        self._config.symbols_per_subframe,
        self._config.fft_size
      ),
      dtype=np.complex128,
    )
    cursor = 0

    # Loop through each symbol in the waveform
    for symbol_index in range(self._config.symbols_per_subframe):
      # Define cp length
      cp_length = self._config.cyclic_prefix_length(symbol_index)

      # Define block length
      block_length = self._config.fft_size + cp_length

      # Define OFDM block by slicing the waveform
      ofdm_block = waveform[cursor:cursor + block_length]

      # Define time domain symbol by removing the cyclic prefix
      time_symbol = ofdm_block[cp_length:]

      # Convert the time-domain symbol to frequency-domain using FFT
      frequency_symbol = np.fft.fft(time_symbol, norm="ortho")

      # Store the frequency-domain symbol in the resource grid
      resource_grids[symbol_index] = np.fft.fftshift(frequency_symbol)

      # Update the cursor for the next symbol
      cursor += block_length
    
    # Check if the cursor has reached the end of the waveform
    if cursor != waveform.size:
      raise ValueError(f"Cursor position {cursor} does not match waveform length {waveform.size}.")
    
    # Return the recovered resource grid
    return resource_grids