'''
QPSK modulation and demodulation implementation in Python.
This module provides functions to modulate and demodulate binary data using Quadrature Phase Shift Key
'''
from typing import Final
from numpy.typing import NDArray
import numpy as np

# Define constants for QPSK modulation
BitArray = NDArray[np.uint8]
ComplexArray = NDArray[np.complex128]

# Define the QPSK modulation function
class QPSK:
  '''
  Gray-coded QPSK mapper and hard-decision demapper.
    Mapping:
        00 -> (+1 + j) / sqrt(2)
        01 -> (+1 - j) / sqrt(2)
        10 -> (-1 + j) / sqrt(2)
        11 -> (-1 - j) / sqrt(2)
  '''
  # Define bits per symbol for QPSK
  bits_per_symbol: Final[int] = 2

  # Define the modulation method
  def map(self, bits: BitArray) -> ComplexArray:
    '''Map binary bits to QPSK symbols.'''
    bits = np.asarray(bits, dtype=np.uint8)

    # Check if the number of bits is even
    if bits.ndim != 1:
      raise ValueError("Input bits must be a 1D array.")
    
    # Check if the size is empty
    if bits.size == 0:
      raise ValueError("Input bits must not be empty.")
    
    # Check if the number of bits is even
    if bits.size % self.bits_per_symbol != 0:
      raise ValueError("Number of bits must be even for QPSK modulation.")
    
    # Check if the bits are binary (0 or 1)
    if not np.all(np.isin(bits, [0, 1])):
      raise ValueError("Input bits must be binary (0 or 1).")
    
    # Reshape bits into pairs for QPSK mapping
    bit_pairs = bits.reshape(-1, 2).astype(np.int8)

    # Map bit pairs to QPSK symbols using Gray coding
    in_phase = 1.0 - 2 * bit_pairs[:, 0]  # Map first bit to in-phase component
    quadrature = 1.0 - 2 * bit_pairs[:, 1]  # Map second bit to quadrature component

    # Combine in-phase and quadrature components into complex symbols\
    symbols = (in_phase + 1j * quadrature) / np.sqrt(2)  # Normalize by sqrt(2)

    return symbols.astype(np.complex128)
  
  # Define the demodulation method
  def demap(self, symbols: ComplexArray) -> BitArray:
    '''Demap QPSK symbols back to binary bits using hard decision.'''
    symbols = np.asarray(symbols, dtype=np.complex128)

    # Check if the symbols are 1D
    if symbols.ndim != 1:
      raise ValueError("Input symbols must be a 1D array.")
    
    # Check if the symbols array is empty
    if symbols.size == 0:
      raise ValueError("Input symbols must not be empty.")
    
    # Perform hard decision demapping based on the sign of the real and imaginary parts
    recovered_bits = np.empty(
      symbols.size * self.bits_per_symbol, 
      dtype=np.uint8
    )

    # Demap in-phase component to first bit
    recovered_bits[0::2] = np.real(symbols) < 0  # First bit: 0 if real part >= 0, else 1
    recovered_bits[1::2] = np.imag(symbols) < 0  # Second bit: 0 if imag part >= 0, else 1

    # Return the recovered bits as a 1D array of uint8
    return recovered_bits

  # Define method to demap symbols to bits with soft decision (optional)
  def soft_demap(self, symbols: ComplexArray, noise_variance: float) -> NDArray[np.float64]:
    """
    Convert received QPSK symbols into bit log-likelihood ratios.
    LLR convention:
      positive LLR -> bit 0 is more likely
      negative LLR -> bit 1 is more likely
    Args:
      symbols:
        Received complex QPSK symbols.
      noise_variance:
        Total complex AWGN variance E[|n|^2]
    Returns:
      One LLR value for every transmitted coded bit.
    """
    # Define symbols as an array of complex numbers
    symbols = np.asarray(symbols, dtype=np.complex128)

    # Check its size and dimension
    if symbols.ndim != 1:
      raise ValueError("Input symbols must be a 1D array.")
    if symbols.size == 0:
      raise ValueError("Input symbols must not be empty.")
    
    # Check if noise variance is positive
    if noise_variance <= 0.0:
      raise ValueError("Noise variance must be a positive value.")
    
    # Define llr values based on the real and imaginary parts of the symbols
    llr = np.empty(symbols.size * self.bits_per_symbol, dtype=np.float64)

    # For symbols 
    # s = ((1 - 2b_I) + j(1 - 2b_Q)) / sqrt(2)
    # Positive real/imaginary component indicates bit 0.
    scale = (2.0 * np.sqrt(2.0)) / noise_variance
    llr[0::2] = scale * symbols.real  # LLR for first bit (in-phase)
    llr[1::2] = scale * symbols.imag  # LLR for second bit (quadrature)

    # Return the computed log-likelihood ratios as a 1D array of float64
    return llr