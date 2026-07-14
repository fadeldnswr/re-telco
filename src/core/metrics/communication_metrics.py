'''
Communication metrics module for the digital-comms project.
This module provides functionality to compute and analyze communication metrics
'''
import numpy as np
from numpy.typing import NDArray
from scipy.special import erfc

# Define BitArray and ComplexArray types for better type hinting
BitArray = NDArray[np.uint8]
ComplexArray = NDArray[np.complex128]

# Define function to calculate Bit Error Rate (BER)
def calculate_ber(transmitted_bits: BitArray, recovered_bits: BitArray) -> float:
  # Define transmitted and recovered bits
  transmitted_bits = np.asarray(transmitted_bits, dtype=np.uint8)
  recovered_bits = np.asarray(recovered_bits, dtype=np.uint8)

  # Check its shape
  if transmitted_bits.shape != recovered_bits.shape:
    raise ValueError("Transmitted and recovered bits must have the same shape.")
  
  # Check transmitted bits size
  if transmitted_bits.size == 0:
    raise ValueError("Transmitted bits array is empty.")  
  
  # Calculate the number of bit errors
  number_of_errors = np.count_nonzero(
    transmitted_bits != recovered_bits
  )

  # Calculate the Bit Error Rate (BER)
  return float(number_of_errors) / transmitted_bits.size

# Define function to calculate theoritical qpsk bit error rate
def theoritical_qpsk_ber(ebn0_db: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
  '''Theoritical QPSK Bit Error Rate (BER) calculation based on Eb/N0 in dB.'''
  # Define Eb/N0 in linear scale
  ebn0_db_array = np.asarray(ebn0_db, dtype=np.float64)
  ebn0_linear = 10.0 ** (ebn0_db_array / 10.0)

  # Calculate the theoretical BER for QPSK using the complementary error function
  ber = 0.5 * erfc(np.sqrt(ebn0_linear))

  # Check the dimension of the input and return the appropriate type
  if np.ndim(ebn0_db) == 0:
    return float(ber)
  
  # If the input is an array, return the array of BER values
  return ber

# Define function to calculate avg symbol energy
def average_symbol_energy(symbols: ComplexArray) -> float:
  # Define symbols as a numpy array of complex numbers
  symbols = np.asarray(symbols, dtype=np.complex128)

  # Check the symbols size
  if symbols.size == 0:
    raise ValueError("Symbols array cannot be empty.")
  
  # Calculate the average symbol energy
  return float(np.mean(np.abs(symbols) ** 2))

# Define function to calculate evm (Error Vector Magnitude) and rms (Root Mean Square) error
def calculate_evm_rms(reference_symbols: ComplexArray, measured_symbols: ComplexArray) -> float:
  # Define reference and measured symbols as numpy arrays of complex numbers
  reference_symbols = np.asarray(reference_symbols, dtype=np.complex128)
  measured_symbols = np.asarray(measured_symbols, dtype=np.complex128)

  # Check the shape of the reference and measured symbols
  if reference_symbols.shape != measured_symbols.shape:
    raise ValueError("Reference and measured symbols must have the same shape.")
  
  # Calculate the reference symbol power
  reference_power = np.mean(
    np.abs(reference_symbols) ** 2
  )

  # Check if the reference power is zero to avoid division by zero
  if reference_power == 0.0:
    raise ValueError("Reference symbols have zero power, cannot compute EVM.")
  
  # Calculate the error power
  error_power = np.mean(
    np.abs(reference_symbols - measured_symbols) ** 2
  )

  # Calculate the EVM (Error Vector Magnitude)
  return float(np.sqrt(error_power / reference_power))