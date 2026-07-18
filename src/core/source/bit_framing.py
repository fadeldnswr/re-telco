"""
Fixed-capacity bit framing for coded transmission.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# Define type alias for binary array
BitArray = NDArray[np.uint8]

# Define a dataclass to represent a bit frame with index, valid bit count, and bits
@dataclass(frozen=True, slots=True)
class BitFrame:
  index: int
  valid_bit_count: int
  bits: BitArray

# Define function to segment a bit stream into fixed-capacity bit frames
def segment_bits(payload_bits: BitArray, frame_capacity: int) -> list[BitFrame]:
  """
  Divide arbitrary-length payload bits into padded fixed-size frames.
  """
  # Define payload bits as a numpy array of type uint8
  payload_bits = np.asarray(payload_bits, dtype=np.uint8)

  # Check if the payload dimension is 1D
  if payload_bits.ndim != 1:
    raise ValueError("Payload bits must be a 1D array.")
  
  # Check if the frame capacity is a positive integer
  if frame_capacity <= 0:
    raise ValueError("Frame capacity must be a positive integer.")
  
  # Check if the payload bits size is zero
  if payload_bits.size == 0:
    raise ValueError("Payload bits must not be empty.")
  
  # Define list of frame to store the segmented bit frames
  frames: list[BitFrame] = []

  # Iterate through the payload bits in steps of frame capacity
  for frame_index, start in enumerate(range(0, payload_bits.size, frame_capacity)):
    # Define the end index for the current frame
    chunk = payload_bits[start:start + frame_capacity]

    # Create padded bits for the current frame
    padded_bits = np.zeros(frame_capacity, dtype=np.uint8)
    padded_bits[:chunk.size] = chunk

    # Append the current frame to the list of frames
    frames.append(BitFrame(
      index=frame_index,
      valid_bit_count=int(chunk.size),
      bits=padded_bits
    ))
  
  # Return the list of segmented bit frames
  return frames

# Define function to reconstruct the original bit stream from a list of bit frames
def reconstruct_payload_bits(recovered_frames: list[BitFrame], original_frames: list[BitFrame]) -> BitArray:
  """Remove frame padding and reconstruct the original payload length."""
  # Check if the length of recovered frames matches the length of original frames
  if len(recovered_frames) != len(original_frames):
    raise ValueError("Recovered frames length does not match original frames length.")
  
  # Define list to store valid chunks
  valid_chunks: list[BitArray] = []

  # Iterate through the recovered frames and original frames
  for recovered, metadata in zip(recovered_frames, original_frames, strict=True):
    # Define recovered frame as an array of type uint8
    recovered = np.asarray(recovered, dtype=np.uint8)

    # Check if the recovered frame dimension is 1D
    if recovered.ndim != 1:
      raise ValueError("Recovered frame must be a 1D array.")
    
    # Check if the recovered frame size matches the original frame size
    if recovered.size != metadata.bits.size:
      raise ValueError("Recovered frame size does not match original frame size.")
    
    # Append the valid chunk from the recovered frame to the list of valid chunks
    valid_chunks.append(recovered[:metadata.valid_bit_count])
  
  # Concatenate the valid chunks to reconstruct the original payload bits
  return np.concatenate(valid_chunks).astype(np.uint8)