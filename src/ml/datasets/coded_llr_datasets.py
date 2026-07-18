'''
Synthetic dataset generator for neural convolutional-code decoding.
Each training sample contains:
  Input:
    Soft QPSK LLR pairs with shape (trellis_steps, 2)

  Target:
    Original information bits plus zero-tail bits with shape
    (trellis_steps, 1)
'''
from collections.abc import Iterator

import numpy as np
import tensorflow as tf
from numpy.typing import NDArray

from src.core.channel.awgn import AWGNChannel
from src.core.coding.convolutional import ConvolutionalEncoder
from src.core.modulation.qpsk import QPSK

# Define type aliases for arrays
FloatArray = NDArray[np.float32]

# Define CodedLLRDataGenerator class 
# Generate convolutionally encoded QPSK-AWGN samples dynamically.
# The dataset is generated during training, so a large dataset does
# not need to be permanently stored in memory.
class CodedLLRGenerator:
  nominal_code_rate: float = 1 / 2
  encoder_memory: int = 2

  # Constructor to initialize the generator with specified parameters
  def __init__(
    self, information_bits_per_frame: int = 1006, 
    batch_size: int = 32, maximum_ebn0_db: float = 8.0,
    minimum_ebn0_db: float = -2.0, random_seed: int = 42
    ) -> None:
    # Check if the number of information bits per frame is a positive integer
    if information_bits_per_frame <= 0:
      raise ValueError("Information bits per frame must be a positive integer.")
    
    # Check if the batch size is a positive integer
    if batch_size <= 0:
      raise ValueError("Batch size must be a positive integer.")
    
    # Check if the maximum Eb/N0 in dB is greater than the minimum Eb/N0 in dB
    if maximum_ebn0_db <= minimum_ebn0_db:
      raise ValueError("Maximum Eb/N0 (dB) must be greater than minimum Eb/N0 (dB).")
    
    # Define instance variables for the generator
    self._information_bits_per_frame = information_bits_per_frame
    self._batch_size = batch_size
    self._maximum_ebn0_db = maximum_ebn0_db
    self._minimum_ebn0_db = minimum_ebn0_db
    self._rng = np.random.default_rng(seed=random_seed)
    self._encoder = ConvolutionalEncoder()
    self._modulation = QPSK()
  
  # Define property for information bits per frame
  @property
  def information_bits_per_frame(self) -> int:
    return self._information_bits_per_frame
  
  # Define property for trellis steps
  @property
  def trellis_steps(self) -> int:
    return self._information_bits_per_frame + self.encoder_memory
  
  # Define coded bits per frame
  @property
  def coded_bits_per_frame(self) -> int:
    return 2 * self.trellis_steps
  
  # Define method to generate a batch of training samples
  def generate_batch(self) -> tuple[FloatArray, FloatArray]:
    """
    Generate one batch.
    Returns:
      features:
        Shape (batch, trellis_steps, 2)

      targets:
        Shape (batch, trellis_steps, 1)
    """
    # Generate features and targets for the batch
    features = np.empty((
      self._batch_size, self.trellis_steps, 2
    ), dtype=np.float32)
    targets = np.empty((
      self._batch_size, self.trellis_steps, 1
    ), dtype=np.float32)

    # Loop over the batch size to generate each sample
    for sample_index in range(self._batch_size):
      # Define information bits
      information_bits = self._rng.integers(
        low=0, high=2, size=self._information_bits_per_frame, dtype=np.uint8
      )

      # Define encoded bits using convolutional encoding
      encoded_bits = self._encoder.encode(information_bits, terminate=True)

      # Check if encoded bits size is not the same as coded bits per frame
      if encoded_bits.size != self.coded_bits_per_frame:
        raise ValueError(
          f"Encoded bits size {encoded_bits.size} does not match expected coded bits per frame {self.coded_bits_per_frame}."
        )
      
      # Map QPSK symbols from encoded bits
      qpsk_symbols = self._modulation.map(encoded_bits)

      # Train across multiple channel conditions
      ebn0_db = float(
        self._rng.uniform(
          self._minimum_ebn0_db,
          self._maximum_ebn0_db
        )
      )

      # Create an AWGN channel with the specified Eb/N0
      channel = AWGNChannel(
        ebn0_db=ebn0_db,
        bits_per_symbol=self._modulation.bits_per_symbol,
        symbol_energy=1.0,
        code_rate=self.nominal_code_rate
      )

      # Received symbols after passing through the AWGN channel
      received_symbols = channel.transmit(qpsk_symbols, self._rng)

      # Soft QPSK demapping of received symbols to LLRs
      llrs = self._modulation.soft_demap(
        symbols=received_symbols, 
        noise_variance=channel.noise_variance
      )

      # Two coded llrs belongs to one trellis step, so reshape to (trellis_steps, 2)
      features[sample_index] = llrs.reshape(self.trellis_steps, 2).astype(np.float32)

      # Original information bits plus zero-tail bits with shape (trellis_steps, 1)
      target_with_tail = np.concatenate((
        information_bits, np.zeros(
          self.encoder_memory, dtype=np.uint8
        ),
      ))
      targets[sample_index, :, 0] = target_with_tail.astype(np.float32)
    
    # Return the features and targets for the batch
    return features, targets
  
  # Define iterator method to yield batches indefinitely
  def iterator(self) -> Iterator[tuple[FloatArray, FloatArray]]:
    """
    Create an iterator that yields batches indefinitely.
    Yields:
      features:
        Shape (batch, trellis_steps, 2)

      targets:
        Shape (batch, trellis_steps, 1)
    """
    while True:
      yield self.generate_batch()
  
  # Define method to create a TensorFlow dataset from the generator
  def to_tf_dataset(self) -> tf.data.Dataset:
    """
      Convert the Python generator into a TensorFlow dataset.
    """
    # Define the output signature for the TensorFlow dataset
    dataset = tf.data.Dataset.from_generator(
      self.iterator,
      output_signature=(
        tf.TensorSpec(shape=(self._batch_size, self.trellis_steps, 2), dtype=tf.float32),
        tf.TensorSpec(shape=(self._batch_size, self.trellis_steps, 1), dtype=tf.float32)
      )
    )
    # Return the TensorFlow dataset
    return dataset.prefetch(
      tf.data.AUTOTUNE
    )