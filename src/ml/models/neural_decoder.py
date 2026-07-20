'''
1D-CNN neural decoder for convolutionally coded QPSK-OFDM over static multipath and AWGN.
'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define function to build 1D neural decoder
def build_neural_decoder(trellis_steps: int = 1000) -> keras.Model:
  '''
  Build a sequence-to-sequence neural decoder.
  Input:
    Shape (trellis_steps, 2)
    Two coded-bit LLRs for each trellis step.
  Output:
    Shape (trellis_steps, 1)
    One information-bit logit for each trellis step.
  '''
  # Check if trellis_steps is a positive integer
  if trellis_steps <= 0:
    raise ValueError("Trellis steps must be a positive integer.")
  
  # Define the input layer for the neural decoder
  inputs = keras.Input(shape=(trellis_steps, 2), name="coded_bit_llrs")

  # Local pattern extraction using 1D convolutional layers
  x = layers.Conv1D(
    filters=32, kernel_size=5,
    padding="same", activation="relu",
    name="conv_1")(inputs)
  
  x = layers.Conv1D(
    filters=64, kernel_size=7,
    padding="same", activation="relu",
    name="conv_2")(x)
  
  x = layers.Conv1D(
    filters=64, kernel_size=9,
    padding="same", activation="relu",
    name="conv_3")(x)
  
  # Residual like refinement
  x = layers.Conv1D(
    filters=32, kernel_size=5,
    padding="same", activation="relu",
    name="conv_4")(x)
  
  # Bit logits prediction with no sigmoid
  bit_logits = layers.Conv1D(
    filters=1, kernel_size=1,
    padding="same", activation=None,
    name="bit_logits")(x)
  
  # Define the neural decoder model
  model = keras.Model(inputs=inputs, outputs=bit_logits, name="neural_convolutional_decoder")

  # Compile the model with Adam optimizer and binary cross-entropy loss
  model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[
      keras.metrics.BinaryAccuracy(name="bit_accuracy", threshold=0.0)]
  )

  # Return the compiled neural decoder model
  return model