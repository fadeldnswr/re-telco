"""
Train the neural convolutional-code decoder.
"""

from pathlib import Path
import sys
import tensorflow as tf

from src.ml.datasets.coded_llr_datasets import CodedLLRGenerator
from src.ml.models.neural_decoder import build_neural_decoder
from src.exception.exception import CustomException

# Define main function to train the neural decoder
def main() -> None:
  try:
    # Define training parameters
    tf.keras.utils.set_random_seed(42)

    # Define output directory for saving the trained model
    output_dir = Path("results/models")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define training parameters
    training_gen = CodedLLRGenerator(
      information_bits_per_frame=1006,
      batch_size=32,
      maximum_ebn0_db=8.0,
      minimum_ebn0_db=-2.0,
      random_seed=42,
    )

    # Define validation parameters
    validation_generator = CodedLLRGenerator(
      information_bits_per_frame=1_006,
      batch_size=32,
      maximum_ebn0_db=8.0,
      minimum_ebn0_db=-2.0,
      random_seed=10_000,
    )

    # Convert the generators to TensorFlow datasets
    train_dataset = training_gen.to_tf_dataset()
    validation_dataset = validation_generator.to_tf_dataset()

    # Build the neural decoder model
    model = build_neural_decoder(trellis_steps=training_gen.trellis_steps)
    model.summary()

    # Define callbacks
    callbacks = [
      tf.keras.callbacks.ModelCheckpoint(
        filepath=output_dir / "best_decoder.keras",
        monitor="val_loss",
        save_best_only=True,
        mode="min",
      ),
      tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
      ),
      tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        minimum_delta=1e-4,
        min_lr=1e-6,
      )
    ]

    # Train the model
    model.fit(
      train_dataset,
      validation_data=validation_dataset,
      epochs=30,
      callbacks=callbacks,
      steps_per_epoch=100,
      validation_steps=20,
    )

    # Save the final trained model
    final_model_path = output_dir / "final_decoder.keras"
    model.save(final_model_path)
    print(f"Final trained model saved to: {final_model_path}")
  except CustomException as e:
    raise CustomException(e, sys)

# Run the main function if this script is executed directly
if __name__ == "__main__":
  main()