from pathlib import Path
import argparse
import sys
from typing import Optional
import tensorflow as tf
import numpy as np


def keras2tflite(
    filepath: str,
    outfile_path: Optional[str] = None,
    overwrite: bool = False,
    optimization_level: str = "default"
) -> str:
    """
    Converts a Keras model file (.keras) to an optimized TFLite model for mobile devices.

    Optimized for Android devices with minSDK 22 (Android 5.1+).

    :param filepath: The path to the input Keras model file.
    :param outfile_path: The output file path for the TFLite model. If None, defaults to <filepath>.tflite.
    :param overwrite: If True, overwrite the output file if it exists.
    :param optimization_level: Optimization level: 'default', 'float16', 'int8', or 'int8_full'.
                               - 'default': No quantization, full float32
                               - 'float16': Float16 quantization (smaller, faster, minimal accuracy loss)
                               - 'int8': Dynamic range quantization (good balance)
                               - 'int8_full': Full integer quantization (smallest, requires representative dataset)

    :raises FileNotFoundError: If no Keras model file is found.
    :raises FileExistsError: If the output file exists and overwrite is False.
    :raises RuntimeError: If there are issues loading or converting the model.
    """
    p_keras = Path(filepath)
    if not p_keras.exists():
        raise FileNotFoundError(f"The keras file {filepath} was not found.")

    if outfile_path is None:
        outfile_path = str(p_keras.with_suffix('.tflite'))

    p_export = Path(outfile_path)

    if p_export.exists():
        if overwrite:
            p_export.unlink()
        else:
            raise FileExistsError(
                f"The output file {outfile_path} already exists (use overwrite=True to overwrite)."
            )

    p_export.parent.mkdir(parents=True, exist_ok=True)

    # Load the Keras model
    try:
        print(f"Loading Keras model from: {p_keras}")
        model: tf.keras.Model = tf.keras.models.load_model(str(p_keras), compile=False)
        print(f"Model loaded successfully. Input shape: {model.input_shape}")
    except Exception as e:
        raise RuntimeError(f"Cannot load the keras file {p_keras}: {e}")

    # Create TFLite converter
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        # Configure optimizations based on level
        if optimization_level == "default":
            print("Using default conversion (float32, no quantization)")
            # No specific optimizations, just convert as-is
            pass

        elif optimization_level == "float16":
            print("Using float16 quantization (recommended for most mobile devices)")
            # Float16 quantization - good balance between size and accuracy
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]

        elif optimization_level == "int8":
            print("Using dynamic range int8 quantization")
            # Dynamic range quantization - weights are int8, activations are float
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

        elif optimization_level == "int8_full":
            print("Using full integer quantization (int8)")
            # Full integer quantization - requires representative dataset
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

            # Generate representative dataset for full int8 quantization
            def representative_dataset_gen():
                # Get input shape from model
                input_shape = model.input_shape
                # Generate sample data (you should replace this with real data samples)
                for _ in range(100):
                    # Generate random data matching the input shape
                    # Note: batch dimension is None, so we use 1
                    sample_shape = [1 if dim is None else dim for dim in input_shape]
                    data = np.random.rand(*sample_shape).astype(np.float32)
                    yield [data]

            converter.representative_dataset = representative_dataset_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8  # or tf.int8
            converter.inference_output_type = tf.uint8  # or tf.int8

        else:
            raise ValueError(
                f"Invalid optimization_level: {optimization_level}. "
                f"Choose from: 'default', 'float16', 'int8', 'int8_full'"
            )

        # Additional optimizations for mobile devices
        # Enable experimental optimizations
        converter.experimental_new_converter = True

        # Optimize for mobile devices (Android SDK 22+)
        # Set supported operations to use only built-in TFLite ops (no TensorFlow ops)
        if optimization_level != "int8_full":
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS  # Use only TFLite built-in ops
            ]

        # Enable other optimizations
        # Allow lower precision for faster inference on mobile
        converter.allow_custom_ops = False  # Strict mode - only TFLite ops

        print("Converting model to TFLite format...")
        tflite_model = converter.convert()

        # Save the model
        with open(str(p_export), 'wb') as f:
            f.write(tflite_model)

        # Print model statistics
        original_size = p_keras.stat().st_size
        converted_size = p_export.stat().st_size
        reduction_percent = ((original_size - converted_size) / original_size) * 100

        print(f"\n{'='*60}")
        print(f"Conversion successful!")
        print(f"Original model size: {original_size / (1024*1024):.2f} MB")
        print(f"TFLite model size: {converted_size / (1024*1024):.2f} MB")
        print(f"Size reduction: {reduction_percent:.1f}%")
        print(f"Output saved to: {p_export}")
        print(f"{'='*60}\n")

    except Exception as e:
        raise RuntimeError(f"Cannot convert the model to TFLite: {e}")

    return str(p_export)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Converts a Keras model to an optimized TFLite model for mobile devices (Android minSDK 22+).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Optimization levels:
  default     - No quantization, full float32 precision
  float16     - Float16 quantization (recommended - smaller size, minimal accuracy loss)
  int8        - Dynamic range int8 quantization (good balance)
  int8_full   - Full integer quantization (smallest size, requires representative data)

Examples:
  # Basic conversion with float16 quantization (recommended)
  python tflite/script.py -k resources/models/SmartPotatoLeaf_ResNet50.keras -opt float16
  
  # Full integer quantization for maximum optimization
  python tflite/script.py -k resources/models/SmartPotatoLeaf_ResNet50.keras -opt int8_full -o model_int8.tflite --overwrite
        """
    )
    p.add_argument(
        '--keras', '-k',
        required=True,
        help='Filepath for the input Keras model (.keras)'
    )
    p.add_argument(
        '--output', '-o',
        required=False,
        default=None,
        help='Output file path for the TFLite model. (default: <keras_file>.tflite)'
    )
    p.add_argument(
        '--optimization', '-opt',
        required=False,
        default='float16',
        choices=['default', 'float16', 'int8', 'int8_full'],
        help='Optimization level (default: float16)'
    )
    p.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite the output file if it exists.'
    )
    return p


def main(argv=None):
    argv = argv or sys.argv[1:]
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    try:
        out = keras2tflite(
            args.keras,
            args.output,
            overwrite=args.overwrite,
            optimization_level=args.optimization
        )
        print(f"✓ TFLite model successfully created at: {out}")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == '__main__':
    tf.keras.config.enable_unsafe_deserialization()
    main()

