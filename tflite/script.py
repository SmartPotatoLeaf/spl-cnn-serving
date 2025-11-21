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
    optimization_level: str = "default",
    representative_data_dir: Optional[str] = None
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
    :param representative_data_dir: Directory containing representative images for int8_full quantization.
                                    Should contain real images (from validation/test set).
                                    If None and int8_full is used, random data will be used (not recommended).

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

        # Display model information
        print(f"\nModel loaded successfully!")
        print(f"Model type: {'Multi-output (Multitask)' if isinstance(model.output, list) else 'Single-output'}")

        # Handle both single and multiple inputs
        if isinstance(model.input, list):
            print(f"Number of inputs: {len(model.input)}")
            for i, inp in enumerate(model.input):
                print(f"  Input {i}: {inp.shape}")
        else:
            print(f"Input shape: {model.input.shape}")

        # Handle both single and multiple outputs
        if isinstance(model.output, list):
            print(f"Number of outputs: {len(model.output)}")
            for i, out in enumerate(model.output):
                print(f"  Output {i}: {out.shape}")
        else:
            print(f"Output shape: {model.output.shape}")

        print()

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
                # Get input shape from model (handle both single and multiple inputs)
                if isinstance(model.input, list):
                    input_shape = model.input[0].shape  # Use first input
                else:
                    input_shape = model.input.shape

                # Extract dimensions (batch, height, width, channels)
                sample_shape = [1 if dim is None else dim for dim in input_shape]
                target_height, target_width = sample_shape[1], sample_shape[2]

                print(f"Calibration input shape: {sample_shape} (H={target_height}, W={target_width})")

                if representative_data_dir is not None:
                    # Load real images from directory
                    data_path = Path(representative_data_dir)
                    if not data_path.exists():
                        print(f"WARNING: Representative data directory not found: {representative_data_dir}")
                        print("WARNING: Using random data instead (not recommended)")
                        # Fallback to random data
                        for _ in range(100):
                            data = np.random.rand(*sample_shape).astype(np.float32)
                            yield [data]
                        return

                    # Find all image files
                    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
                    image_files = [f for f in data_path.rglob('*')
                                   if f.suffix.lower() in image_extensions]

                    if not image_files:
                        print(f"WARNING: No images found in {representative_data_dir}")
                        print("WARNING: Using random data instead (not recommended)")
                        for _ in range(100):
                            data = np.random.rand(*sample_shape).astype(np.float32)
                            yield [data]
                        return

                    print(f"Found {len(image_files)} images for calibration")
                    # Use up to 200 images for calibration (more is better but slower)
                    num_samples = min(200, len(image_files))

                    for i, img_path in enumerate(image_files[:num_samples]):
                        try:
                            # Load and preprocess image
                            img = tf.keras.preprocessing.image.load_img(
                                str(img_path),
                                target_size=(target_height, target_width)
                            )
                            img_array = tf.keras.preprocessing.image.img_to_array(img)

                            # Normalize to [0, 1] range
                            # For segmentation models, typically use [0, 1] normalization
                            # If your model uses different preprocessing (e.g., ImageNet normalization),
                            # you should adjust this accordingly
                            img_array = img_array / 255.0

                            # Add batch dimension
                            img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

                            yield [img_array]

                            if (i + 1) % 50 == 0:
                                print(f"  Processed {i + 1}/{num_samples} calibration images...")
                        except Exception as e:
                            print(f"  WARNING: Could not load image {img_path}: {e}")
                            continue

                    print(f"Calibration complete using {num_samples} images")
                else:
                    # No representative data provided - use random data (not recommended)
                    print("WARNING: No representative data directory provided!")
                    print("WARNING: Using random data for calibration (accuracy may be affected)")
                    print("RECOMMENDATION: Use --repr-data flag with a directory of real images")
                    for _ in range(100):
                        data = np.random.rand(*sample_shape).astype(np.float32)
                        yield [data]

            converter.representative_dataset = representative_dataset_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

            # For multi-output segmentation models, it's often better to keep outputs as float
            # to maintain accuracy, especially for pixel-wise predictions
            is_multi_output = isinstance(model.output, list)
            if is_multi_output:
                print("Multi-output model detected: keeping output types as float for better accuracy")
                # Don't set inference_input_type and inference_output_type for multi-output models
                # This allows the converter to handle each output appropriately
            else:
                converter.inference_input_type = tf.uint8
                converter.inference_output_type = tf.uint8

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
  
  # Full integer quantization with representative data (best optimization)
  python tflite/script.py -k resources/models/SmartPotatoLeaf_ResNet50.keras -opt int8_full -r path/to/validation_images --overwrite
  
  # Dynamic range int8 quantization (no representative data needed)
  python tflite/script.py -k resources/models/SmartPotatoLeaf_ResNet50.keras -opt int8 --overwrite

Note: For int8_full, use --repr-data to point to a directory with validation/test images.
      This ensures better calibration and maintains model accuracy.
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
        '--repr-data', '-r',
        required=False,
        default=None,
        help='Directory containing representative images for int8_full calibration. '
             'Should contain real images from validation/test set. '
             'Required for best results with int8_full optimization.'
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
            optimization_level=args.optimization,
            representative_data_dir=args.repr_data
        )
        print(f"✓ TFLite model successfully created at: {out}")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == '__main__':
    tf.keras.config.enable_unsafe_deserialization()
    main()

