from pathlib import Path
import argparse
import shutil
import sys
from typing import Optional
import tensorflow as tf


def keras2model(filepath: str, outdir_path: Optional[str] = None, overwrite: bool = False) -> str:
    """
    Converts a Keras model file (.keras) to a SavedModel directory ready for TensorFlow Serving.

    :param filepath: The path to the input Keras model file.
    :param outdir_path: The output directory path for the SavedModel. If None, defaults to <filepath>_saved_model.
    :param overwrite: If True, overwrite the output directory if it exists.

    :raises FileNotFoundError: If no Keras model file is found.
    :raises FileExistsError: If the output directory exists and overwrite is False.
    :raises RuntimeError: If there are issues loading or saving the model.

    """
    p_keras = Path(filepath)
    if not p_keras.exists():
        raise FileNotFoundError(f"The keras {filepath} file was not found.")

    if outdir_path is None:
        outdir_path = str(p_keras.with_suffix('')) + '_saved_model'

    p_export = Path(outdir_path)

    if p_export.exists():
        if overwrite:
            shutil.rmtree(p_export)
        else:
            raise FileExistsError(f"The outdir path {outdir_path} already exists (use overwrite=True to overwrite).")

    p_export.parent.mkdir(parents=True, exist_ok=True)

    # Load the Keras model
    try:
        model: tf.keras.Model = tf.keras.models.load_model(str(p_keras), compile=False)
    except Exception as e:
        raise RuntimeError(f"Cannot load the keras {p_keras} file: {e}")

    # Save to SavedModel format
    try:
        # The default conversion.
        model.export(str(p_export))
    except Exception as e:
        # An alternative method if the default fails.
        try:
            model.export(str(p_export), include_optimizer=False)
        except Exception as e2:
            raise RuntimeError(f"Cannot save the model: {e}; fallback error: {e2}")

    return str(p_export)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Converts a Keras file into a SavedModel for Tensorflow Serving.')
    p.add_argument('--keras', '-k', required=True, help='Filepath for the input Keras model (.keras)')
    p.add_argument('--export-dir', '-o', required=False, default=None,
                   help='Output dir path for the saved model. (default: <keras_file>_saved_model)')
    p.add_argument('--overwrite', action='store_true', help='Overwrite the output directory if it exists.')
    return p


def main(argv=None):
    argv = argv or sys.argv[1:]
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    try:
        out = keras2model(args.keras, args.export_dir, overwrite=args.overwrite)
        print(f"Saved model correctly at: {out}")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == '__main__':
    tf.keras.config.enable_unsafe_deserialization()
    main()
