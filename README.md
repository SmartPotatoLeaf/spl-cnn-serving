# spl-cnn-serving

Utilities converting SPL-CNN models into SavedModel and Optimized TFLite Model.

# Dependencies

```bash
python -m pip install -r requirements.txt
```

# Usage

## Convert to SavedModel

```bash
python serving/script.py -k resources/models/SmartPotatoLeaf_ResNet50.keras -o ./serving/v1 --overwrite
```

## Convert to Optimized TFLite Model

### Basic conversion with float16 quantization (recommended)
```bash
python tflite/script.py -k resources/models/SmartPotatoLeaf_ResNet50.keras -opt float16 --overwrite
```

### Other optimization levels

**Default (no optimization):**
```bash
python tflite/script.py -k resources/models/SmartPotatoLeaf_ResNet50.keras -opt default
```

**Dynamic range int8 quantization:**
```bash
python tflite/script.py -k resources/models/SmartPotatoLeaf_ResNet50.keras -opt int8 --overwrite
```

**Full integer quantization (smallest size) - WITH representative data:**
```bash
python tflite/script.py -k resources/models/SmartPotatoLeaf_ResNet50.keras -opt int8_full -r path/to/validation_images --overwrite
```

**Full integer quantization (smallest size) - WITHOUT representative data (not recommended):**
```bash
python tflite/script.py -k resources/models/SmartPotatoLeaf_ResNet50.keras -opt int8_full --overwrite
```

### Optimization Levels Explained

- **default**: No quantization, full float32 precision. Largest size, highest accuracy.
- **float16** (recommended): Float16 quantization. ~50% size reduction with minimal accuracy loss. Best for most Android devices.
- **int8**: Dynamic range quantization. Weights are int8, activations are float. Good balance between size and performance.
- **int8_full**: Full integer quantization. Smallest size, fastest inference. **Requires representative dataset** for calibration to maintain accuracy.

### About Representative Data (for int8_full)

The **representative dataset** is a collection of real images used to calibrate the quantization ranges. You should use:

1. **Images from your validation/test set** (recommended) - These are images that were NOT used during training but are representative of real-world data.
2. **A subset of training images** - If you don't have validation images available.
3. **Real production images** - If available, these are ideal.

**Example structure:**
```
validation_images/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   └── ...
└── ...
```

The script will automatically find all `.jpg`, `.jpeg`, `.png`, and `.bmp` images in the specified directory and use up to 200 images for calibration. More diverse images = better calibration = better accuracy.

**Why is this important?**
- Without representative data, the quantizer uses random values, which can significantly reduce model accuracy.
- With real images, the quantizer learns the actual data distribution and maintains accuracy while reducing model size by ~75%.

All optimizations are configured for Android minSDK 22+ (Android 5.1+) compatibility.

## Working with Multi-Task Segmentation Models

This toolkit is optimized for **multi-task segmentation models** (models with multiple outputs). The script automatically detects and handles:

- **Multiple outputs**: Different segmentation tasks (e.g., classification + segmentation masks)
- **Different output types**: Each output can have different quantization settings
- **Preserved accuracy**: For multi-output models in `int8_full` mode, outputs are kept as float for better accuracy

### Inspecting the TFLite Model

After conversion, you can inspect the model to verify its structure:

```bash
python tflite/inspect_tflite.py -m path/to/model.tflite
```

For detailed information including quantization statistics:
```bash
python tflite/inspect_tflite.py -m path/to/model.tflite --verbose
```

This will show:
- Input/output shapes and types
- Quantization parameters
- Model size
- Tensor statistics (in verbose mode)

### Preprocessing Notes for Segmentation Models

The representative dataset images are preprocessed with normalization to [0, 1] range (`pixel_value / 255.0`). 

**If your model uses different preprocessing** (e.g., ImageNet mean/std normalization), you should modify the preprocessing in `tflite/script.py` in the `representative_dataset_gen()` function to match your training preprocessing.

Common preprocessing strategies:
- **[0, 1] normalization**: `img / 255.0` (default in script)
- **ImageNet normalization**: `(img / 255.0 - mean) / std` where mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- **[-1, 1] normalization**: `(img / 127.5) - 1.0`

