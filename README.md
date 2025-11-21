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

**Full integer quantization (smallest size):**
```bash
python tflite/script.py -k resources/models/SmartPotatoLeaf_ResNet50.keras -opt int8_full -o tflite/model_optimized.tflite --overwrite
```

### Optimization Levels Explained

- **default**: No quantization, full float32 precision. Largest size, highest accuracy.
- **float16** (recommended): Float16 quantization. ~50% size reduction with minimal accuracy loss. Best for most Android devices.
- **int8**: Dynamic range quantization. Weights are int8, activations are float. Good balance between size and performance.
- **int8_full**: Full integer quantization. Smallest size, fastest inference. Requires representative dataset for calibration (uses random data by default).

All optimizations are configured for Android minSDK 22+ (Android 5.1+) compatibility.

