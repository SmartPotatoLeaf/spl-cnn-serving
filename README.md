# spl-cnn-serving

Utilities converting SPL-CNN models into SavedModel and Optimized TFLite Model.

# Dependencies

```bash
python -m pip install -r requirements.txt
```

# Usage

## Convert to SavedModel

```bash
python serving/script.py --k resources/models/SmartPotatoLeaf_ResNet50.keras -o ./serving/v1 --overwrite
```

## Convert to Optimized TFLite Model

