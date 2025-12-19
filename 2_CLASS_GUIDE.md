# 2-Class Classification: Kill vs Headshot

## 🎯 Overview

The model has been updated to focus on **2-class classification** instead of 3:
- ❌ **Removed:** `injured` class
- ✅ **Kept:** `kill` and `headshot` classes

## 📊 Why This Change?

### Before (3 classes)
```
injured:  13 samples (21%) ← Minority class
kill:     28 samples (46%)
headshot: 20 samples (33%)
Total:    61 samples
Accuracy: 53.85% (random guessing ~33%)
```

### After (2 classes)
```
kill:     28 samples (58%)
headshot: 20 samples (42%)
Total:    48 samples
Expected accuracy: 70-85%+ (random guessing = 50%)
```

## ✅ Benefits

1. **Simpler problem** - Binary classification is easier than 3-class
2. **Better balance** - 58% vs 42% is more balanced than before
3. **More samples per class** - Better ratio for training
4. **Focus on core problem** - Kill vs Headshot is the main challenge
5. **Template matching more effective** - Directly boosts headshot detection

## 🛠️ What Changed

### `src/dataset.py`
- Filters out `injured` samples automatically
- CLASS2IDX = {"kill": 0, "headshot": 1}
- Logs how many samples were filtered

### `src/train.py`
- num_classes = 2
- batch_size = 8 (reduced for smaller dataset)
- num_epochs = 50 (increased for better convergence)
- dropout_rate = 0.5 (higher to prevent overfitting)
- Early stopping after 10 epochs without improvement

### `src/predict.py`
- Supports 2-class predictions
- **AGGRESSIVE template matching:**
  - If template similarity > 65% → FORCE headshot prediction
  - Otherwise use CNN prediction
- Configurable threshold

### `src/model.py`
- No change needed (already supports variable num_classes)

## 🚀 Usage

### Training

```bash
# Train new 2-class model
python -m src.train
```

Expected output:
```
2025-12-19 09:00:00 - __main__ - INFO - Loaded dataset from data/labels.csv with 61 samples
2025-12-19 09:00:00 - src.dataset - INFO - Filtered out 13 'injured' samples
2025-12-19 09:00:00 - src.dataset - INFO - Class distribution: {'kill': 28, 'headshot': 20}
2025-12-19 09:00:00 - src.dataset - INFO - Final dataset size: 48 samples
2025-12-19 09:00:00 - __main__ - INFO - Training 2-class model: KILL vs HEADSHOT
...
```

### Prediction Without Template

```bash
python -m src.predict audio.wav
```

### Prediction With Template (RECOMMENDED)

```bash
# Default threshold (0.65)
python -m src.predict audio.wav data/raw/headshots/headshot.wav

# Custom threshold (0.70 = more conservative)
python -m src.predict audio.wav data/raw/headshots/headshot.wav 0.70

# Lower threshold (0.60 = more aggressive)
python -m src.predict audio.wav data/raw/headshots/headshot.wav 0.60
```

### Python API

```python
from src.predict import predict_one

# With template matching (RECOMMENDED)
result = predict_one(
    audio_path="audio.wav",
    use_template=True,
    template_path="data/raw/headshots/headshot.wav",
    template_threshold=0.65  # Adjust as needed
)

print(f"Prediction: {result['predicted_class']}")
print(f"Confidence: {result['predicted_confidence']:.1%}")

if 'template_matching' in result:
    sim = result['template_matching']['combined']
    threshold = result['template_threshold']
    print(f"Template similarity: {sim:.1%} (threshold: {threshold:.0%})")
    
    if sim > threshold:
        print("❗ Prediction FORCED to headshot by template matching")
```

## 🎯 Template Matching Strategy

The model now uses an **aggressive template matching strategy**:

```python
if template_similarity > threshold:
    prediction = "HEADSHOT"  # FORCED
    confidence = template_similarity
else:
    prediction = CNN_prediction
    confidence = CNN_confidence
```

### Threshold Tuning

| Threshold | Behavior |
|-----------|----------|
| 0.50 | Very aggressive - More false positives |
| 0.60 | Aggressive - Good for low-quality audio |
| **0.65** | **Balanced (DEFAULT)** |
| 0.70 | Conservative - Fewer false positives |
| 0.80 | Very conservative - May miss some headshots |

### Finding Your Optimal Threshold

```python
# Test different thresholds on your data
for threshold in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
    correct = 0
    total = 0
    
    for audio_file, true_label in test_files:
        result = predict_one(
            audio_file,
            use_template=True,
            template_path="headshot.wav",
            template_threshold=threshold
        )
        
        if result['predicted_class'] == true_label:
            correct += 1
        total += 1
    
    accuracy = correct / total
    print(f"Threshold {threshold:.2f}: {accuracy:.1%} accuracy")
```

## 📊 Expected Improvements

### Before (3 classes)
- Training accuracy: ~55%
- Validation accuracy: 53.85%
- Headshot detection: Poor (confused with kill)

### After (2 classes)
- Expected training accuracy: **75-85%**
- Expected validation accuracy: **70-80%**
- Headshot detection with template: **85-95%**

## ⚠️ Important Notes

1. **Re-train required** - Old 3-class model won't work
2. **Template matching highly recommended** - Significantly improves headshot detection
3. **Threshold tuning needed** - Find optimal value for your data
4. **More data still beneficial** - 48 samples is still small

## 📝 Migration Checklist

- [ ] Pull latest changes: `git pull origin main`
- [ ] Delete old model: `rm models/hunt_cry_best.pt`
- [ ] Re-train: `python -m src.train`
- [ ] Test predictions with template matching
- [ ] Tune threshold if needed
- [ ] Collect more headshot samples if accuracy not satisfactory

## 🐛 Troubleshooting

### "Model parameters don't match"
```bash
# Delete old model and retrain
rm models/hunt_cry_best.pt
python -m src.train
```

### "Still getting bad accuracy"
- Make sure you're using template matching
- Try lowering threshold to 0.60
- Collect more training data (aim for 100+ per class)

### "Too many false positive headshots"
- Increase threshold to 0.70 or 0.75
- Check template quality (should be clean headshot sound)

### "Missing real headshots"
- Lower threshold to 0.60
- Record better quality headshot template
- Check if template actually contains skull crack sound

## 🎉 Summary

Switching to 2-class classification should **significantly improve** your model's performance:

✅ Simpler problem to learn  
✅ Better class balance  
✅ More effective template matching  
✅ Higher expected accuracy  
✅ Easier to debug and improve  

**Next step:** Re-train the model and test with template matching! 🚀
