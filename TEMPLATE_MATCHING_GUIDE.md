# Template Matching for Headshot Detection

## 🎯 Overview

Template matching is a technique to improve headshot detection by comparing input audio to a reference headshot recording. This is particularly useful for detecting the subtle "skull crack" sound that differentiates headshots from regular kills.

## 🔧 How It Works

### Three Similarity Metrics

1. **Cross-Correlation (30% weight)**
   - Measures temporal pattern similarity
   - Good for detecting the distinctive skull impact sound
   - Range: 0-1 (higher = more similar)

2. **Mel-Spectrogram Similarity (40% weight)**
   - Compares spectral content
   - Captures overall acoustic characteristics
   - Range: 0-1 (higher = more similar)

3. **MFCC Similarity (30% weight)**
   - Mel-Frequency Cepstral Coefficients
   - Mimics human perceptual characteristics
   - Range: 0-1 (higher = more similar)

### Combined Score

Final similarity = 0.3 × cross_correlation + 0.4 × mel_spectrogram + 0.3 × mfcc

If combined score > 0.7 → Likely a headshot!

## 📝 Usage

### Python API

```python
from src.predict import predict_one

# Prediction without template matching
result = predict_one("audio.wav")

# Prediction with template matching for improved headshot detection
result = predict_one(
    audio_path="audio.wav",
    use_template=True,
    template_path="headshot_template.wav"
)

print(f"Predicted: {result['predicted_class']}")
print(f"Confidence: {result['predicted_confidence']:.1%}")

if 'template_matching' in result:
    print(f"Template similarity: {result['template_matching']['combined']:.1%}")
```

### Command Line

```bash
# Without template matching
python -m src.predict audio.wav

# With template matching
python -m src.predict audio.wav headshot_template.wav
```

### Batch Prediction

```python
from src.predict import predict_batch

audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]

results = predict_batch(
    audio_files,
    use_template=True,
    template_path="headshot_template.wav"
)

for result in results:
    print(f"{result['audio_path']}: {result['predicted_class']}")
    if 'template_matching' in result:
        print(f"  Template similarity: {result['template_matching']['combined']:.1%}")
```

## 🎧 Creating a Template

### What Makes a Good Template?

Your headshot template should be:
- ✅ **Clear and isolated** - Minimal background noise
- ✅ **Representative** - Typical headshot sound from your game
- ✅ **Short** - 1-3 seconds maximum (just the skull crack + initial death cry)
- ✅ **High quality** - Good audio recording

### How to Record a Template

1. **Play Hunt Showdown**
2. **Get a clean headshot** (or extract from VOD)
3. **Record the audio** at the exact moment of headshot impact
4. **Trim to just the sound** (skull crack + brief death cry)
5. **Save as WAV** for best compatibility

### Template Recording Tips

```python
# If you have a longer recording, extract just the headshot part
import librosa
import soundfile as sf

# Load full recording
audio, sr = librosa.load("full_recording.wav", sr=22050)

# Extract 2 seconds starting at 1 second mark
start_sample = int(1 * sr)
end_sample = int(3 * sr)
headshot_clip = audio[start_sample:end_sample]

# Save as template
sf.write("headshot_template.wav", headshot_clip, sr)
```

## 📊 Understanding Results

### Sample Output

```
=== Prediction Results ===
Audio: audio.wav
Predicted: headshot (78.5%)

All probabilities:
  - injured:  5.2%
  - kill:     16.3%
  - headshot: 78.5%

Template Matching:
  - Cross-correlation: 72%
  - Mel-Spectrogram:   68%
  - MFCC:              75%
  - Combined:          71%
```

### Interpreting Template Similarity

| Score | Interpretation |
|-------|----------------|
| < 0.5 | Likely NOT a headshot |
| 0.5-0.7 | Maybe a headshot (ambiguous) |
| > 0.7 | Likely a headshot |

## 🔍 Advanced Usage

### Direct Template Matching

```python
from src.template_matching import TemplateAudioMatcher
import librosa

# Initialize matcher with template
matcher = TemplateAudioMatcher("headshot_template.wav")

# Load audio to test
audio, sr = librosa.load("test_audio.wav", sr=22050)

# Get individual similarity scores
similarities = matcher.get_all_similarities(audio)
print(f"Cross-correlation: {similarities['cross_correlation']:.2%}")
print(f"Mel-Spectrogram: {similarities['mel_spectrogram']:.2%}")
print(f"MFCC: {similarities['mfcc']:.2%}")
print(f"Combined: {similarities['combined']:.2%}")

# Or use combined score directly
combined_score = matcher.combined_similarity(audio)
if combined_score > 0.7:
    print("This is likely a headshot!")
```

### Custom Weights

```python
# Adjust weights if needed
custom_weights = {
    "cross_correlation": 0.5,  # More emphasis on temporal pattern
    "mel": 0.3,
    "mfcc": 0.2,
}

similarity = matcher.combined_similarity(audio, weights=custom_weights)
```

## ⚙️ Tuning the Threshold

The default headshot detection threshold is 0.7. You can adjust it:

```python
# In src/predict.py, modify this line:
if combined_sim > 0.7:  # Change 0.7 to your preferred threshold
    headshot_boost = combined_sim * 0.3
    # ...
```

### Finding Your Optimal Threshold

```python
# Test different thresholds
for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
    correct = 0
    for result in results:
        template_sim = result.get('template_matching', {}).get('combined', 0)
        if template_sim > threshold:
            pred = 'headshot'
        else:
            pred = result['predicted_class']
        
        if pred == result['actual_class']:
            correct += 1
    
    accuracy = correct / len(results)
    print(f"Threshold {threshold}: {accuracy:.2%} accuracy")
```

## 🐛 Troubleshooting

### Template matching not helping?

1. **Verify template quality**
   ```python
   from src.template_matching import TemplateAudioMatcher
   matcher = TemplateAudioMatcher("headshot_template.wav")
   print(f"Template loaded: {matcher.template_audio.shape}")
   ```

2. **Check similarity scores**
   - If scores are always < 0.5, template might be too different
   - Try recording a new, clearer template

3. **Adjust weights**
   - If cross-correlation is most reliable, increase its weight
   - Test different combinations

4. **Increase threshold**
   - If getting false positives, raise from 0.7 to 0.75 or 0.8

## 📈 Performance Impact

- ✅ Minimal performance overhead (~0.1s per audio)
- ✅ No additional GPU required
- ✅ Works offline without internet
- ✅ Can significantly boost headshot detection accuracy

## 📚 References

- Cross-Correlation: [Signal Processing](https://en.wikipedia.org/wiki/Cross-correlation)
- MFCC: [Mel-Frequency Cepstral Coefficients](https://librosa.org/doc/main/generated/librosa.feature.mfcc.html)
- Template Matching: [Audio Similarity](https://en.wikipedia.org/wiki/Audio_signal_processing)

## 🎯 Next Steps

1. **Collect multiple templates** for different scenarios
2. **Fine-tune weights** based on your results
3. **Combine with data augmentation** for better CNN performance
4. **Collect more training data** to improve baseline accuracy
