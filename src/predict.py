import logging
import os
import torch
from src.model import HuntCryClassifier
from src.audio import audio_to_mel
from src.dataset import IDX2CLASS
from src.template_matching import TemplateAudioMatcher
import librosa

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_path: str, device: str = "cpu") -> tuple:
    """
    Load trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint (.pt file)
        device: Device to load model to ('cuda' or 'cpu')
        
    Returns:
        Tuple of (model, checkpoint_dict)
        
    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    logger.info(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    model = HuntCryClassifier().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    logger.info(f"Model loaded successfully (trained for {checkpoint['epoch']} epochs)")
    logger.info(f"Best validation accuracy: {checkpoint['accuracy']:.2%}")
    
    return model, checkpoint


def predict_one(audio_path: str, model_path: str = "models/hunt_cry_best.pt",
               device: str = None, use_template: bool = False,
               template_path: str = None) -> dict:
    """
    Predict class for a single audio file.
    
    Optionally uses template matching to improve headshot detection.
    
    Args:
        audio_path: Path to audio file (wav, mp3, etc.)
        model_path: Path to trained model checkpoint
        device: Device to use ('cuda' or 'cpu'). Auto-detects if None.
        use_template: Whether to use template matching for headshot detection
        template_path: Path to headshot template audio (required if use_template=True)
        
    Returns:
        Dictionary with predictions and probabilities
        
    Raises:
        FileNotFoundError: If audio or model file doesn't exist
        RuntimeError: If prediction fails
    """
    # Setup device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Device: {device}")
    
    # Load model
    model, checkpoint = load_model(model_path, device=device)
    
    # Load template matcher if needed
    template_matcher = None
    if use_template:
        if template_path is None:
            logger.warning("use_template=True but template_path not provided, skipping template matching")
        elif os.path.exists(template_path):
            try:
                template_matcher = TemplateAudioMatcher(template_path)
                logger.info("Template matcher loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load template matcher: {str(e)}")
        else:
            logger.warning(f"Template file not found: {template_path}")
    
    # Load and process audio
    logger.info(f"Processing audio: {audio_path}")
    try:
        x = audio_to_mel(audio_path)
        x = x.unsqueeze(0).to(device)  # Add batch dimension
        logger.info(f"Audio shape: {x.shape}")
        
        # Also load raw audio for template matching
        if template_matcher:
            raw_audio, _ = librosa.load(audio_path, sr=22050)
    except Exception as e:
        raise RuntimeError(f"Failed to process audio: {str(e)}")
    
    # Make prediction
    try:
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            pred_idx = logits.argmax(1).item()
            pred_class = IDX2CLASS[pred_idx]
            pred_prob = probs[0, pred_idx].item()
    except Exception as e:
        raise RuntimeError(f"Model prediction failed: {str(e)}")
    
    # Get CNN probabilities
    cnn_probs = {
        IDX2CLASS[i]: probs[0, i].item()
        for i in range(len(IDX2CLASS))
    }
    
    # Template matching for headshot refinement
    template_info = None
    if template_matcher:
        try:
            template_similarities = template_matcher.get_all_similarities(raw_audio)
            template_info = template_similarities
            logger.info(f"Template matching - Combined similarity: {template_similarities['combined']:.2%}")
            
            # Boost headshot probability if template similarity is high
            combined_sim = template_similarities['combined']
            if combined_sim > 0.7:  # Threshold for headshot detection
                # Adjust probabilities: increase headshot, decrease others
                headshot_boost = combined_sim * 0.3  # Max 30% boost
                cnn_probs["headshot"] = min(1.0, cnn_probs["headshot"] + headshot_boost)
                
                # Normalize probabilities
                total = sum(cnn_probs.values())
                cnn_probs = {k: v / total for k, v in cnn_probs.items()}
                
                # Update prediction if headshot now highest
                pred_class = max(cnn_probs, key=cnn_probs.get)
                pred_prob = cnn_probs[pred_class]
                
                logger.info(f"Headshot probability boosted by template matching!")
        except Exception as e:
            logger.warning(f"Template matching failed: {str(e)}")
    
    # Prepare results
    results = {
        "predicted_class": pred_class,
        "predicted_confidence": pred_prob,
        "probabilities": cnn_probs,
        "audio_path": audio_path,
        "model_path": model_path,
    }
    
    if template_info:
        results["template_matching"] = template_info
    
    # Log results
    logger.info(f"\n=== Prediction Results ===")
    logger.info(f"Audio: {audio_path}")
    logger.info(f"Predicted: {pred_class} ({pred_prob:.1%})")
    logger.info(f"\nAll probabilities:")
    for class_name, prob in results["probabilities"].items():
        logger.info(f"  - {class_name:10s}: {prob:.2%}")
    
    if template_info:
        logger.info(f"\nTemplate Matching:")
        logger.info(f"  - Cross-correlation: {template_info['cross_correlation']:.2%}")
        logger.info(f"  - Mel-Spectrogram:   {template_info['mel_spectrogram']:.2%}")
        logger.info(f"  - MFCC:              {template_info['mfcc']:.2%}")
        logger.info(f"  - Combined:          {template_info['combined']:.2%}")
    
    return results


def predict_batch(audio_paths: list, model_path: str = "models/hunt_cry_best.pt",
                 device: str = None, use_template: bool = False,
                 template_path: str = None) -> list:
    """
    Predict classes for multiple audio files.
    
    Args:
        audio_paths: List of paths to audio files
        model_path: Path to trained model checkpoint
        device: Device to use ('cuda' or 'cpu'). Auto-detects if None.
        use_template: Whether to use template matching
        template_path: Path to headshot template audio
        
    Returns:
        List of prediction dictionaries
    """
    results = []
    for i, audio_path in enumerate(audio_paths):
        logger.info(f"\nProcessing {i+1}/{len(audio_paths)}...")
        try:
            result = predict_one(audio_path, model_path, device, use_template, template_path)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to predict {audio_path}: {str(e)}")
            results.append({
                "audio_path": audio_path,
                "error": str(e)
            })
    
    return results


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.predict <audio_file> [template_path]")
        print("Example: python -m src.predict sample.wav headshot_template.wav")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    template_file = sys.argv[2] if len(sys.argv) > 2 else None
    use_template = template_file is not None
    
    try:
        result = predict_one(audio_file, use_template=use_template, template_path=template_file)
        print(f"\n✓ Prediction: {result['predicted_class']} ({result['predicted_confidence']:.1%})")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)
