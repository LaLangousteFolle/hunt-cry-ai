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
    
    # Get num_classes from checkpoint if available
    num_classes = checkpoint.get('config', {}).get('num_classes', 2)
    
    model = HuntCryClassifier(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    logger.info(f"Model loaded successfully (trained for {checkpoint['epoch']} epochs)")
    logger.info(f"Best validation accuracy: {checkpoint['accuracy']:.2%}")
    logger.info(f"Model classes: {num_classes} (kill vs headshot)")
    
    return model, checkpoint


def predict_one(audio_path: str, model_path: str = "models/hunt_cry_best.pt",
               device: str = None, use_template: bool = False,
               template_path: str = None, template_threshold: float = 0.65) -> dict:
    """
    Predict class for a single audio file.
    
    For 2-class model (kill vs headshot), template matching is HIGHLY recommended
    to improve headshot detection accuracy.
    
    Args:
        audio_path: Path to audio file (wav, mp3, etc.)
        model_path: Path to trained model checkpoint
        device: Device to use ('cuda' or 'cpu'). Auto-detects if None.
        use_template: Whether to use template matching for headshot detection
        template_path: Path to headshot template audio (required if use_template=True)
        template_threshold: Threshold for template matching (default: 0.65)
        
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
            
            # AGGRESSIVE APPROACH: If template similarity high, FORCE headshot
            combined_sim = template_similarities['combined']
            if combined_sim > template_threshold:
                logger.info(f"❗ Template similarity ({combined_sim:.1%}) > threshold ({template_threshold:.1%})")
                logger.info("   → FORCING prediction to HEADSHOT")
                
                # Force headshot prediction
                pred_class = "headshot"
                pred_prob = combined_sim
                
                # Update probabilities
                cnn_probs["headshot"] = combined_sim
                cnn_probs["kill"] = 1.0 - combined_sim
            else:
                logger.info(f"Template similarity ({combined_sim:.1%}) < threshold, using CNN prediction")
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
        results["template_threshold"] = template_threshold
    
    # Log results
    logger.info(f"\n=== Prediction Results ===")
    logger.info(f"Audio: {audio_path}")
    logger.info(f"Predicted: {pred_class.upper()} ({pred_prob:.1%})")
    logger.info(f"\nProbabilities:")
    for class_name, prob in results["probabilities"].items():
        logger.info(f"  - {class_name:10s}: {prob:.2%}")
    
    if template_info:
        logger.info(f"\nTemplate Matching:")
        logger.info(f"  - Cross-correlation: {template_info['cross_correlation']:.2%}")
        logger.info(f"  - Mel-Spectrogram:   {template_info['mel_spectrogram']:.2%}")
        logger.info(f"  - MFCC:              {template_info['mfcc']:.2%}")
        logger.info(f"  - Combined:          {template_info['combined']:.2%} (threshold: {template_threshold:.0%})")
    
    return results


def predict_batch(audio_paths: list, model_path: str = "models/hunt_cry_best.pt",
                 device: str = None, use_template: bool = False,
                 template_path: str = None, template_threshold: float = 0.65) -> list:
    """
    Predict classes for multiple audio files.
    
    Args:
        audio_paths: List of paths to audio files
        model_path: Path to trained model checkpoint
        device: Device to use ('cuda' or 'cpu'). Auto-detects if None.
        use_template: Whether to use template matching
        template_path: Path to headshot template audio
        template_threshold: Threshold for forcing headshot detection
        
    Returns:
        List of prediction dictionaries
    """
    results = []
    for i, audio_path in enumerate(audio_paths):
        logger.info(f"\nProcessing {i+1}/{len(audio_paths)}...")
        try:
            result = predict_one(audio_path, model_path, device, use_template, 
                               template_path, template_threshold)
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
        print("Usage: python -m src.predict <audio_file> [template_path] [threshold]")
        print("Example: python -m src.predict sample.wav headshot_template.wav 0.65")
        print("\nNote: Model now predicts KILL vs HEADSHOT (2 classes only)")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    template_file = sys.argv[2] if len(sys.argv) > 2 else None
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.65
    use_template = template_file is not None
    
    try:
        result = predict_one(audio_file, use_template=use_template, 
                           template_path=template_file, template_threshold=threshold)
        print(f"\n✓ Prediction: {result['predicted_class'].upper()} ({result['predicted_confidence']:.1%})")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)
