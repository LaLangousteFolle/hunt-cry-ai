import logging
import numpy as np
import librosa
from scipy import signal
from scipy.spatial.distance import euclidean
import os

logger = logging.getLogger(__name__)


class TemplateAudioMatcher:
    """
    Template-based audio matching using various similarity metrics.
    Useful for detecting specific sounds like headshot impact.
    """
    
    def __init__(self, template_path: str, sr: int = 22050):
        """
        Initialize with a template audio file.
        
        Args:
            template_path: Path to template audio file (e.g., clear headshot recording)
            sr: Sample rate
            
        Raises:
            FileNotFoundError: If template file doesn't exist
        """
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template file not found: {template_path}")
        
        self.sr = sr
        self.template_path = template_path
        
        # Load template
        try:
            self.template_audio, _ = librosa.load(template_path, sr=sr)
            logger.info(f"Loaded template from {template_path}")
            logger.info(f"Template duration: {len(self.template_audio) / sr:.2f}s")
        except Exception as e:
            raise RuntimeError(f"Failed to load template: {str(e)}")
        
        # Pre-compute template features
        self.template_mel = self._compute_mel_spectrogram(self.template_audio)
        self.template_mfcc = librosa.feature.mfcc(y=self.template_audio, sr=sr, n_mfcc=13)
        self.template_chroma = librosa.feature.chroma_stft(y=self.template_audio, sr=sr)
    
    def _compute_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute normalized mel-spectrogram.
        
        Args:
            audio: Audio signal
            
        Returns:
            Normalized mel-spectrogram
        """
        mel = librosa.feature.melspectrogram(y=audio, sr=self.sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        # Normalize
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
        return mel_db
    
    def cross_correlation_similarity(self, audio: np.ndarray) -> float:
        """
        Compute similarity using cross-correlation.
        Good for detecting temporal patterns.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Similarity score (0-1, higher = more similar)
        """
        # Normalize both signals
        audio_norm = (audio - np.mean(audio)) / (np.std(audio) + 1e-8)
        template_norm = (self.template_audio - np.mean(self.template_audio)) / (np.std(self.template_audio) + 1e-8)
        
        # Pad to same length
        if len(audio_norm) < len(template_norm):
            audio_norm = np.pad(audio_norm, (0, len(template_norm) - len(audio_norm)), mode='constant')
        else:
            audio_norm = audio_norm[:len(template_norm)]
        
        # Cross-correlation
        correlation = signal.correlate(audio_norm, template_norm, mode='same')
        correlation = correlation / np.max(np.abs(correlation)) if np.max(np.abs(correlation)) > 0 else correlation
        
        # Return maximum correlation normalized to 0-1
        similarity = (np.max(correlation) + 1) / 2  # Shift from [-1, 1] to [0, 1]
        return float(similarity)
    
    def mel_spectrogram_similarity(self, audio: np.ndarray) -> float:
        """
        Compute similarity using mel-spectrogram cosine distance.
        Good for overall spectral content comparison.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Similarity score (0-1, higher = more similar)
        """
        mel = self._compute_mel_spectrogram(audio)
        
        # Flatten both spectrograms
        template_flat = self.template_mel.flatten()
        mel_flat = mel.flatten()
        
        # Pad shorter one
        if len(mel_flat) < len(template_flat):
            mel_flat = np.pad(mel_flat, (0, len(template_flat) - len(mel_flat)), mode='constant')
        else:
            mel_flat = mel_flat[:len(template_flat)]
        
        # Cosine similarity
        dot_product = np.dot(template_flat, mel_flat)
        norm_template = np.linalg.norm(template_flat)
        norm_mel = np.linalg.norm(mel_flat)
        
        if norm_template == 0 or norm_mel == 0:
            return 0.0
        
        similarity = dot_product / (norm_template * norm_mel)
        # Shift from [-1, 1] to [0, 1]
        similarity = (similarity + 1) / 2
        return float(similarity)
    
    def mfcc_similarity(self, audio: np.ndarray) -> float:
        """
        Compute similarity using MFCC (Mel-Frequency Cepstral Coefficients).
        Good for capturing perceptual characteristics.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Similarity score (0-1, higher = more similar)
        """
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=13)
        
        # Flatten and pad
        template_flat = self.template_mfcc.flatten()
        mfcc_flat = mfcc.flatten()
        
        if len(mfcc_flat) < len(template_flat):
            mfcc_flat = np.pad(mfcc_flat, (0, len(template_flat) - len(mfcc_flat)), mode='constant')
        else:
            mfcc_flat = mfcc_flat[:len(template_flat)]
        
        # Normalize
        template_flat = (template_flat - template_flat.mean()) / (template_flat.std() + 1e-8)
        mfcc_flat = (mfcc_flat - mfcc_flat.mean()) / (mfcc_flat.std() + 1e-8)
        
        # Cosine similarity
        dot_product = np.dot(template_flat, mfcc_flat)
        norm_template = np.linalg.norm(template_flat)
        norm_mfcc = np.linalg.norm(mfcc_flat)
        
        if norm_template == 0 or norm_mfcc == 0:
            return 0.0
        
        similarity = dot_product / (norm_template * norm_mfcc)
        similarity = (similarity + 1) / 2
        return float(similarity)
    
    def combined_similarity(self, audio: np.ndarray, weights: dict = None) -> float:
        """
        Compute combined similarity using multiple metrics.
        
        Args:
            audio: Input audio signal
            weights: Dictionary with weights for each metric
                    Default: {"cross_correlation": 0.3, "mel": 0.4, "mfcc": 0.3}
            
        Returns:
            Combined similarity score (0-1)
        """
        if weights is None:
            weights = {
                "cross_correlation": 0.3,
                "mel": 0.4,
                "mfcc": 0.3,
            }
        
        try:
            cc_sim = self.cross_correlation_similarity(audio)
            mel_sim = self.mel_spectrogram_similarity(audio)
            mfcc_sim = self.mfcc_similarity(audio)
            
            combined = (
                weights["cross_correlation"] * cc_sim +
                weights["mel"] * mel_sim +
                weights["mfcc"] * mfcc_sim
            )
            
            return float(combined)
        except Exception as e:
            logger.error(f"Error computing combined similarity: {str(e)}")
            return 0.0
    
    def get_all_similarities(self, audio: np.ndarray) -> dict:
        """
        Get all similarity metrics at once.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Dictionary with all similarity scores
        """
        return {
            "cross_correlation": self.cross_correlation_similarity(audio),
            "mel_spectrogram": self.mel_spectrogram_similarity(audio),
            "mfcc": self.mfcc_similarity(audio),
            "combined": self.combined_similarity(audio),
        }
