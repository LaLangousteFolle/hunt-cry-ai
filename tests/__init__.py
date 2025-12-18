"""Unit tests for Hunt Showdown Sound AI model."""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import os

from src.model import HuntCryClassifier
from src.audio import audio_to_mel, TARGET_SR, TARGET_SAMPLES
from src.dataset import HuntCryDataset, CLASS2IDX, IDX2CLASS


class TestModel:
    """Test model architecture and forward pass."""
    
    def test_model_initialization(self):
        """Test model can be initialized."""
        model = HuntCryClassifier(num_classes=3, dropout_rate=0.3)
        assert model is not None
        assert model.get_parameter_count() > 0
    
    def test_model_forward_pass(self):
        """Test forward pass with random input."""
        model = HuntCryClassifier(num_classes=3)
        batch_size = 4
        x = torch.randn(batch_size, 1, 128, 44)  # [B, C, H, W]
        output = model(x)
        
        assert output.shape == (batch_size, 3)
    
    def test_model_eval_mode(self):
        """Test model evaluation mode."""
        model = HuntCryClassifier()
        model.eval()
        
        x = torch.randn(1, 1, 128, 44)
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (1, 3)
    
    def test_model_device_transfer(self):
        """Test model can be moved to different devices."""
        model = HuntCryClassifier()
        
        # CPU
        model = model.to('cpu')
        assert next(model.parameters()).device.type == 'cpu'
        
        # CUDA if available
        if torch.cuda.is_available():
            model = model.to('cuda')
            assert next(model.parameters()).device.type == 'cuda'
    
    def test_model_parameter_count(self):
        """Test parameter counting."""
        model = HuntCryClassifier()
        param_count = model.get_parameter_count()
        
        assert param_count > 0
        assert isinstance(param_count, int)


class TestAudio:
    """Test audio processing functions."""
    
    def test_audio_to_mel_shape(self):
        """Test mel-spectrogram output shape."""
        # Create temporary wav file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            import soundfile as sf
            
            # Generate dummy audio
            audio = np.random.randn(TARGET_SR * 2)
            try:
                sf.write(tmp.name, audio, TARGET_SR)
                
                mel = audio_to_mel(tmp.name)
                
                assert mel.shape[0] == 1  # Batch dimension
                assert mel.shape[1] == 128  # n_mels
                assert mel.shape[2] == 128  # Time steps
            finally:
                os.unlink(tmp.name)
    
    def test_audio_to_mel_padding(self):
        """Test padding of short audio."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            import soundfile as sf
            
            # Very short audio (0.5 seconds)
            short_audio = np.random.randn(TARGET_SR // 2)
            try:
                sf.write(tmp.name, short_audio, TARGET_SR)
                mel = audio_to_mel(tmp.name)
                
                # Should still output correct shape
                assert mel.shape == (1, 128, 128)
            finally:
                os.unlink(tmp.name)
    
    def test_audio_to_mel_file_not_found(self):
        """Test error handling for missing files."""
        with pytest.raises(FileNotFoundError):
            audio_to_mel("/nonexistent/file.wav")


class TestDataset:
    """Test dataset loading and processing."""
    
    def test_class_mappings(self):
        """Test class index mappings."""
        assert CLASS2IDX["injured"] == 0
        assert CLASS2IDX["kill"] == 1
        assert CLASS2IDX["headshot"] == 2
        
        assert IDX2CLASS[0] == "injured"
        assert IDX2CLASS[1] == "kill"
        assert IDX2CLASS[2] == "headshot"
    
    def test_dataset_missing_csv(self):
        """Test error handling for missing CSV."""
        with pytest.raises(FileNotFoundError):
            HuntCryDataset(csv_path="/nonexistent/labels.csv")
    
    def test_dataset_empty_csv(self):
        """Test error handling for empty CSV."""
        with tempfile.NamedTemporaryFile(mode='w', suffix=".csv", delete=False) as tmp:
            # Write empty CSV with just headers
            tmp.write("filepath,class\n")
            tmp.flush()
            
            try:
                with pytest.raises(ValueError):
                    HuntCryDataset(csv_path=tmp.name)
            finally:
                os.unlink(tmp.name)
    
    def test_dataset_invalid_columns(self):
        """Test error handling for invalid CSV columns."""
        with tempfile.NamedTemporaryFile(mode='w', suffix=".csv", delete=False) as tmp:
            tmp.write("wrong_col1,wrong_col2\n")
            tmp.write("data1,data2\n")
            tmp.flush()
            
            try:
                with pytest.raises(ValueError):
                    HuntCryDataset(csv_path=tmp.name)
            finally:
                os.unlink(tmp.name)


class TestIntegration:
    """Integration tests."""
    
    def test_model_training_step(self):
        """Test a single training step."""
        model = HuntCryClassifier()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Forward pass
        x = torch.randn(4, 1, 128, 44)
        y = torch.tensor([0, 1, 2, 0], dtype=torch.long)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        assert loss.item() > 0
    
    def test_prediction_pipeline(self):
        """Test end-to-end prediction."""
        model = HuntCryClassifier()
        model.eval()
        
        x = torch.randn(1, 1, 128, 44)
        
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(1).item()
            confidence = probs[0, pred].item()
        
        assert 0 <= pred < 3
        assert 0 <= confidence <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
