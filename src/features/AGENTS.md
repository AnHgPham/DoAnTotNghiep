# Features Directory Agent

## Responsibility
Audio feature extraction (MFCC with narrow + transpose) and data augmentation (noise mixing).

## Files

### `mfcc.py` - MFCC Feature Extraction
```python
class MFCCExtractor:
    def __init__(self, n_mfcc=40, num_features=10, sample_rate=16000,
                 win_length_ms=40, hop_length_ms=20):
        # n_mfcc=40: compute 40 cepstral coefficients
        # num_features=10: keep only first 10 (via torch.narrow)
        # win_length = 640 samples, hop_length = 320 samples at 16kHz

    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        """Input: (1, T) raw audio at 16kHz.
        Processing: MFCC(40) -> narrow(10) -> transpose
        Output: (1, 49, 10) = (channel, T_frames, n_features).
        For 1-sec audio: T_frames=49, n_features=10."""

    def extract_batch(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Input: (B, 1, T) raw audio.
        Output: (B, 1, 49, 10) = (batch, channel, T_frames, n_features).
        Ready to feed DSCNN directly."""
```

Preprocessing flow (matches Rusci et al. `preprocessing.py`):
1. `torchaudio.transforms.MFCC(n_mfcc=40)` -> raw output: `(1, 40, 49)`
2. `torch.narrow(dim=-2, start=0, length=10)` -> keep first 10 coefficients: `(1, 10, 49)`
3. `.mT` (transpose last two dims) -> `(1, 49, 10)` = `(channel, T, n_features)`

Key details:
- Higher-order cepstral coefficients (11-40) are discarded -- they carry less information
- Pad waveforms shorter than 1 second with zeros on the right
- Truncate waveforms longer than 1 second from the right
- Output must be deterministic (no randomness in feature extraction)
- `x_dim` convention from Rusci: `'1,49,10'` = (channel, T, features)

### `augmentation.py` - Noise Augmentation
```python
class NoiseAugmenter:
    def __init__(self, noise_dir: Path, prob: float = 0.95, snr_db: float = 5.0):
        """noise_dir: path to DEMAND dataset WAV files.
        snr_db: fixed SNR level (Rusci uses noise_snr=5)."""

    def augment(self, waveform: torch.Tensor) -> torch.Tensor:
        """Randomly add background noise at fixed 5dB SNR. Only during training."""

    def _mix_snr(self, clean: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
        """Mix at specified SNR with RMS normalization:
        scale = rms(clean) / (rms(noise) * 10^(snr_db/20))
        noisy = clean + scale * noise"""
```

Key details:
- Random noise clip selection from DEMAND dataset
- Fixed SNR = 5dB (not a range -- matches Rusci `noise_snr=5`)
- Noise is looped if shorter than clean signal, cropped if longer
- NEVER augment during evaluation/inference

## Testing

```python
def test_mfcc_shape():
    ext = MFCCExtractor()
    wav = torch.randn(1, 16000)  # 1 second, shape (1, T)
    mfcc = ext.extract(wav)
    assert mfcc.shape == (1, 49, 10)  # (channel, T_frames, n_features)

def test_mfcc_short_audio():
    ext = MFCCExtractor()
    wav = torch.randn(1, 8000)  # 0.5 second
    mfcc = ext.extract(wav)  # should pad to 1 sec
    assert mfcc.shape == (1, 49, 10)  # same output regardless of input length

def test_mfcc_batch():
    ext = MFCCExtractor()
    wavs = torch.randn(4, 1, 16000)  # batch of 4
    mfcc = ext.extract_batch(wavs)
    assert mfcc.shape == (4, 1, 49, 10)  # (B, 1, T, n_features) -- feeds DSCNN directly

def test_mfcc_num_features():
    ext = MFCCExtractor(num_features=10)
    wav = torch.randn(1, 16000)
    mfcc = ext.extract(wav)
    assert mfcc.shape[-1] == 10  # only 10 features kept

def test_augmentation_probability():
    aug = NoiseAugmenter(noise_dir=..., prob=0.0, snr_db=5.0)
    wav = torch.randn(1, 16000)
    assert torch.equal(aug.augment(wav), wav)  # prob=0 means no change
```
