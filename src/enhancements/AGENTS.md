# Enhancements Directory Agent

## Responsibility
Optional enhancement modules: audio denoising and speaker verification gate.

## Files

### `denoiser.py` - Audio Denoising

```python
class AudioDenoiser:
    def __init__(self, method: str = "spectral_gating", device: str = "cpu"):
        """
        Methods:
        - 'spectral_gating': noisereduce library, fast, CPU-only
        - 'speechbrain': SepformerSeparation, better quality, GPU recommended
        - 'none': passthrough (for A/B testing)
        """

    def denoise(self, waveform: torch.Tensor, sr: int = 16000) -> torch.Tensor:
        """Input/Output: (1, T) or (T,) tensor. Always returns (1, T)."""

    def benchmark(self, clean: torch.Tensor, noisy: torch.Tensor) -> dict:
        """Returns {'snr_improvement_db': float, 'processing_time_ms': float}"""
```

### `speaker_verify.py` - Speaker Verification Gate

```python
class SpeakerGate:
    def __init__(self, threshold: float = 0.25, device: str = "cpu"):
        """Uses SpeechBrain ECAPA-TDNN (192-dim embeddings)."""

    def enroll(self, audio_samples: list[torch.Tensor]) -> None:
        """Register speaker from 3-5 samples. Computes mean speaker embedding."""

    def verify(self, query_audio: torch.Tensor) -> tuple[bool, float]:
        """Returns (is_owner, cosine_similarity). is_owner = similarity > threshold."""

    def save_profile(self, path: Path) -> None:
    def load_profile(self, path: Path) -> None:

    @property
    def is_enrolled(self) -> bool:
        """Check if a speaker profile exists."""
```

## Integration Pattern

Both modules are OPTIONAL and follow the same pattern:
1. Can be None in the pipeline (skipped if not provided)
2. Fail gracefully -- never block KWS from functioning
3. Can be toggled on/off at runtime via Gradio UI

## Testing

```python
def test_denoiser_passthrough():
    dn = AudioDenoiser(method="none")
    wav = torch.randn(1, 16000)
    assert torch.equal(dn.denoise(wav), wav)

def test_denoiser_output_shape():
    dn = AudioDenoiser(method="spectral_gating")
    wav = torch.randn(1, 16000)
    out = dn.denoise(wav)
    assert out.shape == (1, 16000)

def test_speaker_gate_not_enrolled():
    gate = SpeakerGate()
    assert not gate.is_enrolled
    with pytest.raises(RuntimeError):
        gate.verify(torch.randn(16000))

def test_speaker_gate_enrollment():
    gate = SpeakerGate()
    samples = [torch.randn(16000) for _ in range(3)]
    gate.enroll(samples)
    assert gate.is_enrolled
    is_owner, score = gate.verify(samples[0])
    assert isinstance(is_owner, bool)
    assert 0 <= score <= 1
```
