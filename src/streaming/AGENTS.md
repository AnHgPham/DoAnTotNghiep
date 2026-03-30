# Streaming Directory Agent

## Responsibility
Real-time keyword spotting: VAD detection + sliding window inference + pipeline composition.

## Files

### `vad.py` - Voice Activity Detection Wrapper

```python
class VADWrapper:
    def __init__(self, threshold: float = 0.5, min_speech_duration_ms: int = 250,
                 sample_rate: int = 16000):
        """Load Silero VAD model. Processes 32ms chunks (512 samples at 16kHz)."""

    def process_chunk(self, chunk: torch.Tensor) -> bool:
        """Input: (512,) float32 tensor. Returns True if speech detected."""

    def get_speech_probability(self, chunk: torch.Tensor) -> float:
        """Returns raw probability [0, 1] without thresholding."""

    def reset(self) -> None:
        """Reset internal hidden states. Call between sessions."""
```

### `sliding_window.py` - Streaming KWS Engine

```python
class StreamingKWS:
    def __init__(self, encoder: nn.Module, classifier: OpenNCMClassifier,
                 vad: VADWrapper, denoiser: AudioDenoiser | None = None,
                 speaker_gate: SpeakerGate | None = None,
                 window_size: int = 16000, stride: int = 8000):
        """
        window_size: 16000 samples = 1 second
        stride: 8000 samples = 0.5 second overlap
        """

    def feed_chunk(self, chunk: torch.Tensor) -> dict | None:
        """Feed audio chunk (any size). Returns result dict when detection occurs.
        Result: {
            'keyword': str,
            'confidence': float,      # L2 distance (lower = more confident)
            'speaker_verified': bool,  # True if speaker gate passed or disabled
            'timestamp_ms': int
        }
        Returns None if window not full yet or VAD says no speech."""

    def start(self) -> None:
        """Start streaming session. Reset all states."""

    def stop(self) -> list[dict]:
        """Stop session. Returns list of all detections."""

    def reset(self) -> None:
        """Reset buffers and VAD state without stopping."""
```

## Pipeline Flow

```
Audio chunk (32ms)
  -> VAD: speech? ──No──> discard, return None
       │ Yes
       v
  -> Append to ring buffer
  -> Buffer full (1 sec)? ──No──> return None
       │ Yes
       v
  -> [Optional] Denoise
  -> Extract MFCC
  -> DSCNN encode
  -> Classifier predict
  -> [Optional] Speaker verify
  -> Return result dict
  -> Slide buffer by stride
```

## Error Handling

- `feed_chunk` must never raise -- catch all exceptions, log, return None
- If denoiser fails, skip denoising and process raw audio
- If speaker_gate fails, treat as verified (fail-open for usability)
- Buffer overflow: drop oldest data silently

## Performance Budget

| Step | Target Latency |
|------|---------------|
| VAD per chunk | < 5ms |
| Denoise (1 sec) | < 30ms |
| MFCC extraction | < 5ms |
| DSCNN inference | < 10ms |
| Classifier | < 1ms |
| Speaker verify | < 20ms |
| **Total** | **< 71ms** |
