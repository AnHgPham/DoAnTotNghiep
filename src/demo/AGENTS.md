# Demo Directory Agent

## Responsibility
Gradio web application for demonstrating the KWS system interactively.

## Files

### `app.py` - Main Gradio Application (3 Tabs)

```python
def create_app(encoder, classifier, vad=None, denoiser=None, speaker_gate=None) -> gr.Blocks:
    """Create full Gradio app with 3 tabs."""

# Tab 1: Offline Detection
def offline_tab() -> gr.Tab:
    """Upload audio file -> detect keywords.
    Shows: keyword result, L2 distance to all prototypes, MFCC spectrogram."""

# Tab 2: Enrollment
def enrollment_tab() -> gr.Tab:
    """Record 3-5 samples per keyword to register new keywords (few-shot).
    Shows: current enrolled keywords list, prototype count."""

# Tab 3: Settings + Streaming
def settings_tab() -> gr.Tab:
    """Controls: threshold slider, denoising toggle, shot count, speaker gate toggle.
    Streaming demo: upload long audio file (10-30s), run StreamingKWS, show timeline results.
    (Simulated streaming is more stable than live mic for demo/defense.)"""
```

## UI Guidelines

- Use `gr.Blocks` for layout (not Interface) for maximum flexibility
- Show clear status: "Keyword Detected: YES (distance: 0.23)", "REJECTED (unknown)"
- Color code results: green for detected keyword, red for rejected, gray for unknown
- Display MFCC spectrogram using `gr.Plot` with matplotlib
- Cold start: show "Waiting for Enrollment" when no keywords registered yet

## Launch Configuration

```python
if __name__ == "__main__":
    app = create_app(encoder, classifier, vad, denoiser, speaker_gate)
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
```

## Error Handling

- "No keywords enrolled" -> show clear enrollment instructions (cold start)
- "Audio too short (< 0.5s)" -> warning with suggestion to re-record
- Model loading errors -> show which models failed and suggest fixes
- Streaming fallback: if live mic fails, use file upload + simulated streaming
