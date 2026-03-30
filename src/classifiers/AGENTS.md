# Classifiers Directory Agent

## Responsibility
Open-set classification using prototypical distances. Main method: Direct L2 distance with openNCM.

## Files

### `open_ncm.py` - Open Nearest Class Mean Classifier

```python
class OpenNCMClassifier:
    def __init__(self, threshold: float | None = None):
        """threshold: L2 distance cutoff for open-set rejection.
        If None, must be calibrated via calibrate()."""

    def set_prototypes(self, prototypes: torch.Tensor, labels: list[str]) -> None:
        """Register keyword prototypes. prototypes shape: (N, embedding_dim)"""

    def predict(self, query_embedding: torch.Tensor) -> tuple[str, float]:
        """Returns (predicted_keyword, l2_distance).
        Returns ('unknown', min_distance) if min_distance > threshold."""

    def predict_batch(self, query_embeddings: torch.Tensor) -> list[tuple[str, float]]:
        """Batch prediction for evaluation."""

    def calibrate(self, val_embeddings: torch.Tensor, val_labels: list[str],
                  target_far: float = 0.05) -> float:
        """Find threshold that achieves target FAR on validation set.
        Returns calibrated threshold value."""

    def get_distances(self, query_embedding: torch.Tensor) -> dict[str, float]:
        """Return distances to ALL prototypes for visualization."""
```

## Classification Logic (Direct L2 Distance)

```python
def predict(self, query_embedding):
    distances = torch.cdist(query_embedding.unsqueeze(0), self.prototypes).squeeze(0)
    min_dist, min_idx = distances.min(dim=0)

    if min_dist > self.threshold:
        return ("unknown", min_dist.item())
    return (self.labels[min_idx], min_dist.item())
```

Key: Do NOT use softmax probability normalization. Use raw L2 distances directly.
The threshold acts as an "acceptance radius" around each prototype.

## Testing

```python
def test_known_keyword():
    clf = OpenNCMClassifier(threshold=1.0)
    protos = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    clf.set_prototypes(protos, ["yes", "no"])
    pred, dist = clf.predict(torch.tensor([0.9, 0.1]))
    assert pred == "yes"

def test_unknown_rejection():
    clf = OpenNCMClassifier(threshold=0.5)
    protos = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    clf.set_prototypes(protos, ["yes", "no"])
    pred, dist = clf.predict(torch.tensor([0.5, 0.5]))  # equidistant
    assert pred == "unknown"  # dist > threshold

def test_calibration():
    clf = OpenNCMClassifier()
    # calibrate should return a float threshold
    threshold = clf.calibrate(val_emb, val_labels, target_far=0.05)
    assert isinstance(threshold, float)
    assert threshold > 0
```
