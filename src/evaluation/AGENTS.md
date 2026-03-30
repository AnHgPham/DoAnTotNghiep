# Evaluation Directory Agent

## Responsibility
Implement evaluation protocols, metrics computation, and DET curve visualization.

## Files

### `metrics.py` - Metrics Computation

```python
def compute_det_curve(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute single DET curve. Returns (far_values, frr_values)."""

def compute_mean_det(y_true_per_class: list[np.ndarray],
                     scores_per_class: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Average DET curves across all keyword classes."""

def compute_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area Under ROC Curve for binary keyword/non-keyword."""

def compute_acc_at_far(y_true: np.ndarray, y_pred: np.ndarray,
                       scores: np.ndarray, target_far: float = 0.05) -> float:
    """Classification accuracy at specified FAR operating point."""

def compute_frr_at_far(y_true: np.ndarray, scores: np.ndarray,
                       target_far: float = 0.05) -> float:
    """False Rejection Rate at specified FAR."""

def plot_det_curves(curves: dict[str, tuple], save_path: Path = None) -> None:
    """Plot multiple DET curves on log-scale axes. curves = {'label': (far, frr)}"""
```

### `protocols.py` - Testing Protocols

```python
class EvaluationProtocol:
    def __init__(self, dataset: str = "gsc", mode: str = "fixed", n_runs: int = 5):
        """dataset: 'gsc' | 'mswc'. mode: 'fixed' | 'random'.
        n_runs: 5 for mandatory evaluation, 10 for extended."""

    def get_partitions(self, run_idx: int) -> tuple[list[str], list[str]]:
        """Returns (positive_words, negative_words) for given run."""

    def evaluate(self, encoder, classifier, data_loader) -> dict:
        """Run full evaluation. Returns metrics dict averaged over n_runs."""
```

Protocol details:
- **GSC Fixed**: positive=10 IoT words, negative=20 others, 5 excluded
- **GSC Random**: randomly sample 10 positive + 20 negative from 35 words each run
- **MSWC**: randomly sample 5 positive + 50 negative from 263 words, split 1:9

## DET Curve Plotting

- Use matplotlib with log-scale axes (both x and y)
- X-axis: FAR (%), Y-axis: FRR (%)
- Include grid lines at 1%, 2%, 5%, 10%, 20%, 50%
- Each curve clearly labeled with legend

## Testing

```python
def test_det_curve_shape():
    y_true = np.array([1, 1, 0, 0, 1])
    scores = np.array([0.9, 0.7, 0.3, 0.2, 0.8])
    far, frr = compute_det_curve(y_true, scores)
    assert len(far) == len(frr)
    assert far[0] <= far[-1]  # monotonically increasing

def test_perfect_classifier_auc():
    y_true = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.1, 0.2])
    assert compute_auc(y_true, scores) == 1.0
```
