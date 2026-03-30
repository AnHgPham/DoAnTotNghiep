# Models Directory Agent

## Responsibility
This directory contains the neural network architectures: DSCNN encoder and Prototypical Network training logic.

## Files

### `dscnn.py` - Depthwise Separable CNN Encoder
- DSCNN-L is the primary model. DSCNN-S (64ch, 22K params) is optional for ablation study only.
- DS blocks: DepthwiseConv2d -> BatchNorm -> ReLU -> Conv2d(1x1) -> BatchNorm -> ReLU
- Last DS block (Block 5): DW still uses BN+ReLU, only post-PW uses LayerNorm(elementwise_affine=False)
- L2-normalization is applied OUTSIDE the model (in ReprModel / training wrapper via `F.normalize`)
- Three feature extraction modes: CONV (raw conv output), RELU (after ReLU), NORM (after L2-norm externally)
- Input shape: `(batch, 1, T, n_features)` = `(B, 1, 47, 10)` -- MFCC after narrow(10) + transpose. T=47 with n_fft=1024, center=False
- Output shape: `(batch, embedding_dim)` where embedding_dim=276 for DSCNN-L

```python
class DSCNN(nn.Module):
    def __init__(self, model_size: str = "L", feature_mode: str = "NORM"):
        # model_size: "L" (276ch, embedding_dim=276) or "S" (64ch, embedding_dim=64)
        # feature_mode: "CONV" | "RELU" | "NORM"
        # DSCNN-L: self.embedding_dim = 276
        # DSCNN-S: self.embedding_dim = 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (B, 1, 47, 10). Returns: (B, 276) for DSCNN-L.
        # L2-norm is NOT applied inside forward() -- caller applies F.normalize.
```

DSCNN-L architecture (following Rusci et al. `model_size_info_DSCNNL`):
- Initial: ZeroPad2d + Conv2d(1, 276, kernel=(10,4), stride=(2,1), bias=True) + BN + ReLU
- DS Block 1: ZeroPad2d + DW(276, k=3x3, stride=(2,2), groups=276) + BN + ReLU + PW(276,276) + BN + ReLU
- DS Block 2-4: ZeroPad2d + DW(276, k=3x3, stride=(1,1), groups=276) + BN + ReLU + PW(276,276) + BN + ReLU
- DS Block 5: ZeroPad2d + DW(276, k=3x3, stride=(1,1), groups=276) + BN + ReLU + PW(276,276) + LayerNorm([276,H,W], elementwise_affine=False)
- AvgPool (global) -> Flatten -> output (B, 276)
- NO Linear projection layer -- embedding_dim = channel count = 276
- L2-norm applied externally: `F.normalize(embedding, p=2, dim=-1)`

Padding: ALL convolutions use padding=0. Padding is applied via nn.ZeroPad2d BEFORE each conv.
Padding is asymmetric, computed dynamically to achieve "SAME" output (ceil(input/stride)).
All Conv2d use bias=True (default).

Shape propagation (input 47x10):
- Init: (47,10) -> pad -> Conv k(10,4) s(2,1) -> (24,10)
- DS1:  (24,10) -> pad -> DW k(3,3) s(2,2) -> (12,5) -> PW -> (12,5)
- DS2-5: (12,5) -> pad -> DW k(3,3) s(1,1) -> (12,5) -> PW -> (12,5)
- LayerNorm on Block 5: nn.LayerNorm([276, 12, 5])
- AvgPool(12,5) -> (1,1) -> Flatten -> (276,)

DSCNN-S architecture (optional, for ablation only):
- 64 channels, 4 DS blocks, embedding_dim=64
- Initial stride = (2,2) -- different from L's (2,1)

Verify param count after implementation: `sum(p.numel() for p in model.parameters())`.

### `prototypical.py` - Training with Triplet Loss
- Episodic batch construction: sample N classes, K samples per class
- Triplet mining: anchor + positive (same class) + negative (different class, random)
- Loss: `max(0, d(anchor, positive) - d(anchor, negative) + margin)`
- margin = 0.5, distance = L2

```python
class EpisodicBatchSampler:
    def __init__(self, labels, n_classes=80, n_samples=20): ...

class TripletLoss(nn.Module):
    def __init__(self, margin: float = 0.5): ...
    def forward(self, embeddings, labels) -> torch.Tensor: ...

def train_one_epoch(encoder, dataloader, optimizer, loss_fn) -> dict:
    # Returns {'loss': float, 'num_episodes': int}
```

## Testing

```python
def test_dscnn_l_output_shape():
    model = DSCNN(model_size="L", feature_mode="CONV")
    x = torch.randn(4, 1, 47, 10)  # (batch, channel, T, n_features)
    out = model(x)
    assert out.shape == (4, 276)  # embedding_dim=276 for DSCNN-L

def test_dscnn_l_l2_norm_external():
    model = DSCNN(model_size="L", feature_mode="NORM")
    x = torch.randn(4, 1, 47, 10)
    out = model(x)  # raw output, not yet L2-normalized
    out_normed = F.normalize(out, p=2, dim=-1)  # L2-norm applied externally
    assert torch.allclose(out_normed.norm(dim=-1), torch.ones(4), atol=1e-5)

def test_dscnn_s_output_shape():
    model = DSCNN(model_size="S", feature_mode="CONV")
    x = torch.randn(4, 1, 47, 10)
    out = model(x)
    assert out.shape == (4, 64)  # embedding_dim=64 for DSCNN-S

def test_triplet_loss():
    loss_fn = TripletLoss(margin=0.5)
    emb = torch.randn(10, 276)  # embedding_dim=276 for DSCNN-L
    labels = torch.tensor([0,0,1,1,2,2,3,3,4,4])
    loss = loss_fn(emb, labels)
    assert loss >= 0
```
