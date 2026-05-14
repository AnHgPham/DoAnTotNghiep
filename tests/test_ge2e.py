import torch
import torch.nn.functional as F

from src.models.ge2e import GE2ELoss


def test_ge2e_loss_low_for_separable_episode():
    torch.manual_seed(0)
    centers = F.normalize(torch.eye(3, 8), p=2, dim=-1)
    embs = []
    labels = []
    for i in range(3):
        samples = centers[i].unsqueeze(0) + 0.01 * torch.randn(6, 8)
        embs.append(samples)
        labels.extend([i] * 6)
    embeddings = F.normalize(torch.cat(embs, dim=0), p=2, dim=-1)
    labels_t = torch.tensor(labels)
    loss_fn = GE2ELoss(init_scale=20.0, init_bias=0.0)
    loss = loss_fn(embeddings, labels_t)
    assert loss.item() < 0.2
    assert loss_fn.last_stats["acc"] == 1.0
