import torch


def center(pos, mask):
    """pos (B,N,3), mask (B,N,1): subtract each molecule's centroid over real atoms; padding stays 0."""
    n = mask.sum(dim=1, keepdim=True).clamp(min=1)
    return (pos - (pos * mask).sum(dim=1, keepdim=True) / n) * mask


def random_rotate(pos, reflections=True):
    """One uniformly random rotation (plus reflection if reflections=True) per molecule."""
    A = torch.randn(pos.size(0), 3, 3, device=pos.device, dtype=pos.dtype)
    Q, R = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diagonal(R, dim1=-2, dim2=-1)).unsqueeze(-2)  # uniform on O(3)
    if not reflections:
        Q = Q * torch.det(Q).sign().view(-1, 1, 1)
    return torch.einsum('bij,bnj->bni', Q, pos)
