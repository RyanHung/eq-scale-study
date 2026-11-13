import torch

"""
`center(pos, mask)` subtracts each molecule’s centroid from its atom coordinates, using only real atoms.
QM9 batches pad every molecule to the same number of atoms. Without the mask, the padded atoms would 
drag the centroid off. This removes translation from the picture.

`random_rotate(pos, reflections=True)` gives every molecule in the batch its own random rotation. It 
does this by drawing a random 3×3 Gaussian matrix and taking its QR decomposition. That gives a 
rotation drawn uniformly from all orthogonal matrices.
    
"""

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
