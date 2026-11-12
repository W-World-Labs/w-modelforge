"""The upstream Hadamard/RHT rules, independent of model checkpoint configuration.

Moved unchanged from opgraph.codes so the quantizer and packed loader share them.
"""

# Original rules: Shin Megami Boson, MIT; W-World-Labs relocation (2026-11-12).

from __future__ import annotations

import math

import torch


def _is_prime(q: int) -> bool:
    return q > 1 and all(q % d for d in range(2, math.isqrt(q) + 1))


def _paley(q: int) -> torch.Tensor:
    """Paley's first construction: a ±1 Hadamard matrix of order q + 1, for a prime q ≡ 3 (mod 4)."""
    squares = {(x * x) % q for x in range(1, q)}
    chi = torch.tensor([0.0] + [1.0 if a in squares else -1.0 for a in range(1, q)], dtype=torch.float64)
    Q = chi[(torch.arange(q)[None, :] - torch.arange(q)[:, None]) % q]
    S = torch.zeros(q + 1, q + 1, dtype=torch.float64)
    S[0, 1:], S[1:, 0], S[1:, 1:] = 1, -1, Q
    return S + torch.eye(q + 1, dtype=torch.float64)


def hadamard(n: int) -> torch.Tensor:
    """An orthogonal Hadamard matrix of order n = 2^a·p with p = 1, 2 or q + 1 for a prime q ≡ 3 (mod 4):
    Sylvester's construction for the power of two, Kronecker with Paley's for the rest (960 = 16·60, 2560 = 128·20)."""
    A, B = hadamard_factors(n)
    return torch.kron(A, B) / math.sqrt(n)


def hadamard_factors(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A and B with hadamard(n) = (A ⊗ B) / √n, as small as the construction allows: Sylvester's matrix of order 2^a is itself
    the product of those of orders 2^(a//2) and 2^(a − a//2), so a pure power of two splits in the middle."""
    for a in range(int(math.log2(n)), -1, -1):
        if n % 2**a:
            continue
        p = n // 2**a
        if p in (1, 2) or (_is_prime(p - 1) and (p - 1) % 4 == 3):
            H = torch.ones(1, 1, dtype=torch.float64)
            while H.shape[0] < 2**a:
                H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
            P = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float64) if p == 2 else torch.ones(1, 1, dtype=torch.float64) if p == 1 else _paley(p - 1)
            if p == 1 and a > 1:
                return _sylvester(a // 2), _sylvester(a - a // 2)
            return H, P
    raise ValueError(f"no Hadamard construction here for order {n}")


def _sylvester(a: int) -> torch.Tensor:
    H = torch.ones(1, 1, dtype=torch.float64)
    while H.shape[0] < 2**a:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H


def rht(n: int, seed: int) -> torch.Tensor:
    """A randomized Hadamard transform H·diag(s), random signs s: orthogonal, and it spreads any weight across all n inputs."""
    s = torch.where(torch.rand(n, generator=torch.Generator().manual_seed(seed), dtype=torch.float64) < 0.5, -1.0, 1.0)
    return hadamard(n) * s[None, :]
