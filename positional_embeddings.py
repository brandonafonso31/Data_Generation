import torch
from torch import nn
from torch.nn import functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SinusoidalEmbedding(nn.Module):
    def __init__(self, size: int, t_emb: str, scale: float):
        super().__init__()
        self.size = size
        self.scale = scale
        self.device = device

    def forward(self, x: torch.Tensor):
        x = x * self.scale
        half_size = self.size // 2
        emb = torch.log(torch.Tensor([10000.0])).to(self.device) / (half_size - 1)
        emb = torch.exp(-emb * torch.arange(half_size).to(self.device))
        # print("x", x.get_device(), "emb", emb.get_device())

        emb = x.unsqueeze(-1) * emb.unsqueeze(0)

        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb

    def __len__(self):
        return self.size


class LinearEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x: torch.Tensor):
        x = x / self.size * self.scale
        return x.unsqueeze(-1)

    def __len__(self):
        return 1


class LearnableEmbedding(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.size = size
        self.linear = nn.Linear(1, size)

    def forward(self, x: torch.Tensor):
        return self.linear(x.unsqueeze(-1).float() / self.size)

    def __len__(self):
        return self.size


class IdentityEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x.unsqueeze(-1)

    def __len__(self):
        return 1


class ZeroEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x.unsqueeze(-1) * 0

    def __len__(self):
        return 1


class PositionalEmbedding(nn.Module):
    def __init__(self, size: int, t_emb: str, scale: float, **kwargs):
        super().__init__()

        if t_emb == "sinusoidal":
            self.layer = SinusoidalEmbedding(size, t_emb, scale)
        elif t_emb == "linear":
            self.layer = LinearEmbedding(size, **kwargs)
        elif t_emb == "learnable":
            self.layer = LearnableEmbedding(size)
        elif t_emb == "zero":
            self.layer = ZeroEmbedding()
        elif t_emb == "identity":
            self.layer = IdentityEmbedding()
        else:
            raise ValueError(f"Unknown positional embedding type: {t_emb}")

    def forward(self, x: torch.Tensor):
        return self.layer(x)
