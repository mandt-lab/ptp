import torch


class BinaryFloatEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.u_adapter = torch.nn.Linear(32, embedding_dim)

    def forward(self, u):
        res = u.to(torch.float32).view(torch.int32)
        bit_masks = 2 ** torch.arange(31, -1, -1, device=u.device, dtype=torch.int32)
        bits_tensor = ((res[..., None] & bit_masks[None, None, :]) != 0).int()
        # u_reconstruction = (bits_tensor.to(torch.int32) * bit_masks[None, None, :]).sum(dim=2).to(torch.int32).view(torch.float32)
        # assert torch.all(
        #     u == u_reconstruction
        # ), u[u != u_reconstruction]
        return self.u_adapter(bits_tensor.to(self.u_adapter.weight.dtype))


class SawtoothFloatEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim, num_bins=32):
        super().__init__()
        self.bin_widths = 1 / torch.arange(1, num_bins + 1).float()
        self.embedding = torch.nn.Linear(num_bins, embedding_dim)

    def forward(self, u):
        # u is (batch_size, seq_len)
        u = torch.clamp(u, 0.0, 1.0 - 1e-6)
        u = u.unsqueeze(-1)  # (batch_size, seq_len, 1)
        # (batch_size, seq_len, num_bins)
        bin_positions = u % self.bin_widths.to(u.device)[None, None, :]
        return self.embedding(bin_positions)


class QuarterCosEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim, num_frequencies=32):
        super().__init__()
        self.frequencies = torch.arange(1, num_frequencies + 1).float()
        assert self.frequencies.shape == (num_frequencies,), f"{self.frequencies.shape} != {(num_frequencies,)}"
        self.embedding = torch.nn.Linear(num_frequencies, embedding_dim)

    def forward(self, u):
        # u is (batch_size, seq_len)
        u = torch.clamp(u, 0.0, 1.0)
        u = u.unsqueeze(-1)  # (batch_size, seq_len, 1)
        # (batch_size, seq_len, num_frequencies)
        cos_features = torch.cos(self.frequencies.to(u.device)[None, None, :] * torch.pi * u)
        return self.embedding(cos_features)


class LinearInterpolationEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim, num_embeddings=3):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, u):
        # u is (batch_size, seq_len)
        num = self.embedding.num_embeddings
        if num == 1:
            emb = self.embedding.weight[0].to(u.device)
            return emb.unsqueeze(0).unsqueeze(0).expand(u.shape[0], u.shape[1], -1)
        scaled = u * (num - 1)
        lower_f = torch.floor(scaled)
        lower = lower_f.long()
        upper = torch.clamp(lower + 1, max=num - 1)
        frac = (scaled - lower_f).unsqueeze(-1)
        return self.embedding(lower) * (1.0 - frac) + self.embedding(upper) * frac


class RoundingEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim, num_bins=32):
        super().__init__()
        self.num_bins = num_bins
        self.embedding = torch.nn.Embedding(num_bins, embedding_dim)

    def forward(self, u):
        # u is (batch_size, seq_len)
        # Clamp u to [0, 1 - 1/(2*num_bins)] so that the upper edge of the last bin is not included
        u = torch.clamp(u, 0.0, 1.0 - 1 / (2 * self.num_bins))
        bin_indices = (u * self.num_bins).long()
        return self.embedding(bin_indices)
