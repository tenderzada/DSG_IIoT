# # model.py
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# def timestep_embedding(timesteps, dim, max_period=10000):
#     half = dim // 2
#     freqs = torch.exp(
#         -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
#     ).to(device=timesteps.device)
#     args = timesteps[:, None].float() * freqs[None]
#     embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
#     if dim % 2:
#         embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
#     return embedding

# class ResidualBlock1D(nn.Module):
#     def __init__(self, in_channels, out_channels, time_channels, dropout):
#         super().__init__()
#         self.conv1 = nn.Sequential(
#             nn.GroupNorm(32, in_channels),
#             nn.SiLU(),
#             nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
#         )
#         self.time_emb = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(time_channels, out_channels)
#         )
#         self.conv2 = nn.Sequential(
#             nn.GroupNorm(32, out_channels),
#             nn.SiLU(),
#             nn.Dropout(p=dropout),
#             nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
#         )
#         if in_channels != out_channels:
#             self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
#         else:
#             self.shortcut = nn.Identity()

#     def forward(self, x, t):
#         h = self.conv1(x)
#         h += self.time_emb(t)[:, :, None]
#         h = self.conv2(h)
#         return h + self.shortcut(x)

# class UNetModel1D(nn.Module):
#     def __init__(self, in_channels=2, model_channels=128, out_channels=2, num_res_blocks=2, dropout=0):
#         super().__init__()
#         time_embed_dim = model_channels * 4
#         self.time_embed = nn.Sequential(
#             nn.Linear(model_channels, time_embed_dim),
#             nn.SiLU(),
#             nn.Linear(time_embed_dim, time_embed_dim)
#         )
#         self.down_blocks = nn.ModuleList([
#             nn.Sequential(nn.Conv1d(in_channels, model_channels, kernel_size=3, padding=1))
#         ])
#         ch = model_channels
#         for _ in range(num_res_blocks):
#             self.down_blocks.append(
#                 nn.Sequential(ResidualBlock1D(ch, model_channels, time_embed_dim, dropout))
#             )
#         self.middle_block = nn.Sequential(
#             ResidualBlock1D(model_channels, model_channels, time_embed_dim, dropout)
#         )
#         self.up_blocks = nn.ModuleList([])
#         for _ in range(num_res_blocks):
#             self.up_blocks.append(
#                 nn.Sequential(ResidualBlock1D(model_channels, model_channels, time_embed_dim, dropout))
#             )
#         self.out = nn.Sequential(
#             nn.GroupNorm(32, model_channels),
#             nn.SiLU(),
#             nn.Conv1d(model_channels, out_channels, kernel_size=3, padding=1)
#         )

#     def forward(self, x: torch.FloatTensor, timesteps: torch.LongTensor):
#         hs = []
#         t: torch.FloatTensor = self.time_embed(timestep_embedding(timesteps, self.time_embed[0].in_features))
#         h = x
#         for module in self.down_blocks:
#             h = module(h, t)
#             hs.append(h)
#         h = self.middle_block(h, t)
#         for module in self.up_blocks:
#             h = module(h + hs.pop(), t)
#         return self.out(h)


###############
###############

# model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, t):
        pass

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, t):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, t)
            else:
                x = layer(x)
        return x

class ResidualBlock1D(TimestepBlock):
    def __init__(self, in_channels, out_channels, time_channels, dropout):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t):
        h = self.conv1(x)
        h += self.time_emb(t)[:, :, None]
        h = self.conv2(h)
        return h + self.shortcut(x)

class UNetModel1D(nn.Module):
    def __init__(self, in_channels=2, model_channels=128, out_channels=2, num_res_blocks=2, dropout=0):
        super().__init__()
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        self.down_blocks = nn.ModuleList([
            TimestepEmbedSequential(nn.Conv1d(in_channels, model_channels, kernel_size=3, padding=1))
        ])
        ch = model_channels
        for _ in range(num_res_blocks):
            self.down_blocks.append(
                TimestepEmbedSequential(ResidualBlock1D(ch, model_channels, time_embed_dim, dropout))
            )
        self.middle_block = TimestepEmbedSequential(
            ResidualBlock1D(model_channels, model_channels, time_embed_dim, dropout)
        )
        self.up_blocks = nn.ModuleList([])
        for _ in range(num_res_blocks):
            self.up_blocks.append(
                TimestepEmbedSequential(ResidualBlock1D(model_channels, model_channels, time_embed_dim, dropout))
            )
        self.out = nn.Sequential(
            nn.GroupNorm(32, model_channels),
            nn.SiLU(),
            nn.Conv1d(model_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.FloatTensor, timesteps: torch.LongTensor):
        hs = []
        t: torch.FloatTensor = self.time_embed(timestep_embedding(timesteps, self.time_embed[0].in_features))
        h = x
        for module in self.down_blocks:
            h = module(h, t)
            hs.append(h)
        h = self.middle_block(h, t)
        for module in self.up_blocks:
            h = module(h + hs.pop(), t)
        return self.out(h)
