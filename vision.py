# Copyright © 2024 Apple Inc.

import inspect
import math
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from torchvision.transforms.v2 import (
    Compose,
    Resize,
    InterpolationMode,
    ToImage,
    ToDtype,
    Normalize,
)

import torch

@dataclass
class VisionConfig:
    model_type: str
    num_hidden_layers: int = 24
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_attention_heads: int = 16
    image_size: int = 336
    patch_size: int = 14
    projection_dim: int = 768
    vocab_size: int = 32000
    num_channels: int = 3
    layer_norm_eps: float = 1e-5

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


class Attention(nn.Module):
    def __init__(
        self,
        dims: int,
        num_heads: int = 16,
        query_input_dims: Optional[int] = None,
        key_input_dims: Optional[int] = None,
        value_input_dims: Optional[int] = None,
        value_dims: Optional[int] = None,
        value_output_dims: Optional[int] = None,
        bias: bool = True,
    ):
        super().__init__()

        if (dims % num_heads) != 0:
            raise ValueError(
                "The input feature dimensions should be divisible by the "
                f"number of heads ({dims} % {num_heads}) != 0"
            )

        query_input_dims = query_input_dims or dims
        key_input_dims = key_input_dims or dims
        value_input_dims = value_input_dims or key_input_dims
        value_dims = value_dims or dims
        value_output_dims = value_output_dims or dims

        self.num_heads = num_heads
        self.q_proj = nn.Linear(query_input_dims, dims, bias=bias)
        self.k_proj = nn.Linear(key_input_dims, dims, bias=bias)
        self.v_proj = nn.Linear(value_input_dims, value_dims, bias=bias)
        self.out_proj = nn.Linear(value_dims, value_output_dims, bias=bias)

    def __call__(self, queries, keys, values, mask=None):
        queries = self.q_proj(queries)
        keys = self.k_proj(keys)
        values = self.v_proj(values)

        num_heads = self.num_heads
        B, L, D = queries.shape
        _, S, _ = keys.shape
        queries = queries.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, S, num_heads, -1).transpose(0, 2, 3, 1)
        values = values.reshape(B, S, num_heads, -1).transpose(0, 2, 1, 3)

        scale = math.sqrt(1 / queries.shape[-1])
        scores = (queries * scale) @ keys
        if mask is not None:
            scores = scores + mask.astype(scores.dtype)
        scores = mx.softmax(scores, axis=-1)
        values_hat = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.out_proj(values_hat)


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approx='fast')
        self.fc2 = nn.Linear(hidden_features, out_features)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class VitBlock(nn.Module):
    def __init__(self, embed_dim, use_flash_attn=False):
        super().__init__()
        self.attn = Attention(embed_dim)
        self.mlp = MLP(embed_dim, 4304)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def __call__(self, x):

        x_i = self.norm1(x)
        x = x + self.attn(x_i, x_i, x_i)
        x_i = self.norm2(x)
        x = x + self.mlp(x_i)
        return x

class Encoder(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.layers = [EncoderLayer(config) for _ in range(config.num_hidden_layers)]


class LinearPatchEmbedding(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(588, 1152)

    def __call__(self, x):
        b, c, hp1, wp2 = x.shape
        p1, p2 = 14, 14
        h, w = hp1 // p1, wp2 // p2
        x = x.reshape(b, c, h, p1, w, p2)
        x = x.transpose(0, 2, 4, 1, 3, 5)
        x = x.reshape(b, h * w, c * p1 * p2)

        return self.linear(x)
    
class VisionTransformer(nn.Module):

    def __init__(self, use_flash_attn=False):
        super().__init__()

        embed_len = 729
        embed_dim = 1152

        self.patch_embed = LinearPatchEmbedding()
        self.pos_embed = mx.random.normal((1, embed_len, embed_dim)) * 0.02
        self.blocks = nn.Sequential(
            *[VitBlock(embed_dim, use_flash_attn=use_flash_attn) for _ in range(27)]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def __call__(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.blocks(x)
        return self.norm(x)

class VisionEmbeddings(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = mx.zeros((config.hidden_size,))

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def __call__(self, x: mx.array) -> mx.array:
        batch_size = x.shape[0]
        patch_embeddings = self.patch_embedding(x)
        patch_embeddings = mx.flatten(patch_embeddings, start_axis=1, end_axis=2)
        embed_dim = patch_embeddings.shape[-1]
        cls_embeddings = mx.broadcast_to(
            self.class_embedding, (batch_size, 1, embed_dim)
        )
        embeddings = mx.concatenate((cls_embeddings, patch_embeddings), axis=1)
        embeddings += self.position_embedding.weight
        return embeddings

class VisionProjection(nn.Module):
    def __init__(self):
        super().__init__()

        image_embedding_dim = 1152
        model_dim = 2048
        hidden_dim = model_dim * 4

        self.mlp = MLP(image_embedding_dim, hidden_dim, model_dim)

    @property
    def device(self):
        return self.mlp.fc1.weight.device

    def __call__(self, x):
        return self.mlp(x)


class EncoderWrapper(nn.Module):

    def __init__(self, use_flash_attn=False):
        super().__init__()
        self.model = VisualWrapper()

    def __call__(self, x):
        return self.model(x)


class VisualWrapper(nn.Module):

    def __init__(self, use_flash_attn=False):
        super().__init__()
        self.visual = VisionTransformer(use_flash_attn)

    def __call__(self, x):
        return self.visual(x)

class VisionEncoder(nn.Module):

    def __init__(self, use_flash_attn=False):
        super().__init__()

        self.encoder = EncoderWrapper(use_flash_attn)
        self.projection = VisionProjection()


        self.preprocess = Compose(
            [
                Resize(size=(378, 378), interpolation=InterpolationMode.BICUBIC),
                ToImage(),
                ToDtype(torch.float32, scale=True),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )


    @property
    def device(self):
        return self.projection.mlp.fc1.weight.device

    @property
    def dtype(self):
        return self.projection.mlp.fc1.weight.dtype

    def __call__(self, images) -> mx.array:
        if not isinstance(images, list) and not isinstance(images, mx.array):
            images = [images]

        # Skip preprocess if images are already tensors
        if not isinstance(images, mx.array) and not isinstance(
            images[0], mx.array
        ):
            images = [self.preprocess(image.convert("RGB")) for image in images]
        
        if isinstance(images, list):
            images = torch.stack(images)
        

        #x = images.to(self.device, dtype=self.dtype)
        x = mx.array(images)
        x = self.encoder(x)
        x = self.projection(x)

        return x