# Copyright © 2024 Apple Inc.

import glob
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import phi

import numpy as np
from huggingface_hub import snapshot_download
from language import Model
from vision import VisionConfig, VisionEncoder

@dataclass
class LlaVAConfig:
    vision_config: VisionConfig = VisionConfig("moondream")
    ignore_index: int = -100
    image_token_index: int = 32000
    vision_feature_select_strategy: str = "default"
    vision_feature_layer: int = -2
    vocab_size: int = 32000

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )

class Moondream(nn.Module):
    def __init__(self, config: LlaVAConfig):
        self.config = config
        self.vision_tower = VisionEncoder()
        args = phi.ModelArgs(
           max_position_embeddings=2048, 
           vocab_size=51200, 
           hidden_size=2048, 
           num_attention_heads=32, 
           num_hidden_layers=24, 
           num_key_value_heads=None, 
           partial_rotary_factor=0.5, 
           intermediate_size=8192, 
           layer_norm_eps=1e-05, 
           rope_theta=10000.0
        )
        self.language_model = Model(args)
        #self.multi_modal_projector = LlavaMultiModalProjector(config)
        self.vision_feature_layer = config.vision_feature_layer
        self.vision_feature_select_strategy = config.vision_feature_select_strategy

    @staticmethod
    def from_pretrained(path_or_hf_repo: str):
        path = Path(path_or_hf_repo)
        if not path.exists():
            path = Path(
                snapshot_download(
                    repo_id=path_or_hf_repo,
                    allow_patterns=[
                        "*.json",
                        "*.safetensors",
                        "*.py",
                        "tokenizer.model",
                        "*.tiktoken",
                    ],
                )
            )

        with open(path / "config.json", "r") as f:
            model_config = json.load(f)


        model_config = LlaVAConfig.from_dict(model_config)


        model_config.vision_config = VisionConfig("moondream")#.from_dict(model_config.vision_config)
        #model_config.text_config = TextConfig.from_dict(model_config.text_config)

        model = Moondream(model_config)
        weight_files = glob.glob(str(path / "*.safetensors"))
        if not weight_files:
            raise FileNotFoundError(f"No safetensors found in {path}")

        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf))
        

        def to_wqkv(k, weights):
            _q = weights[k][:weights[k].shape[0]//3]
            _k = weights[k][weights[k].shape[0]//3:2*weights[k].shape[0]//3]
            _v = weights[k][2*(weights[k].shape[0]//3):]
            return _q, _k, _v

        # attn

        k_to_del = []
        k_to_add = {}

        for i in range(24):
            q, k, v = to_wqkv(f'text_model.transformer.h.{i}.mixer.Wqkv.weight', weights)
            k_to_del.append(f'text_model.transformer.h.{i}.mixer.Wqkv.weight')
            k_to_add[f'language_model.model.layers.{i}.self_attn.q_proj.weight'] = q
            k_to_add[f'language_model.model.layers.{i}.self_attn.k_proj.weight'] = k
            k_to_add[f'language_model.model.layers.{i}.self_attn.v_proj.weight'] = v
            dense = weights[f'text_model.transformer.h.{i}.mixer.out_proj.weight']
            k_to_del.append(f'text_model.transformer.h.{i}.mixer.out_proj.weight')
            k_to_add[f'language_model.model.layers.{i}.self_attn.dense.weight'] = dense


            q_b, k_b, v_b = to_wqkv(f'text_model.transformer.h.{i}.mixer.Wqkv.bias', weights)

            k_to_del.append(f'text_model.transformer.h.{i}.mixer.Wqkv.bias')
            k_to_add[f'language_model.model.layers.{i}.self_attn.q_proj.bias'] = q_b
            k_to_add[f'language_model.model.layers.{i}.self_attn.k_proj.bias'] = k_b
            k_to_add[f'language_model.model.layers.{i}.self_attn.v_proj.bias'] = v_b

            dense_b = weights[f'text_model.transformer.h.{i}.mixer.out_proj.bias']
            k_to_del.append(f'text_model.transformer.h.{i}.mixer.out_proj.bias')
            k_to_add[f'language_model.model.layers.{i}.self_attn.dense.bias'] = dense_b


        weights.update(k_to_add)

        for k in k_to_del:
            del weights[k]

        # ln

        k_to_del = []
        k_to_add = {}
        for i in range(24):
            weight, bias = f'text_model.transformer.h.{i}.ln.weight', f'text_model.transformer.h.{i}.ln.bias'

            k_to_del.append(weight)
            k_to_del.append(bias)

            k_to_add[f'language_model.model.layers.{i}.input_layernorm.weight'] = weights[weight]
            k_to_add[f'language_model.model.layers.{i}.input_layernorm.bias'] = weights[bias]

        weights.update(k_to_add)

        for k in k_to_del:
            del weights[k]
        # mlp
        k_to_del = []
        k_to_add = {}
        for i in range(24):

            weight, bias = f'text_model.transformer.h.{i}.mlp.fc1.weight', f'text_model.transformer.h.{i}.mlp.fc1.bias'
            k_to_del.append(weight)
            k_to_del.append(bias)

            k_to_add[f'language_model.model.layers.{i}.mlp.fc1.weight'] = weights[weight]
            k_to_add[f'language_model.model.layers.{i}.mlp.fc1.bias'] = weights[bias]

            weight, bias = f'text_model.transformer.h.{i}.mlp.fc2.weight', f'text_model.transformer.h.{i}.mlp.fc2.bias'
            k_to_del.append(weight)
            k_to_del.append(bias)

            k_to_add[f'language_model.model.layers.{i}.mlp.fc2.weight'] = weights[weight]
            k_to_add[f'language_model.model.layers.{i}.mlp.fc2.bias'] = weights[bias]

        weights.update(k_to_add)

        for k in k_to_del:
            del weights[k]


        # embed tokens

        key = 'text_model.transformer.embd.wte.weight'
        weights['language_model.model.embed_tokens.weight'] = weights[key]
        del weights[key]


        # lm head 
        w, b = 'text_model.lm_head.linear.weight', 'text_model.lm_head.linear.bias'
        weights['language_model.lm_head.weight'] = weights[w]   
        weights['language_model.lm_head.bias'] = weights[b]

        del weights[w]
        del weights[b]

        # final layer norm 

        w, b = 'text_model.lm_head.ln.weight', 'text_model.lm_head.ln.bias'
        weights['language_model.model.final_layernorm.weight'] = weights[w]
        weights['language_model.model.final_layernorm.bias'] = weights[b]

        del weights[w]
        del weights[b]



        ### VISION MODEL ###

        del_k = []
        add_k = {}

        for k in weights.keys():
            if "vision_encoder" in k:
                new_k = k.replace('vision_encoder', 'vision_tower')

                del_k.append(k)

                add_k[new_k] = weights[k]
        
        for k in del_k:
            del weights[k]
        weights.update(add_k)

        k_to_del = []
        k_to_add = {}


        for i in range(27):
            q, k, v = to_wqkv(f'vision_tower.encoder.model.visual.blocks.{i}.attn.qkv.weight', weights)
            k_to_del.append(f'vision_tower.encoder.model.visual.blocks.{i}.attn.qkv.weight')
            k_to_add[f'vision_tower.encoder.model.visual.blocks.{i}.attn.q_proj.weight'] = q
            k_to_add[f'vision_tower.encoder.model.visual.blocks.{i}.attn.k_proj.weight'] = k
            k_to_add[f'vision_tower.encoder.model.visual.blocks.{i}.attn.v_proj.weight'] = v

            proj= weights[f'vision_tower.encoder.model.visual.blocks.{i}.attn.proj.weight']

            k_to_del.append(f'vision_tower.encoder.model.visual.blocks.{i}.attn.proj.weight')
            k_to_add[f'vision_tower.encoder.model.visual.blocks.{i}.attn.out_proj.weight'] = proj

            q, k, v = to_wqkv(f'vision_tower.encoder.model.visual.blocks.{i}.attn.qkv.bias', weights)
            k_to_del.append(f'vision_tower.encoder.model.visual.blocks.{i}.attn.qkv.bias')
            k_to_add[f'vision_tower.encoder.model.visual.blocks.{i}.attn.q_proj.bias'] = q
            k_to_add[f'vision_tower.encoder.model.visual.blocks.{i}.attn.k_proj.bias'] = k
            k_to_add[f'vision_tower.encoder.model.visual.blocks.{i}.attn.v_proj.bias'] = v

            proj= weights[f'vision_tower.encoder.model.visual.blocks.{i}.attn.proj.bias']

            k_to_del.append(f'vision_tower.encoder.model.visual.blocks.{i}.attn.proj.bias')
            k_to_add[f'vision_tower.encoder.model.visual.blocks.{i}.attn.out_proj.bias'] = proj


        weights.update(k_to_add)

        for k in k_to_del:
            del weights[k]

        k_to_del = []
        k_to_add = {}

        for key in weights.keys():
            if 'vision_tower' in key:
                new_k = key.replace('blocks', 'blocks.layers')

                k_to_del.append(key)
                k_to_add[new_k] = weights[key]

        for k in k_to_del:
            del weights[k]

        weights.update(k_to_add)

        #weights = VisionModel.sanitize(weights)
        #weights = LanguageModel.sanitize(weights)

        model.load_weights(list(weights.items()))
        return model
