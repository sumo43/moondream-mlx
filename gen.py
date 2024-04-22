# Copyright © 2024 Apple Inc.

import unittest
import time

import mlx.core as mx
import requests
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoTokenizer
import numpy as np
from mlx_lm.utils import generate_step

from llava import Moondream

MODEL_PATH = "vikhyatk/moondream2"
PROMPT = "USER: <image>\What is in this picture? Explain, be detailed.\nASSISTANT:"
IMAGE_FILE = "http://images.cocodataset.org/val2017/000000039769.jpg"

#proc = AutoProcessor.from_pretrained()
raw_image = Image.open(requests.get(IMAGE_FILE, stream=True).raw)


model = Moondream.from_pretrained(MODEL_PATH)
model.eval()


image_embeds = model.vision_tower(
    raw_image,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def _tokenize(txt):
    return mx.array(tokenizer(
        txt, return_tensors="pt", add_special_tokens=False
    ).input_ids)

tokenized_text = _tokenize(PROMPT)

text_embed = model.language_model.model.embed_tokens(tokenized_text)


# Add BOS token
embeds = []
embeds.append(
    model.language_model.model.embed_tokens((mx.array([[tokenizer.bos_token_id]])))
)
text_emb = model.language_model.model.embed_tokens

prompt = PROMPT

if "<image>" not in prompt:
    embeds.append(text_emb(_tokenize(prompt)))
else:
    assert prompt.count("<image>") == 1
    before, after = prompt.split("<image>")
    if len(before) > 0:
        embeds.append(text_emb(_tokenize(before)))
    embeds.append(image_embeds)
    if len(after) > 0:
        embeds.append(text_emb(_tokenize(after)))


print([i.shape for i in embeds])

inputs_embeds = mx.concatenate(embeds, axis=1)

logits, cache = model.language_model(
    embeds, cache=None, inputs_embeds=inputs_embeds
)
logits = logits[:, -1, :]

def sample(logits, temperature=0.0):
    if temperature == 0:
        return mx.argmax(logits, axis=-1)
    else:
        return mx.random.categorical(logits * (1 / temperature))

print("generating")

t1 = time.time()

temperature=0.969420
max_tokens = 100

y = sample(logits, temperature=0.969420)
tokens = [y.item()]
for n in range(max_tokens - 1):
    logits, cache = model.language_model(y[None], cache=cache)
    logits = logits[:, -1, :]
    y = sample(logits, temperature)
    token = y.item()
    if token == tokenizer.eos_token_id:
        break
    tokens.append(token)

t2 = time.time()


print(tokenizer.decode(tokens))

print(f"Time taken: {t2 - t1}, tokens: {len(tokens)}, tokens/s: {len(tokens) / (t2 - t1)}")
