import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from sparsify import SaeConfig, Trainer, TrainConfig
from sparsify.data import chunk_and_tokenize

MODEL = "HuggingFaceTB/SmolLM2-135M"
dataset = load_dataset(
    "EleutherAI/SmolLM2-135M-10B", split="train",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenized = chunk_and_tokenize(dataset, tokenizer)


gpt = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map={"": "cuda"},
    torch_dtype=torch.bfloat16,
)

cfg = TrainConfig(SaeConfig(), batch_size=16, hookpoints=["layers.20"] ) # Model has 30 layers, we train on layer 2/3 of the way through.
trainer = Trainer(cfg, tokenized, gpt)

trainer.fit()