# README

Interpreting a Walrus.

## Setup

Setup environment with uv: `uv sync`

Make sure you have dependencies:

**walrus**

A branch of my fork is included as a submodule. Install without dependencies.
```
git submodule update --init --recursive
cd src/walrus
uv pip install . --no-deps
```

**FFmpeg**
```
sudo apt update
sudo apt install ffmpeg
```
**The well**

Download limited dataset for example:
```
the-well-download --dataset shear_flow --basepath ./data
```
You can only use `--first-only` flag to download the first file of the dataset.


## References
- [Walrus Blog](https://polymathic-ai.org/blog/walrus/)
- [Walrus code](https://github.com/PolymathicAI/walrus)
- [Steering code](https://github.com/DJ-Fear/walrus_steering)
- Graphcast interpretability ([paper](https://arxiv.org/abs/2512.24440))([github](https://github.com/theodoremacmillan/graphcast-interpretability))([graphcase-sae](https://github.com/theodoremacmillan/graphcast/tree/sae-hooks))