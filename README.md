# README

## Setup

Setup environment with uv: `uv sync` and make sure you have dependencies:

**FFmpeg**
```
sudo apt update
sudo apt install ffmpeg
```
**The well**
Download limited dataset for example:
```
the-well-download --dataset turbulent_radiative_layer_2D --basepath ./data
```
You can only use `--first-only` flag to download the first file of the dataset.

**gfortran**
Or F90 compiler

## References
- [Walrus Blog](https://polymathic-ai.org/blog/walrus/)
- [Walrus code](https://github.com/PolymathicAI/walrus)
- [Steering code](https://github.com/DJ-Fear/walrus_steering)
- [Graphcast features](https://arxiv.org/abs/2512.24440)