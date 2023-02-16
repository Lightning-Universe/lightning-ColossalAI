# Lightning + Colossal-AI

**Efficient Large-Scale Distributed Training with [Colossal-AI](https://colossalai.org/) and [Lightning AI](https://lightning.ai)**

[![Documentation](https://readthedocs.org/projects/pytorch-lightning/badge/?version=stable)](https://pytorch-lightning.readthedocs.io/en/latest/advanced/third_party/colossalai.html)
[![CI testing](https://github.com/Lightning-AI/lightning-colossalai/actions/workflows/ci-testing.yml/badge.svg?event=push)](https://github.com/Lightning-AI/lightning-colossalai/actions/workflows/ci-testing.yml)
[![General checks](https://github.com/Lightning-AI/lightning-colossalai/actions/workflows/ci-checks.yml/badge.svg?event=push)](https://github.com/Lightning-AI/lightning-colossalai/actions/workflows/ci-checks.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Lightning-AI/lightning-colossalai/main.svg?badge_token=SP8B_IRmT32JEhKRT6afQg)](https://results.pre-commit.ci/latest/github/Lightning-AI/lightning-colossalai/main?badge_token=SP8B_IRmT32JEhKRT6afQg)

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lightning-colossalai)](https://pypi.org/project/lightning-colossalai/)
[![PyPI Status](https://badge.fury.io/py/lightning-colossalai.svg)](https://badge.fury.io/py/lightning-colossalai)
[![PyPI Status](https://pepy.tech/badge/lightning-colossalai)](https://pepy.tech/project/lightning-colossalai)

______________________________________________________________________

## Installation

```bash
pip install -U lightning lightning-colossalai
```

## Usage

Simply set the strategy argument in the Trainer:

```py
import lightning as L

trainer = L.Trainer(strategy="colossalai", precision=16, devices=...)
```

For more fine-grained tuning of Colossal-AI's parameters, pass the strategy object to the Trainer:

```py
import lightning as L
from lightning_colossalai import ColossalAIStrategy

strategy = ColossalAIStrategy(...)
trainer = L.Trainer(strategy=strategy, precision=16, devices=...)
```

Find all configuration options [in the docs](https://pytorch-lightning.readthedocs.io/en/latest/advanced/third_party/colossalai.html)!
