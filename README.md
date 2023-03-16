# Lightning ⚡ Colossal-AI

**Efficient Large-Scale Distributed Training with [Colossal-AI](https://colossalai.org/) and [Lightning AI](https://lightning.ai)**

[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://lightning.ai/)
[![PyPI Status](https://badge.fury.io/py/lightning-colossalai.svg)](https://badge.fury.io/py/lightning-colossalai)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lightning-colossalai)](https://pypi.org/project/lightning-colossalai/)
[![PyPI Status](https://pepy.tech/badge/lightning-colossalai)](https://pepy.tech/project/lightning-colossalai)
[![Deploy Docs](https://github.com/Lightning-AI/lightning-ColossalAI/actions/workflows/docs-deploy.yml/badge.svg)](https://lightning-ai.github.io/lightning-ColossalAI/)

[![General checks](https://github.com/Lightning-AI/lightning-colossalai/actions/workflows/ci-checks.yml/badge.svg?event=push)](https://github.com/Lightning-AI/lightning-colossalai/actions/workflows/ci-checks.yml)
[![CI testing](https://github.com/Lightning-AI/lightning-colossalai/actions/workflows/ci-testing.yml/badge.svg?event=push)](https://github.com/Lightning-AI/lightning-colossalai/actions/workflows/ci-testing.yml)
[![Build Status](https://dev.azure.com/Lightning-AI/compatibility/_apis/build/status/Lightning-AI.lightning-ColossalAI?branchName=main)](https://dev.azure.com/Lightning-AI/compatibility/_build/latest?definitionId=42&branchName=main)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Lightning-AI/lightning-ColossalAI/main.svg)](https://results.pre-commit.ci/latest/github/Lightning-AI/lightning-ColossalAI/main)

______________________________________________________________________

## Installation

```bash
pip install -U lightning lightning-colossalai
```

## Usage

Simply set the strategy argument in the Trainer:

```py
import lightning as L

trainer = L.Trainer(strategy="colossalai", precision="16-mixed", devices=...)
```

For more fine-grained tuning of Colossal-AI's parameters, pass the strategy object to the Trainer:

```py
import lightning as L
from lightning_colossalai import ColossalAIStrategy

strategy = ColossalAIStrategy(...)
trainer = L.Trainer(strategy=strategy, precision="16-mixed", devices=...)
```

Find all configuration options [in the docs](https://pytorch-lightning.readthedocs.io/en/latest/advanced/third_party/colossalai.html)!
