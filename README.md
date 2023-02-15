# Lightning + Colossal-AI

**Large-Scale Distributed Model Training with [Colossal AI](https://colossalai.org/) and [Lightning AI](https://lightning.ai)**

______________________________________________________________________

[![CI testing](https://github.com/Lightning-AI/lightning-colossalai/actions/workflows/ci-testing.yml/badge.svg?event=push)](https://github.com/Lightning-AI/lightning-colossalai/actions/workflows/ci-testing.yml)
[![General checks](https://github.com/Lightning-AI/lightning-colossalai/actions/workflows/ci-checks.yml/badge.svg?event=push)](https://github.com/Lightning-AI/lightning-colossalai/actions/workflows/ci-checks.yml)

## Installation

```commandline
pip install -U lightning lightning-colossalai
```

## Usage

Simply set the strategy arugment in the Trainer:

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
