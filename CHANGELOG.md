# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2023-03-22

### Added

- Added ColossalAIStrategy ([#1](https://github.com/Lightning-AI/lightning-colossalai/pull/1))
- Added overview to local docs ([#16](https://github.com/Lightning-AI/lightning-colossalai/pull/16))

### Changed

- Changed precision argument to only support `precision="16-mixed"` argument ([#9](https://github.com/Lightning-AI/lightning-colossalai/pull/9))

### Fixed

- Allowed using lightning and PL (
    [#10](https://github.com/Lightning-AI/lightning-colossalai/pull/10),
    [#15](https://github.com/Lightning-AI/lightning-colossalai/pull/15)
)
- Validated `configure_sharded_model` hook is overridden ([#12](https://github.com/Lightning-AI/lightning-colossalai/pull/12))
