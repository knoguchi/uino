# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Uino (初乃)** — Biologically-inspired sensory processing. This repo implements the redesign plan in `uino_再設計計画書.md`: predictive coding from periphery to cortex.

The crate name is `uino`. Cochlea (Zilany2014) lives separately in https://github.com/knoguchi/cochlea-rs.

## Build Commands

```bash
cargo build            # Build
cargo test             # Run tests
cargo test test_name   # Single test
```

## Architecture (current, pre-redesign)

- **retina/** — Photoreceptor → Ganglion cells (ON/OFF, midget/parasol). Preserves retinotopy.
- **cortex/** — Thalamus → A1 → Belt. Will be redesigned around predictive coding.
- **ribbon_synapse.rs** — Shared synapse model (vesicle dynamics, power-law adaptation).
- **spike_generator.rs** — Inhomogeneous Poisson with refractory period.

## Redesign Plan

See `uino_再設計計画書.md` for the full plan. Key principles:
- **A**: One canonical microcircuit, tiled per modality
- **B**: Unified learning via predictive coding + temporal continuity
- **C**: Single metric: prediction error spike count = learning progress = energy efficiency

The cortex/ module will be substantially rewritten. Retina and ribbon_synapse are kept.
