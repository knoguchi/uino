# CLAUDE.md

## Project

**Uino (初乃)** — predictive coding cortex driven by spike streams from peripheral crates [cochlea-rs](https://github.com/knoguchi/cochlea-rs) (auditory) and [retinula](https://github.com/knoguchi/retinula) (visual).

Full design plan: `uino_再設計計画書.md`.

## Build

```bash
cargo build            # Build
cargo test             # All tests
cargo test test_name   # Single test
```

## Layout

- **cortex/** — Thalamus, A1, Belt, oscillator, reward, STRF. Current code uses reward-gated Hebbian; will be rewritten around predictive coding (see design principles below).

Periphery is external: `cochlea-rs` and `retinula` crates.

## Design Principles

Three principles govern all design decisions. Do not add mechanisms that violate them.

### A: One canonical microcircuit, tiled per modality

Cortex uses a single computational unit repeated everywhere. Modality-specific behavior emerges from input geometry and wiring, not from different circuit designs.
- Auditory: tiled over time × frequency
- Visual: tiled over 2D space (retinotopic)

### B: Unified learning — predictive coding + temporal continuity

Each layer receives top-down predictions, computes error against bottom-up input, sends error upward, updates predictions. No separate reward gating, no ad-hoc oscillators, no scattered STDP.
- Local, online, spiking — no backprop
- Each synapse updates from local pre/post correlation + prediction error
- Temporal continuity (smoothly changing = same object) provides unsupervised signal

### C: Single metric — prediction error spike count

One number measures both learning progress and energy efficiency: total prediction error spikes per inference. Better predictions → fewer error spikes → less energy. Every design decision should reduce this.

## Decision Rule

Before adding any mechanism: "Does this reduce prediction error spike count, and can I write a test predicting that?" If no, don't add it.
