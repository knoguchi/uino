# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Uino (初乃)** — Biologically-inspired sensory processing: predictive coding from periphery to cortex.

Crate name: `uino`. Cochlea (Zilany2014) is a separate crate at https://github.com/knoguchi/cochlea-rs.

## Build Commands

```bash
cargo build            # Build
cargo test             # Run tests
cargo test test_name   # Single test
```

## Current Module Layout

- **retina/** — Photoreceptor → Ganglion cells (ON/OFF, midget/parasol). Preserves retinotopy. Kept as-is.
- **cortex/** — Thalamus → A1 → Belt. Will be substantially rewritten per the redesign.
- **ribbon_synapse.rs** — Shared vesicle dynamics, power-law adaptation. Used by both cochlea and retina.
- **spike_generator.rs** — Inhomogeneous Poisson with refractory period.

## Design Principles (from uino_再設計計画書.md)

Three principles govern all design decisions. Do not add mechanisms that violate them.

### Principle A: One canonical microcircuit, tiled per modality

The cortex uses a single computational unit (canonical microcircuit) repeated everywhere. Modality-specific behavior emerges from input geometry and wiring, not from different circuit designs.
- Auditory: tiled over time × frequency
- Visual: tiled over 2D space (retinotopic)
- Same unit, different tiling

### Principle B: Unified learning — predictive coding + temporal continuity

All learning follows one rule: each layer receives top-down predictions, computes error against bottom-up input, sends error upward, updates predictions. No separate reward gating, no ad-hoc oscillators, no scattered STDP.
- Learning is local, online, spiking — no backprop
- Each synapse updates from local pre/post correlation + prediction error
- Temporal continuity (smoothly changing = same object) provides unsupervised signal

### Principle C: Single metric — prediction error spike count

One number measures both learning progress and energy efficiency: total prediction error spikes per inference. Better predictions → fewer error spikes → less energy. This is the "compass" — every design decision should reduce this quantity. If a change doesn't reduce it (or can't predict whether it will), don't make it.

## Target Architecture

```
Periphery (modality-specific, biologically faithful — don't touch)
  cochlea (Zilany2014, separate repo) → frequency × time spikes
  retina (photoreceptor → ganglion)   → 2D spatial spikes (retinotopy preserved)
        │                                    │
Relay (thalamus / LGN+MGN: predictive gating)
        │                                    │
Modality-specific cortical hierarchy (same canonical microcircuit, tiled)
  Auditory: A1 → belt → ...         Visual: V1 → V2 → V4 → IT (new)
  time-frequency RFs emerge          orientation → parts → invariant objects emerge
        └────────────┬──────────────┘
Associative convergence zone (modality-independent)
  Binds sound + object by temporal co-occurrence
```

## What to keep vs replace

**Keep as-is:** retina (periphery), ribbon_synapse, spike_generator, thalamus structure (onset/sustained + predictive suppression)

**Replace internals:** cortex/a1_core.rs, cortex/belt.rs — gut the learning rules, rebuild around predictive coding. Current state: A1 uses reward-gated Hebbian with eligibility traces, Belt uses reward-gated Hebbian without eligibility traces. Both need unified error-driven learning.

**Remove:** cortex/reward.rs (3 separate reward modes → replaced by prediction error as the learning signal), cortex/oscillator.rs (hard phase resets → smooth phase continuity if needed)

**Build new:** Canonical microcircuit implementation, visual cortex hierarchy (V1→V2→V4→IT), convergence zone, prediction error measurement infrastructure.

## Phased Verification Plan

Each phase has explicit tests. Test-driven: write assertions before implementation.

- **Phase 0**: Audit current code. Build prediction error spike counter. Verify retinotopy. Measure baseline.
- **Phase 1**: Implement one canonical microcircuit. Verify: error spikes decrease with learning. Verify prediction with apparent motion illusion (prediction ON → illusion appears, prediction OFF → no illusion).
- **Phase 2**: Tile circuit into visual hierarchy on top of retina via thalamus. Verify: retinotopy preserved, object invariance emerges, cat/dog separate.
- **Phase 3**: Connect both modality hierarchies in convergence zone. Verify: unsupervised audio-visual binding via temporal co-occurrence.
- **Phase 4**: Confirm energy efficiency — more learning = fewer spikes across the whole system.

## Decision Rule

Before adding any mechanism, answer: "Does this reduce prediction error spike count, and can I write a test predicting that?" If no, don't add it.
