# Uino (初乃)

Biologically-inspired sensory processing in Rust.

**Uino** models how mammals perceive the world — from photons and pressure waves to neural spikes. Named 初乃 — a Japanese girl's name meaning "first" (初) with a soft, elegant ending (乃). A nod to Hatsune Miku (初音, "first sound"), but she's learning to hear, not sing.

## What it does

Sound and light enter as continuous signals. Uino transforms them into spike trains, the language of the brain.

```
Sound  →  Middle Ear  →  Cochlea  →  Auditory Nerve  →  Cortex
Light  →  Retina  →  Ganglion Cells  →  Thalamus  →  Cortex
```

## Modules

### Auditory pathway
- **middle_ear** — acoustic filtering
- **ihc** — inner hair cell mechanoelectrical transduction
- **synapse** — auditory nerve synapse with power-law adaptation
- **spike_generator** — stochastic spike generation

### Visual pathway
- **photoreceptor** — cone/rod phototransduction
- **ganglion** — ON/OFF center-surround receptive fields
- **ribbon_synapse** — shared with cochlea (see below)

### Cortex
- **thalamus** — sensory relay and gating
- **a1_core** — primary auditory cortex
- **belt** — higher auditory processing
- **strf** — spectro-temporal receptive fields
- **oscillator** — theta-gamma neural oscillations

### Shared
- **ribbon_synapse** — the same synaptic machinery used by both ear and eye

## Key insight: Ribbon Synapses

Both cochlea and retina use **ribbon synapses** — specialized structures for sustained, graded neurotransmitter release. This isn't a software abstraction; it's biology. The `ribbon_synapse` module implements this shared mechanism.

> "Sensory Processing at Ribbon Synapses in the Retina and the Cochlea"

## Motivation: How babies learn language

A mother says "mama" to her infant. What happens next?

```
"mama" → air pressure waves
       → cochlea (frequency decomposition)
       → auditory nerve (spike patterns)
       → cortex (phoneme recognition)
       → learning (statistical patterns)
```

Babies don't learn language from text. They learn from sound — processed through the same biological pathway Uino models.

By 6 months, infants distinguish phonemes in any language. By 12 months, they've tuned to their native tongue. A Japanese baby stops hearing the difference between "r" and "l". An English baby loses sensitivity to pitch-accent distinctions.

This is **perceptual narrowing** — the brain optimizes for the sounds that matter.

Uino lets you simulate this journey:
- Generate speech with the **vocal_tract** module
- Process it through the **cochlea**
- Learn patterns in the **cortex**
- Watch a simulated infant become tuned to English, Japanese, or any language

The goal: understand how raw acoustic input becomes language.

## Features

- **Biologically grounded** — based on peer-reviewed models
- **O(n) power-law adaptation** — fixes O(n²) bug in original implementations
- **Parallel processing** — uses Rayon for multi-channel computation
- **Python bindings** — via PyO3 for easy integration

## Why: 20 Watts vs 1 Gigawatt

The human brain runs on 20W. AI data centers need 1GW. Why the 50,000,000x gap?

Hypothesis: we're solving the wrong problems. The cochlea already transforms sound into neural-ready features — optimized by 300 million years of evolution. We don't need to learn that from scratch.

Early experiment on CMU Arctic phoneme classification:

| Classifier | Accuracy | Compute |
|------------|----------|---------|
| **K-means** | ~53% | 230M ops |
| Reservoir | 47.3% | 2,770M ops |

The simple classifier wins. **12x less compute, better accuracy.**

This is a hint, not a proof. Reservoir computing tries to learn temporal structure that the cochlea already encodes — perhaps that's redundant work.

Uino is an exploration of this idea: **copy biology, don't compete with it.**

We're not there yet. But that's the direction.

## Installation

### Rust

```toml
[dependencies]
uino = "0.1"
```

### Python

```bash
pip install maturin
maturin develop --release
```

## Usage

### Rust

```rust
use uino::{run_zilany2014, ModelConfig, Species, AnfType, generate_cfs};

let signal: Vec<f64> = /* your audio samples */;
let cfs = generate_cfs(200.0, 8000.0, 30, Species::Human);

let config = ModelConfig {
    fs: 44100.0,
    species: Species::Human,
    cohc: 1.0,
    cihc: 1.0,
    anf_type: AnfType::Hsr,
    use_ffgn: true,
    seed: Some(42),
};

let output = run_zilany2014(&signal, &cfs, &config);

for channel in output.channels {
    println!("CF {}: {} spikes", channel.cf, channel.spike_times.len());
}
```

### Python

```python
import uino

# Generate characteristic frequencies (human cochlea)
cfs = uino.generate_cfs_py(200, 8000, 30, species="human")

# Run full model: sound → spikes
result = uino.run_zilany2014_full(
    signal,
    fs=44100,
    cfs=cfs,
    species="human",
    anf_type="hsr"
)

spike_times = result["spike_times"]  # List of spike time arrays per channel
```

## Credits

Cochlear model based on:
- [cochlea](https://github.com/mrkrd/cochlea) Python library by **Marek Rudnicki**
- Zilany, M.S.A., Bruce, I.C., & Carney, L.H. (2014). "Updated parameters and expanded simulation options for a model of the auditory periphery." *J. Acoust. Soc. Am.* 135(1), 283-286.

## License

GPL-3.0
