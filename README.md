# Uino (初乃)

Biologically-inspired sensory processing in Rust. Early stage — being redesigned around predictive coding.

## Current modules

- **retina** — photoreceptor → ganglion cells (ON/OFF, midget/parasol), retinotopy preserved
- **cortex** — thalamus, A1, belt (being rewritten)
- **ribbon_synapse** — vesicle dynamics shared by cochlea and retina
- **spike_generator** — inhomogeneous Poisson with refractory period

Cochlear model (Zilany 2014) lives in [cochlea-rs](https://github.com/knoguchi/cochlea-rs).

## Build

```bash
cargo build
cargo test
```

## License

GPL-3.0
