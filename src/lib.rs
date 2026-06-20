//! Uino (初乃): Biologically-inspired cortical processing.
//!
//! Predictive coding cortex driven by spike streams from peripheral
//! sensory crates: [`cochlea`](https://crates.io/crates/cochlea-rs) for
//! auditory input and [`retinula`](https://crates.io/crates/retinula) for
//! visual input.

pub mod cortex;

pub use cortex::{
    A1Core, Belt, CorticalConfig, CorticalModel, CorticalModelBuilder, CorticalOutput,
    RewardModule, STRF, ThalamicRelay, ThetaGammaOscillator,
};
