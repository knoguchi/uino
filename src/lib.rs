//! Uino (初乃): Biologically-inspired cortical processing.
//!
//! Predictive coding cortex driven by spike streams from peripheral
//! sensory crates: [`cochlea`](https://crates.io/crates/cochlea-rs) for
//! auditory input and [`retinula`](https://crates.io/crates/retinula) for
//! visual input.

pub mod bridge;
pub mod cortex;
pub mod metrics;
pub mod microcircuit;

pub use bridge::{Channel, RetinaBridge};

pub use cortex::{
    A1Core, Belt, CorticalConfig, CorticalModel, CorticalModelBuilder, CorticalOutput,
    RewardModule, STRF, ThalamicRelay, ThetaGammaOscillator,
};
pub use microcircuit::{AdEx, AdExParams, AmpaSynapse, HebbianCa, HebbianParams, NmdaSynapse, SynapseParams};
