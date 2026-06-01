//! Auditory Cortex Model (A1 + Belt)
//!
//! Biologically-plausible cortical processing for speech recognition.
//!
//! # Architecture
//!
//! ```text
//!                     ┌─────────────────────────────────────┐
//!                     │         Belt (Parabelt)             │
//!                     │   Context integration (τ=200ms)     │
//!                     │   Prediction generation             │
//!                     └──────────────┬──────────────────────┘
//!                                    │ prediction
//!                                    ▼
//!                     ┌─────────────────────────────────────┐
//!                     │         A1 Core (τ=50ms)            │
//!                     │   Learned STRFs, sparse coding      │
//!                     │   Prediction error computation      │
//!                     └──────────────┬──────────────────────┘
//!                                    │ prediction
//!                                    ▼
//!  Cochlea ────────▶  ┌─────────────────────────────────────┐
//!                     │    Thalamus (MGN) - Gating          │
//!                     │   Onset (τ=10ms) + Sustained (τ=30ms)│
//!                     │   Predictive suppression            │
//!                     └─────────────────────────────────────┘
//! ```
//!
//! # Key Features
//!
//! - **Predictive coding**: Top-down predictions suppress expected input
//! - **Multi-timescale**: Parallel streams with τ = 10ms, 50ms, 200ms
//! - **Sparse overcomplete**: 10-50× expansion, ~5-10% sparsity
//! - **Reward-gated plasticity**: STDP modulated by global reward signal
//! - **Theta-gamma coupling**: Rate-invariant temporal normalization

pub mod thalamus;
pub mod strf;
pub mod a1_core;
pub mod oscillator;
pub mod belt;
pub mod reward;
pub mod model;

// Re-exports
pub use thalamus::{ThalamicRelay, ThalamicOutput};
pub use strf::STRF;
pub use a1_core::{A1Core, A1Output};
pub use oscillator::{ThetaGammaOscillator, OscillatorState};
pub use belt::{Belt, BeltOutput};
pub use reward::RewardModule;
pub use model::{CorticalModel, CorticalConfig, CorticalOutput, CorticalModelBuilder};
