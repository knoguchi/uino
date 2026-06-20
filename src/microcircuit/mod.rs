//! Spiking primitives for the predictive-coding canonical microcircuit.
//!
//! Building blocks per docs/research-cortex.md:
//! - [`AdEx`] neuron (Brette & Gerstner 2005)
//! - AMPA + NMDA conductance-based synapses
//! - Local Hebbian Ca²⁺-proxy plasticity with L1 weight decay

pub mod adex;
pub mod plasticity;
pub mod synapse;

pub use adex::{AdEx, AdExParams};
pub use plasticity::{HebbianCa, HebbianParams};
pub use synapse::{AmpaSynapse, NmdaSynapse, SynapseParams};
