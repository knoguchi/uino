//! Measurement infrastructure for the compass metric (prediction error
//! spikes per inference) and manifold separability.

pub mod manifold;
pub mod spike_counter;

pub use manifold::{class_stats, separability, ClassStats, Separability};
pub use spike_counter::{Snapshot, SpikeCounter};
