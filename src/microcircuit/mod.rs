//! Spiking primitives for the predictive-coding canonical microcircuit.
//!
//! Building blocks per docs/research-cortex.md:
//! - [`AdEx`] neuron (Brette & Gerstner 2005)
//! - AMPA + NMDA conductance-based synapses
//! - Local Hebbian Ca²⁺-proxy plasticity with L1 weight decay

pub mod adex;
pub mod apparent_motion;
pub mod canonical;
pub mod plasticity;
pub mod synapse;

pub use adex::{AdEx, AdExParams};
pub use apparent_motion::{alternating_stimulus, ApparentMotion, TwoUnitOutput};
pub use canonical::{Microcircuit, MicrocircuitParams, StepOutput};
pub use plasticity::{HebbianCa, HebbianParams};
pub use synapse::{AmpaSynapse, NmdaSynapse, SynapseParams};

#[cfg(test)]
mod integration_tests {
    use super::*;

    /// Drive pre→ampa+nmda→post with plastic weight; confirm composition:
    /// (1) pre fires under suprathreshold current,
    /// (2) post fires via synaptic drive,
    /// (3) weight grows via pre-before-post coincidence.
    #[test]
    fn pre_drives_post_and_potentiates() {
        let mut pre = AdEx::with_defaults();
        let mut post = AdEx::with_defaults();
        let mut ampa = AmpaSynapse::new();
        let mut nmda = NmdaSynapse::new();
        // Single lumped synapse representing many contacts — weight chosen so
        // one pre spike is enough to push post past threshold via AMPA+NMDA.
        let params = HebbianParams { w_max: 200.0, ..Default::default() };
        let mut plast = HebbianCa::new(params, 80.0);

        let dt = 0.1;
        let n_steps = 5000; // 500 ms
        let drive_pre = 1500.0; // pA, strongly suprathreshold

        let initial_w = plast.w;
        let mut pre_count = 0;
        let mut post_count = 0;

        for _ in 0..n_steps {
            if pre.step(drive_pre, dt) {
                ampa.receive_spike(plast.w);
                nmda.receive_spike(plast.w);
                plast.pre_spike();
                pre_count += 1;
            }

            let i_syn = ampa.current(post.v) + nmda.current(post.v);
            if post.step(i_syn, dt) {
                plast.post_spike();
                post_count += 1;
            }

            ampa.step(dt);
            nmda.step(dt);
            plast.step(dt);
        }

        assert!(pre_count > 10, "pre should spike repeatedly, got {}", pre_count);
        assert!(post_count > 0, "post should spike from synaptic drive, got {}", post_count);
        assert!(
            plast.w > initial_w,
            "weight should grow via pre→post coincidence: initial={}, final={}",
            initial_w,
            plast.w,
        );
    }

    /// Without presynaptic drive, the postsynaptic neuron stays silent and
    /// the weight decays via L1. Composition with quiescent inputs is safe.
    #[test]
    fn silent_input_keeps_post_silent_and_decays_weight() {
        let mut pre = AdEx::with_defaults();
        let mut post = AdEx::with_defaults();
        let mut ampa = AmpaSynapse::new();
        let mut nmda = NmdaSynapse::new();
        let mut plast = HebbianCa::with_defaults(2.0);

        let dt = 0.1;
        let n_steps = 5000;

        let initial_w = plast.w;
        let mut post_count = 0;

        for _ in 0..n_steps {
            // No drive on pre.
            if pre.step(0.0, dt) {
                ampa.receive_spike(plast.w);
                nmda.receive_spike(plast.w);
                plast.pre_spike();
            }
            let i_syn = ampa.current(post.v) + nmda.current(post.v);
            if post.step(i_syn, dt) {
                plast.post_spike();
                post_count += 1;
            }
            ampa.step(dt);
            nmda.step(dt);
            plast.step(dt);
        }

        assert_eq!(post_count, 0, "post must be silent without input");
        assert!(plast.w < initial_w, "L1 should pull weight down, got {}", plast.w);
    }
}
