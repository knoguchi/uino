//! Conductance-based AMPA and NMDA synaptic currents.
//!
//! Each receptor type is modeled as an aggregate conductance on the
//! postsynaptic neuron. Presynaptic spikes inject conductance scaled by
//! weight; conductance decays exponentially. The synapse computes a
//! current given the postsynaptic voltage.
//!
//! NMDA has voltage-dependent Mg²⁺ block (Jahr & Stevens 1990):
//!
//!   block(V) = 1 / (1 + [Mg²⁺] * exp(-0.062 * V) / 3.57)
//!
//! Time constants: AMPA τ ≈ 5 ms, NMDA τ ≈ 50 ms (decay).

/// Shared parameter shape for both AMPA and NMDA channels.
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SynapseParams {
    /// Decay time constant (ms).
    pub tau: f64,
    /// Reversal potential (mV). 0 for excitatory glutamatergic receptors.
    pub e_rev: f64,
}

impl SynapseParams {
    pub fn ampa() -> Self {
        Self { tau: 5.0, e_rev: 0.0 }
    }

    pub fn nmda() -> Self {
        Self { tau: 50.0, e_rev: 0.0 }
    }
}

/// AMPA receptor channel (fast, linear).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AmpaSynapse {
    pub params: SynapseParams,
    /// Aggregate conductance (nS).
    pub g: f64,
}

impl AmpaSynapse {
    pub fn new() -> Self {
        Self { params: SynapseParams::ampa(), g: 0.0 }
    }

    pub fn with_params(params: SynapseParams) -> Self {
        Self { params, g: 0.0 }
    }

    /// Inject a presynaptic spike with the given weight (nS).
    #[inline]
    pub fn receive_spike(&mut self, weight: f64) {
        self.g += weight;
    }

    /// Decay conductance by `dt` ms.
    #[inline]
    pub fn step(&mut self, dt_ms: f64) {
        let decay = (-dt_ms / self.params.tau).exp();
        self.g *= decay;
    }

    /// Inward synaptic current given postsynaptic voltage (pA).
    ///
    /// Returns the current flowing INTO the cell, with the same sign convention
    /// as [`AdEx::step`]'s `i_input`: positive when depolarizing. Specifically,
    /// `I_in = g * (E_rev - V)` — positive when V < E_rev (typical for resting
    /// glutamatergic synapses).
    #[inline]
    pub fn current(&self, v_post: f64) -> f64 {
        self.g * (self.params.e_rev - v_post)
    }

    pub fn reset(&mut self) {
        self.g = 0.0;
    }
}

impl Default for AmpaSynapse {
    fn default() -> Self {
        Self::new()
    }
}

/// NMDA receptor channel (slow, voltage-dependent Mg²⁺ block).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NmdaSynapse {
    pub params: SynapseParams,
    /// Aggregate conductance (nS).
    pub g: f64,
    /// External Mg²⁺ concentration (mM). Physiological default 1.0.
    pub mg_conc: f64,
}

impl NmdaSynapse {
    pub fn new() -> Self {
        Self {
            params: SynapseParams::nmda(),
            g: 0.0,
            mg_conc: 1.0,
        }
    }

    pub fn with_params(params: SynapseParams) -> Self {
        Self { params, g: 0.0, mg_conc: 1.0 }
    }

    #[inline]
    pub fn receive_spike(&mut self, weight: f64) {
        self.g += weight;
    }

    #[inline]
    pub fn step(&mut self, dt_ms: f64) {
        let decay = (-dt_ms / self.params.tau).exp();
        self.g *= decay;
    }

    /// Mg²⁺ block factor: ~0 at very hyperpolarized V, →1 at depolarized V.
    #[inline]
    pub fn mg_block(&self, v_post: f64) -> f64 {
        1.0 / (1.0 + self.mg_conc * (-0.062 * v_post).exp() / 3.57)
    }

    /// Inward NMDA current including Mg²⁺ block (pA).
    /// Sign convention matches [`AdEx::step`]'s `i_input`: positive depolarizes.
    #[inline]
    pub fn current(&self, v_post: f64) -> f64 {
        self.g * (self.params.e_rev - v_post) * self.mg_block(v_post)
    }

    pub fn reset(&mut self) {
        self.g = 0.0;
    }
}

impl Default for NmdaSynapse {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn ampa_decays_with_tau() {
        let mut s = AmpaSynapse::new();
        s.receive_spike(1.0);
        let initial = s.g;
        // After one tau, conductance should be 1/e of initial.
        s.step(s.params.tau);
        assert_relative_eq!(s.g, initial / std::f64::consts::E, max_relative = 1e-9);
    }

    #[test]
    fn ampa_current_zero_at_reversal() {
        let mut s = AmpaSynapse::new();
        s.receive_spike(1.0);
        // At V = E_rev, no driving force.
        assert_relative_eq!(s.current(s.params.e_rev), 0.0);
    }

    #[test]
    fn ampa_current_linear_in_weight() {
        let mut a = AmpaSynapse::new();
        let mut b = AmpaSynapse::new();
        a.receive_spike(1.0);
        b.receive_spike(3.0);
        assert_relative_eq!(b.current(-70.0), 3.0 * a.current(-70.0));
    }

    #[test]
    fn nmda_decays_slower_than_ampa() {
        let mut a = AmpaSynapse::new();
        let mut n = NmdaSynapse::new();
        a.receive_spike(1.0);
        n.receive_spike(1.0);
        for _ in 0..50 {
            a.step(1.0);
            n.step(1.0);
        }
        assert!(n.g > a.g, "NMDA must outlive AMPA: nmda={}, ampa={}", n.g, a.g);
    }

    #[test]
    fn nmda_mg_block_blocks_at_rest() {
        let n = NmdaSynapse::new();
        let block_rest = n.mg_block(-70.0);
        let block_depol = n.mg_block(0.0);
        assert!(block_rest < 0.2, "NMDA should be largely blocked at rest, got {}", block_rest);
        assert!(block_depol > 0.5, "NMDA should be largely unblocked at 0 mV, got {}", block_depol);
    }

    #[test]
    fn nmda_current_grows_with_depolarization_then_falls() {
        // I_NMDA = g * (V - 0) * mg_block(V) — non-monotonic in V; rises then falls beyond reversal.
        let mut n = NmdaSynapse::new();
        n.receive_spike(1.0);
        let i_at = |v| n.current(v).abs();
        let at_rest = i_at(-70.0);
        let at_threshold = i_at(-40.0);
        assert!(
            at_threshold > at_rest,
            "NMDA current should grow with depolarization: rest={}, thresh={}",
            at_rest,
            at_threshold,
        );
    }

    #[test]
    fn reset_clears_conductance() {
        let mut a = AmpaSynapse::new();
        let mut n = NmdaSynapse::new();
        a.receive_spike(5.0);
        n.receive_spike(5.0);
        a.reset();
        n.reset();
        assert_eq!(a.g, 0.0);
        assert_eq!(n.g, 0.0);
    }
}
