//! Pair-based Hebbian plasticity with Ca²⁺-proxy traces and L1 weight decay.
//!
//! Pre- and post-synaptic activity each leave an exponentially decaying trace
//! interpreted as a calcium proxy. Updates occur at spike events:
//!
//!   on pre-spike:  Δw = -η_ltd * x_post
//!   on post-spike: Δw = +η_ltp * x_pre
//!
//! Continuous L1 decay applied each step: w ← w − λ * sign(w) * dt.
//! Weight clipped to [w_min, w_max].
//!
//! This is the pair-based STDP form of the NMDA-Ca²⁺-dependent learning rule
//! used in Frontiers SNN-PC 2024; pre-before-post produces LTP and
//! post-before-pre produces LTD via the asymmetric trace ordering.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HebbianParams {
    /// Presynaptic trace time constant (ms).
    pub tau_pre: f64,
    /// Postsynaptic trace time constant (ms).
    pub tau_post: f64,
    /// LTP learning rate.
    pub eta_ltp: f64,
    /// LTD learning rate.
    pub eta_ltd: f64,
    /// L1 weight-decay rate per ms.
    pub lambda_l1: f64,
    /// Min weight (clipped).
    pub w_min: f64,
    /// Max weight (clipped).
    pub w_max: f64,
}

impl Default for HebbianParams {
    fn default() -> Self {
        Self {
            tau_pre: 20.0,
            tau_post: 20.0,
            eta_ltp: 0.005,
            eta_ltd: 0.005,
            lambda_l1: 1e-5,
            w_min: 0.0,
            w_max: 5.0,
        }
    }
}

/// A single plastic connection: weight + spike traces + parameters.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HebbianCa {
    pub params: HebbianParams,
    pub w: f64,
    /// Presynaptic activity trace ("Ca²⁺ proxy from pre side").
    pub x_pre: f64,
    /// Postsynaptic activity trace ("Ca²⁺ proxy from post side").
    pub x_post: f64,
}

impl HebbianCa {
    pub fn new(params: HebbianParams, initial_w: f64) -> Self {
        let w = initial_w.clamp(params.w_min, params.w_max);
        Self { params, w, x_pre: 0.0, x_post: 0.0 }
    }

    pub fn with_defaults(initial_w: f64) -> Self {
        Self::new(HebbianParams::default(), initial_w)
    }

    pub fn reset(&mut self) {
        self.x_pre = 0.0;
        self.x_post = 0.0;
    }

    /// Decay traces and apply L1 weight decay over `dt` ms.
    pub fn step(&mut self, dt_ms: f64) {
        self.x_pre *= (-dt_ms / self.params.tau_pre).exp();
        self.x_post *= (-dt_ms / self.params.tau_post).exp();

        // L1 decay: pull weight toward zero by λ*sign(w) per ms.
        let decay = self.params.lambda_l1 * dt_ms * self.w.signum();
        self.w -= decay;
        self.w = self.w.clamp(self.params.w_min, self.params.w_max);
    }

    /// Register a presynaptic spike. Updates pre-trace and depresses the weight
    /// proportional to recent postsynaptic activity (post-before-pre = LTD).
    pub fn pre_spike(&mut self) {
        self.x_pre += 1.0;
        self.w -= self.params.eta_ltd * self.x_post;
        self.w = self.w.clamp(self.params.w_min, self.params.w_max);
    }

    /// Register a postsynaptic spike. Updates post-trace and potentiates the
    /// weight proportional to recent presynaptic activity (pre-before-post = LTP).
    pub fn post_spike(&mut self) {
        self.x_post += 1.0;
        self.w += self.params.eta_ltp * self.x_pre;
        self.w = self.w.clamp(self.params.w_min, self.params.w_max);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drive paired pre→post pulses at `isi_ms` apart, repeated `n` times.
    /// Returns the final weight.
    fn paired_pulses(initial_w: f64, isi_ms: f64, n: usize, period_ms: f64) -> f64 {
        let mut h = HebbianCa::with_defaults(initial_w);
        let dt = 0.5;
        for _ in 0..n {
            h.pre_spike();
            advance(&mut h, isi_ms, dt);
            h.post_spike();
            advance(&mut h, period_ms - isi_ms, dt);
        }
        h.w
    }

    fn advance(h: &mut HebbianCa, total_ms: f64, dt: f64) {
        let n_steps = (total_ms / dt).round() as usize;
        for _ in 0..n_steps {
            h.step(dt);
        }
    }

    #[test]
    fn pre_before_post_potentiates() {
        let final_w = paired_pulses(0.5, 5.0, 50, 100.0);
        assert!(final_w > 0.5, "pre→post should potentiate, got {}", final_w);
    }

    #[test]
    fn post_before_pre_depresses() {
        // Reverse the order: post first, then pre.
        let mut h = HebbianCa::with_defaults(2.0);
        let dt = 0.5;
        for _ in 0..50 {
            h.post_spike();
            advance(&mut h, 5.0, dt);
            h.pre_spike();
            advance(&mut h, 95.0, dt);
        }
        assert!(h.w < 2.0, "post→pre should depress, got {}", h.w);
    }

    #[test]
    fn isolated_pre_decays_to_zero_via_l1() {
        let mut h = HebbianCa::with_defaults(1.0);
        // Many pre-spikes with no post — only L1 decay acts on the weight.
        for _ in 0..200 {
            h.pre_spike();
            advance(&mut h, 100.0, 0.5);
        }
        assert!(h.w < 1.0, "L1 should pull weight down without post-spikes, got {}", h.w);
    }

    #[test]
    fn weight_clipped_at_max() {
        let mut h = HebbianCa::with_defaults(4.99);
        // Saturate via many tight pre→post pairs.
        for _ in 0..200 {
            h.pre_spike();
            advance(&mut h, 2.0, 0.5);
            h.post_spike();
            advance(&mut h, 10.0, 0.5);
        }
        assert!(h.w <= h.params.w_max + 1e-12, "weight {} exceeded w_max {}", h.w, h.params.w_max);
    }

    #[test]
    fn weight_clipped_at_min() {
        let params = HebbianParams { w_min: 0.0, ..Default::default() };
        let mut h = HebbianCa::new(params, 0.1);
        for _ in 0..200 {
            h.post_spike();
            advance(&mut h, 2.0, 0.5);
            h.pre_spike();
            advance(&mut h, 10.0, 0.5);
        }
        assert!(h.w >= -1e-12, "weight {} fell below w_min", h.w);
    }

    #[test]
    fn traces_decay_with_tau() {
        let mut h = HebbianCa::with_defaults(1.0);
        h.pre_spike();
        let initial = h.x_pre;
        let tau = h.params.tau_pre;
        advance(&mut h, tau, 0.1);
        let after = h.x_pre;
        // After one tau, trace should be roughly 1/e of initial (slightly less due to discrete steps).
        let expected = initial / std::f64::consts::E;
        let ratio = after / expected;
        assert!((0.95..=1.05).contains(&ratio), "trace decay off: got {}, expected ≈ {}", after, expected);
    }
}
