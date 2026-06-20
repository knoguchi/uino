//! Canonical predictive-coding microcircuit (v0).
//!
//! One unit with scalar bottom-up input `s` and scalar top-down prediction `μ̂`.
//! Two AdEx populations encode signed prediction error as non-negative rates
//! (eLife 95127, Frontiers SNN-PC 2024):
//!
//!   PE+ population: excited by `s`, inhibited by `μ̂` → fires when s > μ̂
//!   PE− population: excited by `μ̂`, inhibited by `s` → fires when μ̂ > s
//!
//! Each step, `μ̂` is updated by η * (PE+ rate − PE− rate), driving μ̂ toward s.
//! With predictive coupling enabled, the prediction tracks the input and the
//! PE populations grow quiet — the compass metric "PE spikes per inference"
//! falls.

use crate::microcircuit::adex::{AdEx, AdExParams};

/// Parameters for one canonical microcircuit unit.
#[derive(Clone, Debug)]
pub struct MicrocircuitParams {
    /// Number of neurons in each of PE+ and PE−.
    pub n_per_population: usize,
    /// Bottom-up gain: pA of drive per unit of `s`.
    pub beta: f64,
    /// Top-down gain: pA of drive per unit of `μ̂`.
    pub gamma: f64,
    /// Prediction learning rate (per spike-count difference).
    pub eta: f64,
    /// AdEx parameters for PE neurons.
    pub neuron: AdExParams,
}

impl Default for MicrocircuitParams {
    fn default() -> Self {
        // PE neurons use the fast-spiking profile so they keep firing under
        // sustained mismatch until prediction catches up. Regular-spiking
        // adaptation would silence them before inference converges.
        //
        // β/γ chosen so the rheobase deadzone (~600 pA) is small relative to
        // typical unit-scale inputs: at β=2000, an `s − μ̂` of ~0.3 already
        // crosses threshold. Inference converges to μ̂ ≈ s within ~0.3.
        Self {
            n_per_population: 16,
            beta: 2000.0,
            gamma: 2000.0,
            eta: 0.0005,
            neuron: AdExParams::fast_spiking(),
        }
    }
}

/// One canonical predictive-coding microcircuit.
#[derive(Clone, Debug)]
pub struct Microcircuit {
    pub params: MicrocircuitParams,
    pub pe_plus: Vec<AdEx>,
    pub pe_minus: Vec<AdEx>,
    /// Current prediction estimate.
    pub mu_hat: f64,
    /// Whether prediction updates from PE activity (Principle B coupling).
    pub coupling: bool,
}

#[derive(Clone, Debug, Default)]
pub struct StepOutput {
    pub pe_plus_spikes: usize,
    pub pe_minus_spikes: usize,
}

impl Microcircuit {
    pub fn new(params: MicrocircuitParams) -> Self {
        let pe_plus = (0..params.n_per_population).map(|_| AdEx::new(params.neuron.clone())).collect();
        let pe_minus = (0..params.n_per_population).map(|_| AdEx::new(params.neuron.clone())).collect();
        Self { params, pe_plus, pe_minus, mu_hat: 0.0, coupling: true }
    }

    pub fn with_defaults() -> Self {
        Self::new(MicrocircuitParams::default())
    }

    pub fn reset(&mut self) {
        for n in &mut self.pe_plus {
            n.reset();
        }
        for n in &mut self.pe_minus {
            n.reset();
        }
        self.mu_hat = 0.0;
    }

    /// Disable the prediction → input coupling. Used as the falsification
    /// control: with coupling off, μ̂ stays constant and PE populations
    /// purely report instantaneous bottom-up activity.
    pub fn disable_coupling(&mut self) {
        self.coupling = false;
    }

    pub fn enable_coupling(&mut self) {
        self.coupling = true;
    }

    /// Advance one timestep with bottom-up input `s`.
    pub fn step(&mut self, s: f64, dt_ms: f64) -> StepOutput {
        // Net drives per population (positive depolarizing, matching AdEx convention).
        let drive_plus = self.params.beta * s - self.params.gamma * self.mu_hat;
        let drive_minus = self.params.gamma * self.mu_hat - self.params.beta * s;

        let mut pe_plus_spikes = 0usize;
        let mut pe_minus_spikes = 0usize;

        for n in &mut self.pe_plus {
            if n.step(drive_plus, dt_ms) {
                pe_plus_spikes += 1;
            }
        }
        for n in &mut self.pe_minus {
            if n.step(drive_minus, dt_ms) {
                pe_minus_spikes += 1;
            }
        }

        if self.coupling {
            let diff = pe_plus_spikes as f64 - pe_minus_spikes as f64;
            self.mu_hat += self.params.eta * diff;
        }

        StepOutput { pe_plus_spikes, pe_minus_spikes }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::SpikeCounter;

    fn run(
        mc: &mut Microcircuit,
        input_at: impl Fn(usize) -> f64,
        n_steps: usize,
        dt_ms: f64,
    ) -> (usize, usize) {
        let mut tot_p = 0;
        let mut tot_m = 0;
        for k in 0..n_steps {
            let s = input_at(k);
            let out = mc.step(s, dt_ms);
            tot_p += out.pe_plus_spikes;
            tot_m += out.pe_minus_spikes;
        }
        (tot_p, tot_m)
    }

    #[test]
    fn silent_when_input_matches_prediction() {
        // Manually set μ̂ to match s, disable coupling so it stays. No PE activity expected.
        let mut mc = Microcircuit::with_defaults();
        mc.mu_hat = 2.0;
        mc.disable_coupling();
        let (p, m) = run(&mut mc, |_| 2.0, 1000, 0.1);
        assert_eq!(p, 0, "PE+ should be silent when s == μ̂, got {}", p);
        assert_eq!(m, 0, "PE- should be silent when s == μ̂, got {}", m);
    }

    #[test]
    fn pe_plus_fires_when_s_exceeds_prediction() {
        let mut mc = Microcircuit::with_defaults();
        mc.disable_coupling(); // hold μ̂ at 0
        let (p, m) = run(&mut mc, |_| 3.0, 1000, 0.1);
        assert!(p > 0, "PE+ should fire with s > μ̂");
        assert_eq!(m, 0, "PE- should be silent with s > μ̂, got {}", m);
    }

    #[test]
    fn pe_minus_fires_when_prediction_exceeds_input() {
        let mut mc = Microcircuit::with_defaults();
        mc.mu_hat = 3.0;
        mc.disable_coupling();
        let (p, m) = run(&mut mc, |_| 0.0, 1000, 0.1);
        assert_eq!(p, 0, "PE+ should be silent with μ̂ > s, got {}", p);
        assert!(m > 0, "PE- should fire with μ̂ > s");
    }

    #[test]
    fn prediction_converges_to_input_with_coupling() {
        // μ̂ converges to s minus the rheobase deadzone (≈0.3 with default β=2000).
        let mut mc = Microcircuit::with_defaults();
        run(&mut mc, |_| 2.0, 30_000, 0.1);
        assert!(
            (mc.mu_hat - 2.0).abs() < 0.5,
            "μ̂ should converge within 0.5 of s=2, got {}",
            mc.mu_hat
        );
    }

    #[test]
    fn pe_spike_count_decreases_with_learning() {
        // The compass metric. Run the same constant input through two phases
        // (early vs late) and check PE spike count drops as prediction learns.
        let mut mc = Microcircuit::with_defaults();
        let mut early = SpikeCounter::new();
        let mut late = SpikeCounter::new();

        for _ in 0..2000 {
            let out = mc.step(2.0, 0.1);
            early.record_n("pe_plus", out.pe_plus_spikes);
            early.record_n("pe_minus", out.pe_minus_spikes);
        }
        let early_snap = early.snapshot();

        // Skip some steady-state steps, then measure again.
        for _ in 0..6000 {
            mc.step(2.0, 0.1);
        }
        for _ in 0..2000 {
            let out = mc.step(2.0, 0.1);
            late.record_n("pe_plus", out.pe_plus_spikes);
            late.record_n("pe_minus", out.pe_minus_spikes);
        }
        let late_snap = late.snapshot();

        let early_total = SpikeCounter::pe_total(&early_snap);
        let late_total = SpikeCounter::pe_total(&late_snap);
        assert!(
            late_total < early_total,
            "compass metric must fall with learning: early={}, late={}",
            early_total,
            late_total,
        );
    }

    #[test]
    fn coupling_off_means_no_prediction_change() {
        let mut mc = Microcircuit::with_defaults();
        mc.disable_coupling();
        let mu_before = mc.mu_hat;
        run(&mut mc, |_| 5.0, 5000, 0.1);
        assert_eq!(mc.mu_hat, mu_before, "μ̂ must stay frozen with coupling off");
    }

    #[test]
    fn step_change_provokes_pe_minus_after_overshooting() {
        // First train on s=2 so μ̂ rises to ≈1.5–1.8. Then drop s to 0 and verify PE- fires.
        let mut mc = Microcircuit::with_defaults();
        run(&mut mc, |_| 2.0, 30_000, 0.1);
        assert!(mc.mu_hat > 1.0, "prerequisite: μ̂ should have risen, got {}", mc.mu_hat);

        let mut m_after = 0;
        for _ in 0..500 {
            let out = mc.step(0.0, 0.1);
            m_after += out.pe_minus_spikes;
        }
        assert!(m_after > 0, "PE- should fire when input drops below learned prediction");
    }
}
