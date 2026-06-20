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
    /// PC learning rate (per spike-count difference).
    pub eta: f64,
    /// SFA time constant (ms). The slow trace `s_slow` exponentially tracks
    /// input with this time constant; signals faster than ~1/tau_sfa cycle
    /// are filtered out. Predicts only the slow component.
    pub tau_sfa_ms: f64,
    /// SFA pull strength per ms: fraction of (s_slow − μ̂) added to μ̂ each
    /// step. Anchors the prediction to the slow trace.
    pub sfa_pull: f64,
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
        //
        // tau_sfa_ms ≈ 200 ms cleanly separates 50 Hz noise (filtered) from
        // 1 Hz signal (tracked). sfa_pull = 0.005 / ms gives ~τ = 200 ms
        // effective μ̂ relaxation toward s_slow.
        Self {
            n_per_population: 16,
            beta: 2000.0,
            gamma: 2000.0,
            eta: 0.0005,
            tau_sfa_ms: 80.0,
            sfa_pull: 0.005,
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
    /// Slow trace of the bottom-up input (SFA component). Tracks the slow
    /// envelope of `s` with time constant `tau_sfa_ms`.
    pub s_slow: f64,
    /// Whether prediction updates from PE activity (Principle B coupling).
    pub coupling: bool,
    /// Whether the slow trace anchors μ̂ (SFA). With this off but `coupling`
    /// on, only PC drives μ̂ — the falsification control for SFA.
    pub sfa_enabled: bool,
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
        Self {
            params,
            pe_plus,
            pe_minus,
            mu_hat: 0.0,
            s_slow: 0.0,
            coupling: true,
            // SFA disabled by default. Multi-stage learning uses PC, which
            // needs sustained PE signal to propagate up; SFA silences PE
            // too quickly for that. Enable per-cell when SFA-driven slow
            // tracking is what you want (Phase 1b).
            sfa_enabled: false,
        }
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
        self.s_slow = 0.0;
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

    /// Disable SFA (the slow-trace anchor on μ̂). PC still runs.
    /// Falsification control for SFA-dependent behavior.
    pub fn disable_sfa(&mut self) {
        self.sfa_enabled = false;
    }

    pub fn enable_sfa(&mut self) {
        self.sfa_enabled = true;
    }

    /// Advance one timestep with bottom-up input `s`.
    pub fn step(&mut self, s: f64, dt_ms: f64) -> StepOutput {
        self.step_with_top_down(s, 0.0, dt_ms)
    }

    /// Advance one timestep with both bottom-up input `s` and an externally
    /// supplied top-down prediction `top_down`. The effective prediction is
    /// the sum of this unit's internal `mu_hat` and the external signal —
    /// allowing a higher-level controller (e.g. a multi-unit predictive
    /// hierarchy) to inject prediction biases without overwriting state.
    pub fn step_with_top_down(&mut self, s: f64, top_down: f64, dt_ms: f64) -> StepOutput {
        let effective_mu = self.mu_hat + top_down;
        let drive_plus = self.params.beta * s - self.params.gamma * effective_mu;
        let drive_minus = self.params.gamma * effective_mu - self.params.beta * s;

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

        // SFA: update slow trace of input.
        let alpha_sfa = dt_ms / (self.params.tau_sfa_ms + dt_ms);
        self.s_slow = (1.0 - alpha_sfa) * self.s_slow + alpha_sfa * s;

        if self.coupling {
            if self.sfa_enabled {
                // SFA mode: μ̂ tracks the slow trace of the input. Required
                // for representing time-varying slow signals (Phase 1b
                // mechanism). PE update is suppressed in this mode because
                // it pulls μ̂ toward the stationary mean and fights the
                // time-varying slow target.
                self.mu_hat = self.s_slow;
            } else {
                // PC mode (default): μ̂ moves with PE imbalance. Converges
                // to the mean of `s`. Used by multi-stage learning where
                // sustained PE signal must propagate upward.
                let diff = pe_plus_spikes as f64 - pe_minus_spikes as f64;
                self.mu_hat += self.params.eta * diff;
            }
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

    fn correlation(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());
        let n = a.len() as f64;
        let ma = a.iter().sum::<f64>() / n;
        let mb = b.iter().sum::<f64>() / n;
        let mut num = 0.0;
        let mut da = 0.0;
        let mut db = 0.0;
        for i in 0..a.len() {
            let xa = a[i] - ma;
            let xb = b[i] - mb;
            num += xa * xb;
            da += xa * xa;
            db += xb * xb;
        }
        if da < 1e-12 || db < 1e-12 {
            0.0
        } else {
            num / (da.sqrt() * db.sqrt())
        }
    }

    /// SFA function test (Phase 1b). With SFA enabled, the cell's
    /// prediction μ̂ should track the slow component of a mixed slow+fast
    /// input, not the fast component. Verifies the slow-trace mechanism
    /// the Phase 2b invariance claim depends on.
    #[test]
    fn mu_hat_tracks_slow_component_ignoring_fast() {
        use std::f64::consts::PI;

        let mut mc = Microcircuit::with_defaults();
        mc.enable_sfa();
        let dt_ms = 0.1;
        let n_steps = 60_000; // 6 seconds
        let slow_period_ms = 1000.0; // 1 Hz
        let fast_period_ms = 20.0; // 50 Hz

        let mut mu_history = Vec::with_capacity(n_steps);
        let mut slow_history = Vec::with_capacity(n_steps);
        let mut fast_history = Vec::with_capacity(n_steps);

        for k in 0..n_steps {
            let t_ms = k as f64 * dt_ms;
            let slow = 2.0 + 0.8 * (2.0 * PI * t_ms / slow_period_ms).sin();
            let fast = 0.5 * (2.0 * PI * t_ms / fast_period_ms).sin();
            mc.step(slow + fast, dt_ms);
            mu_history.push(mc.mu_hat);
            slow_history.push(slow);
            fast_history.push(fast);
        }

        // Discard transient (first 1/3) and measure correlations on the rest.
        let start = n_steps / 3;
        let mu_late = &mu_history[start..];
        let slow_late = &slow_history[start..];
        let fast_late = &fast_history[start..];

        let corr_slow = correlation(mu_late, slow_late);
        let corr_fast = correlation(mu_late, fast_late);

        assert!(
            corr_slow > 0.5,
            "μ̂ should correlate with slow component: corr={}",
            corr_slow
        );
        assert!(
            corr_slow.abs() > 2.0 * corr_fast.abs(),
            "μ̂ should track slow much more than fast: |slow|={}, |fast|={}",
            corr_slow.abs(),
            corr_fast.abs(),
        );
    }

    /// Falsification control for SFA: with the slow-trace mechanism disabled,
    /// PC alone cannot track time-varying slow signals — it converges to the
    /// mean and stays there. Correlation with slow component should be far
    /// below the SFA-on value.
    #[test]
    fn pc_alone_cannot_track_slow_oscillation() {
        use std::f64::consts::PI;

        let mut mc = Microcircuit::with_defaults();
        // SFA is off by default; explicit for clarity.
        mc.disable_sfa();
        let dt_ms = 0.1;
        let n_steps = 60_000;
        let slow_period_ms = 1000.0;
        let fast_period_ms = 20.0;

        let mut mu_history = Vec::with_capacity(n_steps);
        let mut slow_history = Vec::with_capacity(n_steps);
        for k in 0..n_steps {
            let t_ms = k as f64 * dt_ms;
            let slow = 2.0 + 0.8 * (2.0 * PI * t_ms / slow_period_ms).sin();
            let fast = 0.5 * (2.0 * PI * t_ms / fast_period_ms).sin();
            mc.step(slow + fast, dt_ms);
            mu_history.push(mc.mu_hat);
            slow_history.push(slow);
        }
        let start = n_steps / 3;
        let corr = correlation(&mu_history[start..], &slow_history[start..]);
        assert!(
            corr.abs() < 0.4,
            "without SFA, μ̂ should NOT track slow oscillation: corr={}",
            corr,
        );
    }
}
