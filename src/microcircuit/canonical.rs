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
    /// Prediction integration rate per PE spike-count difference. The
    /// cell's prediction `μ̂` follows its input with an effective low-pass
    /// time constant set by this rate and the PE firing-rate gain. Larger
    /// `eta` → faster tracking, picks up higher input frequencies. Smaller
    /// `eta` → slower tracking, only follows slow input components.
    ///
    /// This is the per-cell timescale that produces multi-timescale
    /// behavior across a stack: lower stages use larger `eta` to follow
    /// fast input; higher stages use smaller `eta` to extract slow
    /// features. No mode switch — the same PC mechanism at different speeds.
    pub eta: f64,
    /// AdEx parameters for PE neurons.
    pub neuron: AdExParams,
}

impl Default for MicrocircuitParams {
    fn default() -> Self {
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
        Self {
            params,
            pe_plus,
            pe_minus,
            mu_hat: 0.0,
            coupling: true,
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

    fn run_mixed_input(mc: &mut Microcircuit, n_steps: usize, dt_ms: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        use std::f64::consts::PI;
        const SLOW_PERIOD_MS: f64 = 1000.0; // 1 Hz
        const FAST_PERIOD_MS: f64 = 20.0; // 50 Hz
        let mut mu = Vec::with_capacity(n_steps);
        let mut slow = Vec::with_capacity(n_steps);
        let mut fast = Vec::with_capacity(n_steps);
        for k in 0..n_steps {
            let t_ms = k as f64 * dt_ms;
            let s_slow = 2.0 + 0.8 * (2.0 * PI * t_ms / SLOW_PERIOD_MS).sin();
            let s_fast = 0.5 * (2.0 * PI * t_ms / FAST_PERIOD_MS).sin();
            mc.step(s_slow + s_fast, dt_ms);
            mu.push(mc.mu_hat);
            slow.push(s_slow);
            fast.push(s_fast);
        }
        (mu, slow, fast)
    }

    /// Phase 1b: multi-timescale predictive coding. The cell's effective
    /// low-pass cutoff is set by `eta` × PE-gain. A cell with the right
    /// `eta` selectively tracks the slow component of a mixed slow+fast
    /// input — picks up the 1 Hz signal and filters out the 50 Hz noise.
    ///
    /// No mode switch: this is the same PC mechanism, just tuned. Higher
    /// cortical stages would use smaller `eta` (longer effective tau) to
    /// extract slower features.
    #[test]
    fn cell_with_slow_eta_tracks_slow_component_ignoring_fast() {
        let params = MicrocircuitParams { eta: 0.03, ..Default::default() };
        let mut mc = Microcircuit::new(params);
        let (mu, slow, fast) = run_mixed_input(&mut mc, 60_000, 0.1);
        let start = mu.len() / 3;

        let corr_slow = correlation(&mu[start..], &slow[start..]);
        let corr_fast = correlation(&mu[start..], &fast[start..]);

        assert!(
            corr_slow > 0.5,
            "μ̂ should track slow component: corr={}",
            corr_slow
        );
        assert!(
            corr_slow.abs() > 2.0 * corr_fast.abs(),
            "μ̂ should track slow much more than fast: |slow|={}, |fast|={}",
            corr_slow.abs(),
            corr_fast.abs(),
        );
    }

    /// Falsification: a cell with `eta` too small has an effective cutoff
    /// well below 1 Hz, so it can't track the slow oscillation either —
    /// μ̂ stays near the stationary mean and barely correlates with the
    /// slow signal. This confirms that selective tracking is a property
    /// of the integration timescale, not magic.
    #[test]
    fn cell_with_tiny_eta_fails_to_track_slow() {
        let params = MicrocircuitParams { eta: 0.00005, ..Default::default() };
        let mut mc = Microcircuit::new(params);
        let (mu, slow, _fast) = run_mixed_input(&mut mc, 60_000, 0.1);
        let start = mu.len() / 3;
        let corr = correlation(&mu[start..], &slow[start..]);
        assert!(
            corr.abs() < 0.4,
            "tiny-eta cell should fail to track slow: corr={}",
            corr,
        );
    }
}
