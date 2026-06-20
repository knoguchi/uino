//! Two-stage stacked predictive-coding architecture.
//!
//! A lower stage of microcircuits receives external input. Each upper-stage
//! cell pools PE+ activity from a non-overlapping receptive-field square of
//! lower-stage cells. The upper cell's prediction feeds back down to every
//! lower cell in its receptive field as top-down input.
//!
//! Routing preserves retinotopy: lower cell (lx, ly) is governed by upper
//! cell (lx / pool, ly / pool). When the lower input has a single hot spot,
//! activity at the upper stage appears at the spatially-corresponding cell,
//! never at the spatial mirror.

use crate::microcircuit::canonical::{Microcircuit, MicrocircuitParams};

#[derive(Clone, Debug)]
pub struct StageGrid {
    pub cells_x: usize,
    pub cells_y: usize,
    pub units: Vec<Microcircuit>,
}

impl StageGrid {
    pub fn new(cells_x: usize, cells_y: usize, params: MicrocircuitParams) -> Self {
        let units = (0..cells_x * cells_y).map(|_| Microcircuit::new(params.clone())).collect();
        Self { cells_x, cells_y, units }
    }

    /// Apply a step to every cell with its own bottom-up input and per-cell
    /// top-down feedback. Returns per-cell PE+ and PE− counts.
    pub fn step(&mut self, input: &[f64], top_down: &[f64], dt_ms: f64) -> (Vec<usize>, Vec<usize>) {
        assert_eq!(input.len(), self.units.len(), "input shape mismatch");
        assert_eq!(top_down.len(), self.units.len(), "top_down shape mismatch");
        let mut pe_plus = vec![0usize; self.units.len()];
        let mut pe_minus = vec![0usize; self.units.len()];
        for (i, u) in self.units.iter_mut().enumerate() {
            let out = u.step_with_top_down(input[i], top_down[i], dt_ms);
            pe_plus[i] = out.pe_plus_spikes;
            pe_minus[i] = out.pe_minus_spikes;
        }
        (pe_plus, pe_minus)
    }

    pub fn predictions(&self) -> Vec<f64> {
        self.units.iter().map(|u| u.mu_hat).collect()
    }

    pub fn reset(&mut self) {
        for u in &mut self.units {
            u.reset();
        }
    }

    pub fn disable_coupling(&mut self) {
        for u in &mut self.units {
            u.disable_coupling();
        }
    }
}

/// Two stacked stages with non-overlapping square pooling between them.
///
/// `lower` has shape (cells_x_low, cells_y_low). `upper` has shape
/// (cells_x_low / pool, cells_y_low / pool). The lower dimensions must be
/// divisible by `pool`.
#[derive(Clone, Debug)]
pub struct TwoStage {
    pub lower: StageGrid,
    pub upper: StageGrid,
    pub pool: usize,
    /// Feedforward gain: lower PE+ count per step is divided by this to feed
    /// into the upper cell's scalar input. Tuned so a single hot lower cell
    /// firing at peak rate puts the upper cell into its working range.
    pub ff_gain: f64,
    /// Feedback gain: upper μ̂ is multiplied by this before being injected
    /// as top-down to each lower cell in its receptive field.
    pub fb_gain: f64,
}

#[derive(Clone, Debug, Default)]
pub struct TwoStageStepOutput {
    pub lower_pe_plus: Vec<usize>,
    pub lower_pe_minus: Vec<usize>,
    pub upper_pe_plus: Vec<usize>,
    pub upper_pe_minus: Vec<usize>,
}

impl TwoStageStepOutput {
    pub fn lower_pe_total(&self) -> usize {
        self.lower_pe_plus.iter().sum::<usize>() + self.lower_pe_minus.iter().sum::<usize>()
    }
    pub fn upper_pe_total(&self) -> usize {
        self.upper_pe_plus.iter().sum::<usize>() + self.upper_pe_minus.iter().sum::<usize>()
    }
    pub fn pe_total(&self) -> usize {
        self.lower_pe_total() + self.upper_pe_total()
    }
}

impl TwoStage {
    pub fn new(cells_x_low: usize, cells_y_low: usize, pool: usize, params: MicrocircuitParams) -> Self {
        assert!(pool > 0);
        assert_eq!(cells_x_low % pool, 0, "cells_x must be divisible by pool");
        assert_eq!(cells_y_low % pool, 0, "cells_y must be divisible by pool");
        let lower = StageGrid::new(cells_x_low, cells_y_low, params.clone());
        let upper = StageGrid::new(cells_x_low / pool, cells_y_low / pool, params);
        Self { lower, upper, pool, ff_gain: 0.3, fb_gain: 1.0 }
    }

    pub fn with_defaults(cells_x_low: usize, cells_y_low: usize, pool: usize) -> Self {
        Self::new(cells_x_low, cells_y_low, pool, MicrocircuitParams::default())
    }

    pub fn reset(&mut self) {
        self.lower.reset();
        self.upper.reset();
    }

    pub fn disable_coupling(&mut self) {
        self.lower.disable_coupling();
        self.upper.disable_coupling();
    }

    /// Upper-stage parent for a lower cell at index `i`.
    fn upper_idx_of(&self, lower_i: usize) -> usize {
        let lx = lower_i % self.lower.cells_x;
        let ly = lower_i / self.lower.cells_x;
        let ux = lx / self.pool;
        let uy = ly / self.pool;
        uy * self.upper.cells_x + ux
    }

    pub fn step(&mut self, input: &[f64], dt_ms: f64) -> TwoStageStepOutput {
        assert_eq!(input.len(), self.lower.cells_x * self.lower.cells_y);

        // 1. Feedback from current upper predictions to lower cells.
        let upper_mu = self.upper.predictions();
        let feedback_for_lower: Vec<f64> = (0..self.lower.units.len())
            .map(|i| self.fb_gain * upper_mu[self.upper_idx_of(i)])
            .collect();

        // 2. Step lower with input + feedback.
        let (lower_pe_plus, lower_pe_minus) =
            self.lower.step(input, &feedback_for_lower, dt_ms);

        // 3. Pool lower PE+ into upper's bottom-up input.
        let n_up = self.upper.units.len();
        let mut upper_input = vec![0.0f64; n_up];
        for (i, &pp) in lower_pe_plus.iter().enumerate() {
            upper_input[self.upper_idx_of(i)] += pp as f64;
        }
        for v in &mut upper_input {
            *v /= self.ff_gain;
        }

        // 4. Step upper with that input and no further feedback (top of stack).
        let upper_top_down = vec![0.0; n_up];
        let (upper_pe_plus, upper_pe_minus) =
            self.upper.step(&upper_input, &upper_top_down, dt_ms);

        TwoStageStepOutput {
            lower_pe_plus,
            lower_pe_minus,
            upper_pe_plus,
            upper_pe_minus,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::separability;

    fn one_hot(cells_x: usize, cells_y: usize, x: usize, y: usize, mag: f64) -> Vec<f64> {
        let mut v = vec![0.0; cells_x * cells_y];
        v[y * cells_x + x] = mag;
        v
    }

    /// Two-hot pattern: bright at the two diagonal corners of a 4×4.
    fn diagonal_pattern(mag: f64) -> Vec<f64> {
        let mut v = vec![0.0; 16];
        v[0] = mag;
        v[3 * 4 + 3] = mag;
        v
    }

    /// Two-hot pattern: bright at the two anti-diagonal corners of a 4×4.
    fn anti_diagonal_pattern(mag: f64) -> Vec<f64> {
        let mut v = vec![0.0; 16];
        v[3] = mag;
        v[3 * 4] = mag;
        v
    }

    /// Present `pattern` to `ts` for `n_steps`, return upper-stage signed
    /// activity per cell (PE+ − PE−) summed over the presentation window.
    /// This is the "class representation" used for separability scoring.
    fn collect_upper_signature(ts: &mut TwoStage, pattern: &[f64], n_steps: usize) -> Vec<f64> {
        let n_up = ts.upper.units.len();
        let mut sig = vec![0.0; n_up];
        for _ in 0..n_steps {
            let out = ts.step(pattern, 0.1);
            for i in 0..n_up {
                sig[i] += out.upper_pe_plus[i] as f64 - out.upper_pe_minus[i] as f64;
            }
        }
        sig
    }

    fn run_steady(ts: &mut TwoStage, input: &[f64], n_steps: usize) {
        for _ in 0..n_steps {
            ts.step(input, 0.1);
        }
    }

    /// A hot spot at one lower position must activate the upper stage at the
    /// spatially-corresponding cell, and ONLY there. Tests retinotopy
    /// preservation across the feedforward route.
    #[test]
    fn retinotopy_preserved_to_upper_stage() {
        let mut ts = TwoStage::with_defaults(4, 4, 2);
        // Upper grid is 2×2. Hot lower cell at (3, 1) should map to upper (1, 0).
        let input = one_hot(4, 4, 3, 1, 2.0);
        run_steady(&mut ts, &input, 5000);

        let upper_mu = ts.upper.predictions();
        let target_idx = 0 * 2 + 1; // upper (1, 0)
        let target_mu = upper_mu[target_idx];

        // The target upper cell should have learned a positive prediction.
        assert!(target_mu > 0.1, "target upper cell μ̂ should be positive, got {}", target_mu);

        // Every other upper cell should have stayed near zero (or negative due to dynamics).
        for (i, &mu) in upper_mu.iter().enumerate() {
            if i != target_idx {
                assert!(
                    mu.abs() < target_mu * 0.5,
                    "non-target upper cell {} μ̂ {} too close to target {}",
                    i,
                    mu,
                    target_mu,
                );
            }
        }
    }

    /// PE spike count at the lower stage must decrease as the upper stage
    /// learns to predict the input pattern. This is the compass-metric
    /// confirmation that the feedback route is doing real work.
    #[test]
    fn lower_pe_falls_as_upper_learns() {
        let mut ts = TwoStage::with_defaults(4, 4, 2);
        let input = one_hot(4, 4, 1, 1, 2.0);

        // Early window.
        let mut early_pe = 0usize;
        for _ in 0..2000 {
            let out = ts.step(&input, 0.1);
            early_pe += out.lower_pe_total();
        }

        // Train through.
        run_steady(&mut ts, &input, 20_000);

        // Late window.
        let mut late_pe = 0usize;
        for _ in 0..2000 {
            let out = ts.step(&input, 0.1);
            late_pe += out.lower_pe_total();
        }

        assert!(
            late_pe < early_pe,
            "lower PE must fall as upper predicts: early={}, late={}",
            early_pe,
            late_pe,
        );
    }

    /// With feedback disabled (no coupling anywhere), the upper stage cannot
    /// drive the lower stage's prediction down. Lower PE stays roughly flat
    /// between early and late windows — the falsification control.
    #[test]
    fn coupling_off_no_lower_pe_reduction() {
        let mut ts = TwoStage::with_defaults(4, 4, 2);
        ts.disable_coupling();
        let input = one_hot(4, 4, 1, 1, 2.0);

        let mut early_pe = 0usize;
        for _ in 0..2000 {
            let out = ts.step(&input, 0.1);
            early_pe += out.lower_pe_total();
        }
        run_steady(&mut ts, &input, 20_000);
        let mut late_pe = 0usize;
        for _ in 0..2000 {
            let out = ts.step(&input, 0.1);
            late_pe += out.lower_pe_total();
        }

        // Some shrinkage from intrinsic neuron dynamics is acceptable; demand
        // that the difference is much smaller than the coupling-on case.
        // Specifically: late must be at least 80% of early (no learning).
        let ratio = late_pe as f64 / early_pe.max(1) as f64;
        assert!(ratio > 0.8, "coupling off should keep PE flat: ratio={}", ratio);
    }

    /// Off-target upper cells must not develop large predictions when input
    /// is concentrated elsewhere — no spatial cross-talk through the routing.
    #[test]
    fn no_spatial_crosstalk() {
        let mut ts = TwoStage::with_defaults(4, 4, 2);
        // Hot spot in the top-left quadrant; only upper (0,0) should grow.
        let input = one_hot(4, 4, 0, 0, 2.0);
        run_steady(&mut ts, &input, 10_000);

        let upper_mu = ts.upper.predictions();
        let target = upper_mu[0];
        for i in 1..upper_mu.len() {
            assert!(
                upper_mu[i].abs() < target.abs().max(0.1) * 0.5,
                "non-target upper cell {} ({}) too close to target {}",
                i,
                upper_mu[i],
                target,
            );
        }
    }

    /// Train on alternating exposures of two distinct spatial patterns
    /// (diagonal vs anti-diagonal hot spots). After training, present
    /// multiple held-out samples of each class and verify the upper-stage
    /// activity signatures cluster separately — manifold separability index
    /// exceeds 1 by a clear margin.
    #[test]
    fn upper_stage_discriminates_two_classes() {
        let mut ts = TwoStage::with_defaults(4, 4, 2);

        // Train: interleave the two patterns with mild magnitude jitter
        // so the system sees varied examples per class.
        let presentation_steps = 200;
        let n_cycles = 50;
        for cycle in 0..n_cycles {
            let mag_a = 1.8 + 0.4 * ((cycle * 7) % 5) as f64 / 5.0;
            let mag_b = 1.8 + 0.4 * ((cycle * 11) % 5) as f64 / 5.0;
            for _ in 0..presentation_steps {
                ts.step(&diagonal_pattern(mag_a), 0.1);
            }
            for _ in 0..presentation_steps {
                ts.step(&anti_diagonal_pattern(mag_b), 0.1);
            }
        }

        // Collect class signatures from held-out magnitudes.
        let test_mags = [1.85, 1.95, 2.05, 2.15];
        let class_a: Vec<Vec<f64>> = test_mags
            .iter()
            .map(|&m| collect_upper_signature(&mut ts, &diagonal_pattern(m), 500))
            .collect();
        let class_b: Vec<Vec<f64>> = test_mags
            .iter()
            .map(|&m| collect_upper_signature(&mut ts, &anti_diagonal_pattern(m), 500))
            .collect();

        let sep = separability(&class_a, &class_b).expect("separability must compute");
        assert!(
            sep.index > 2.0,
            "expected class separability > 2, got {} (radius {}, centroid dist {})",
            sep.index,
            sep.mean_radius,
            sep.centroid_distance,
        );
    }

    /// The compass-metric claim: with coupling, the system achieves class
    /// discrimination at lower PE energy than without coupling. Same
    /// information, fewer spikes — the "low-power AI" promise of predictive
    /// coding. Both conditions remain separable; coupling adds efficiency.
    #[test]
    fn coupling_reduces_pe_energy_at_comparable_separability() {
        let train_and_probe = |coupling: bool| -> (f64, usize) {
            let mut ts = TwoStage::with_defaults(4, 4, 2);
            if !coupling {
                ts.disable_coupling();
            }
            let presentation_steps = 200;
            for cycle in 0..50 {
                let mag_a = 1.8 + 0.4 * ((cycle * 7) % 5) as f64 / 5.0;
                let mag_b = 1.8 + 0.4 * ((cycle * 11) % 5) as f64 / 5.0;
                for _ in 0..presentation_steps {
                    ts.step(&diagonal_pattern(mag_a), 0.1);
                }
                for _ in 0..presentation_steps {
                    ts.step(&anti_diagonal_pattern(mag_b), 0.1);
                }
            }

            // Probe: collect signatures AND total PE energy used during probing.
            let test_mags = [1.85, 1.95, 2.05, 2.15];
            let mut total_pe = 0usize;
            let mut measure = |pattern: &[f64]| -> Vec<f64> {
                let n_up = ts.upper.units.len();
                let mut sig = vec![0.0; n_up];
                for _ in 0..500 {
                    let out = ts.step(pattern, 0.1);
                    total_pe += out.pe_total();
                    for i in 0..n_up {
                        sig[i] += out.upper_pe_plus[i] as f64 - out.upper_pe_minus[i] as f64;
                    }
                }
                sig
            };
            let class_a: Vec<Vec<f64>> = test_mags.iter().map(|&m| measure(&diagonal_pattern(m))).collect();
            let class_b: Vec<Vec<f64>> = test_mags.iter().map(|&m| measure(&anti_diagonal_pattern(m))).collect();
            let sep = separability(&class_a, &class_b).map(|s| s.index).unwrap_or(0.0);
            (sep, total_pe)
        };

        let (sep_on, pe_on) = train_and_probe(true);
        let (sep_off, pe_off) = train_and_probe(false);

        // Both must achieve clear separability.
        assert!(sep_on > 2.0, "coupling-on must remain separable, got {}", sep_on);
        assert!(sep_off > 2.0, "coupling-off must remain separable, got {}", sep_off);

        // Coupling must reduce PE energy used during probing.
        assert!(
            pe_on < pe_off,
            "coupling should reduce PE energy: on={}, off={} (sep_on={}, sep_off={})",
            pe_on,
            pe_off,
            sep_on,
            sep_off,
        );
    }
}
