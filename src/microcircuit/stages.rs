//! N-stage stacked predictive-coding architecture.
//!
//! `MultiStage` holds an ordered list of microcircuit grids stacked bottom
//! to top. Between each adjacent pair, a non-overlapping square pooling
//! factor governs routing: each upper-stage cell pools PE+ activity from a
//! `pool × pool` square of lower cells, and its prediction (μ̂) is broadcast
//! back down to every lower cell in that square as top-down input.
//!
//! Routing preserves retinotopy at every interface: a single hot input cell
//! drives activity through the spatially-corresponding cell at every stage
//! up the stack, never the spatial mirror.

use serde::{Deserialize, Serialize};

use crate::microcircuit::canonical::{Microcircuit, MicrocircuitParams};
use crate::microcircuit::plasticity::{HebbianCa, HebbianParams};

#[derive(Clone, Debug, Serialize, Deserialize)]
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultiStage {
    pub stages: Vec<StageGrid>,
    /// Pool factor between stage[i] and stage[i+1]. Length = stages.len() - 1.
    pub pools: Vec<usize>,
    /// Feedforward divisor: weighted PE+ contribution divided by this to
    /// match historical scaling.
    pub ff_gain: f64,
    /// Feedback gain applied to higher stage's μ̂ before injecting as
    /// top-down to the lower stage's receptive-field cells.
    pub fb_gain: f64,
    /// Plastic per-connection feedforward weights. `ff_weights[stage_pair]`
    /// has one [`HebbianCa`] per lower-stage cell (giving its weight to
    /// its single parent cell in the next stage). Length = stages.len() − 1.
    /// Weights start at 1.0 (uniform routing); Hebbian-Ca updates strengthen
    /// connections whose child→parent PE+ activity coincides.
    pub ff_weights: Vec<Vec<HebbianCa>>,
    /// Master switch for Hebbian updates on feedforward weights. Off by
    /// default so existing tests with synthetic patterns run with stable
    /// uniform routing. Enable to learn pattern-selective connections.
    pub plasticity_enabled: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MultiStageStepOutput {
    /// Per-stage PE+ counts (outer index = stage, bottom to top).
    pub pe_plus: Vec<Vec<usize>>,
    /// Per-stage PE− counts.
    pub pe_minus: Vec<Vec<usize>>,
}

impl MultiStageStepOutput {
    pub fn stage_pe_total(&self, stage: usize) -> usize {
        self.pe_plus[stage].iter().sum::<usize>() + self.pe_minus[stage].iter().sum::<usize>()
    }

    pub fn pe_total(&self) -> usize {
        (0..self.pe_plus.len()).map(|s| self.stage_pe_total(s)).sum()
    }
}

impl MultiStage {
    /// Build from a list of stage shapes (bottom to top) and adjacent-pair pools.
    /// `shapes[i+1].x * pools[i] == shapes[i].x` (same for y) must hold.
    pub fn new(shapes: &[(usize, usize)], pools: &[usize], params: MicrocircuitParams) -> Self {
        assert!(shapes.len() >= 1);
        assert_eq!(pools.len() + 1, shapes.len(), "pools must have len = stages - 1");
        for (i, &p) in pools.iter().enumerate() {
            assert!(p > 0);
            assert_eq!(
                shapes[i].0,
                shapes[i + 1].0 * p,
                "x mismatch at stage {}: {} != {} * {}",
                i,
                shapes[i].0,
                shapes[i + 1].0,
                p
            );
            assert_eq!(shapes[i].1, shapes[i + 1].1 * p, "y mismatch at stage {}", i);
        }
        let stages: Vec<StageGrid> = shapes
            .iter()
            .map(|&(x, y)| StageGrid::new(x, y, params.clone()))
            .collect();
        // One Hebbian-Ca per lower-stage cell, giving its weight to its
        // single parent cell. Slow learning rates by default so weights
        // are nearly static unless plasticity_enabled is on for many steps.
        let hebbian_params = HebbianParams {
            tau_pre: 20.0,
            tau_post: 20.0,
            eta_ltp: 5e-4,
            eta_ltd: 5e-4,
            lambda_l1: 1e-6,
            w_min: 0.0,
            w_max: 5.0,
        };
        let ff_weights: Vec<Vec<HebbianCa>> = (0..stages.len().saturating_sub(1))
            .map(|i| (0..stages[i].units.len()).map(|_| HebbianCa::new(hebbian_params.clone(), 1.0)).collect())
            .collect();
        Self {
            stages,
            pools: pools.to_vec(),
            ff_gain: 0.3,
            fb_gain: 1.0,
            ff_weights,
            plasticity_enabled: false,
        }
    }

    pub fn enable_plasticity(&mut self) {
        self.plasticity_enabled = true;
    }

    pub fn disable_plasticity(&mut self) {
        self.plasticity_enabled = false;
    }

    /// Serialize the entire model state (weights + per-cell state) to a
    /// bincode-encoded byte vector. Round-trips through [`Self::load`].
    pub fn save(&self) -> Vec<u8> {
        bincode::serialize(self).expect("MultiStage serialize")
    }

    /// Deserialize from bytes produced by [`Self::save`].
    pub fn load(bytes: &[u8]) -> Result<Self, bincode::Error> {
        bincode::deserialize(bytes)
    }

    /// Save the model to a file (bincode binary).
    pub fn save_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> std::io::Result<()> {
        std::fs::write(path, self.save())
    }

    /// Load a model from a file written by [`Self::save_to_file`].
    pub fn load_from_file<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        Self::load(&bytes).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    pub fn with_defaults(shapes: &[(usize, usize)], pools: &[usize]) -> Self {
        Self::new(shapes, pools, MicrocircuitParams::default())
    }

    pub fn reset(&mut self) {
        for s in &mut self.stages {
            s.reset();
        }
    }

    pub fn disable_coupling(&mut self) {
        for s in &mut self.stages {
            s.disable_coupling();
        }
    }

    /// Set the prediction integration rate `eta` for every cell in one stage.
    /// Lets a stack carry a timescale gradient (fast lower → slow upper) so
    /// different stages select different input frequencies.
    pub fn set_stage_eta(&mut self, stage_idx: usize, eta: f64) {
        for u in &mut self.stages[stage_idx].units {
            u.params.eta = eta;
        }
    }

    /// Parent cell index in stage[i+1] for a child cell at index `child_i` in stage[i].
    fn parent_idx(&self, lower_stage: usize, child_i: usize) -> usize {
        let pool = self.pools[lower_stage];
        let cells_x_low = self.stages[lower_stage].cells_x;
        let cells_x_up = self.stages[lower_stage + 1].cells_x;
        let lx = child_i % cells_x_low;
        let ly = child_i / cells_x_low;
        let ux = lx / pool;
        let uy = ly / pool;
        uy * cells_x_up + ux
    }

    pub fn step(&mut self, input: &[f64], dt_ms: f64) -> MultiStageStepOutput {
        let n_stages = self.stages.len();
        assert_eq!(input.len(), self.stages[0].units.len(), "input shape mismatch");

        // Pre-fetch all μ̂ vectors so feedback uses the state at start-of-step.
        let mu_per_stage: Vec<Vec<f64>> = self.stages.iter().map(|s| s.predictions()).collect();

        let mut pe_plus: Vec<Vec<usize>> = vec![Vec::new(); n_stages];
        let mut pe_minus: Vec<Vec<usize>> = vec![Vec::new(); n_stages];

        // Walk bottom to top.
        let mut current_input: Vec<f64> = input.to_vec();
        for i in 0..n_stages {
            let n_cells = self.stages[i].units.len();
            let top_down: Vec<f64> = if i + 1 < n_stages {
                let upper_mu = &mu_per_stage[i + 1];
                (0..n_cells)
                    .map(|child_i| self.fb_gain * upper_mu[self.parent_idx(i, child_i)])
                    .collect()
            } else {
                vec![0.0; n_cells]
            };

            let (pp, pm) = self.stages[i].step(&current_input, &top_down, dt_ms);

            // Build next stage's input from PLASTIC-WEIGHTED PE+ of this stage.
            if i + 1 < n_stages {
                let n_up = self.stages[i + 1].units.len();
                let mut pooled = vec![0.0f64; n_up];
                for (child_i, &pp_count) in pp.iter().enumerate() {
                    let parent_i = self.parent_idx(i, child_i);
                    let w = self.ff_weights[i][child_i].w;
                    pooled[parent_i] += w * pp_count as f64;
                }
                for v in &mut pooled {
                    *v /= self.ff_gain;
                }

                // Pre-spikes: register each lower PE+ event on its outgoing
                // Hebbian connection. Plasticity update on the weight happens
                // here (uses recent post-trace for LTD term).
                if self.plasticity_enabled {
                    for (child_i, &pp_count) in pp.iter().enumerate() {
                        for _ in 0..pp_count {
                            self.ff_weights[i][child_i].pre_spike();
                        }
                    }
                }

                current_input = pooled;
            }

            pe_plus[i] = pp;
            pe_minus[i] = pm;
        }

        // Post-spikes: walk the stage pairs, register each parent's PE+ on
        // all its children's outgoing connections. LTP term uses each
        // child's pre-trace from earlier in this step.
        if self.plasticity_enabled {
            for stage_pair in 0..self.stages.len().saturating_sub(1) {
                let n_low = self.stages[stage_pair].units.len();
                for child_i in 0..n_low {
                    let parent_i = self.parent_idx(stage_pair, child_i);
                    let parent_pp = pe_plus[stage_pair + 1][parent_i];
                    for _ in 0..parent_pp {
                        self.ff_weights[stage_pair][child_i].post_spike();
                    }
                }
            }
            // Decay traces and apply L1 weight decay.
            for stage_weights in &mut self.ff_weights {
                for w in stage_weights {
                    w.step(dt_ms);
                }
            }
        }

        MultiStageStepOutput { pe_plus, pe_minus }
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

    fn diagonal_pattern(mag: f64) -> Vec<f64> {
        let mut v = vec![0.0; 16];
        v[0] = mag;
        v[3 * 4 + 3] = mag;
        v
    }

    fn anti_diagonal_pattern(mag: f64) -> Vec<f64> {
        let mut v = vec![0.0; 16];
        v[3] = mag;
        v[3 * 4] = mag;
        v
    }

    /// Build the canonical 2-stage 4×4 → 2×2 network used by the migrated tests.
    fn two_stage() -> MultiStage {
        MultiStage::with_defaults(&[(4, 4), (2, 2)], &[2])
    }

    fn run_steady(ms: &mut MultiStage, input: &[f64], n_steps: usize) {
        for _ in 0..n_steps {
            ms.step(input, 0.1);
        }
    }

    fn collect_top_signature(ms: &mut MultiStage, pattern: &[f64], n_steps: usize) -> Vec<f64> {
        let top = ms.stages.len() - 1;
        let n = ms.stages[top].units.len();
        let mut sig = vec![0.0; n];
        for _ in 0..n_steps {
            let out = ms.step(pattern, 0.1);
            for i in 0..n {
                sig[i] += out.pe_plus[top][i] as f64 - out.pe_minus[top][i] as f64;
            }
        }
        sig
    }

    #[test]
    fn retinotopy_preserved_to_upper_stage() {
        let mut ms = two_stage();
        let input = one_hot(4, 4, 3, 1, 2.0);
        run_steady(&mut ms, &input, 5000);
        let upper_mu = ms.stages[1].predictions();
        let target_idx = 0 * 2 + 1; // upper (1, 0)
        let target_mu = upper_mu[target_idx];
        assert!(target_mu > 0.1, "target upper cell μ̂ should be positive, got {}", target_mu);
        for (i, &mu) in upper_mu.iter().enumerate() {
            if i != target_idx {
                assert!(
                    mu.abs() < target_mu * 0.5,
                    "non-target upper cell {} μ̂ {} too close to target {}",
                    i, mu, target_mu,
                );
            }
        }
    }

    #[test]
    fn lower_pe_falls_as_upper_learns() {
        let mut ms = two_stage();
        let input = one_hot(4, 4, 1, 1, 2.0);
        let mut early_pe = 0usize;
        for _ in 0..2000 {
            early_pe += ms.step(&input, 0.1).stage_pe_total(0);
        }
        run_steady(&mut ms, &input, 20_000);
        let mut late_pe = 0usize;
        for _ in 0..2000 {
            late_pe += ms.step(&input, 0.1).stage_pe_total(0);
        }
        assert!(late_pe < early_pe, "lower PE must fall as upper predicts: early={}, late={}", early_pe, late_pe);
    }

    #[test]
    fn coupling_off_no_lower_pe_reduction() {
        let mut ms = two_stage();
        ms.disable_coupling();
        let input = one_hot(4, 4, 1, 1, 2.0);
        let mut early_pe = 0usize;
        for _ in 0..2000 {
            early_pe += ms.step(&input, 0.1).stage_pe_total(0);
        }
        run_steady(&mut ms, &input, 20_000);
        let mut late_pe = 0usize;
        for _ in 0..2000 {
            late_pe += ms.step(&input, 0.1).stage_pe_total(0);
        }
        let ratio = late_pe as f64 / early_pe.max(1) as f64;
        assert!(ratio > 0.8, "coupling off should keep PE flat: ratio={}", ratio);
    }

    #[test]
    fn no_spatial_crosstalk() {
        let mut ms = two_stage();
        let input = one_hot(4, 4, 0, 0, 2.0);
        run_steady(&mut ms, &input, 10_000);
        let upper_mu = ms.stages[1].predictions();
        let target = upper_mu[0];
        for i in 1..upper_mu.len() {
            assert!(
                upper_mu[i].abs() < target.abs().max(0.1) * 0.5,
                "non-target upper cell {} ({}) too close to target {}",
                i, upper_mu[i], target,
            );
        }
    }

    #[test]
    fn upper_stage_discriminates_two_classes() {
        let mut ms = two_stage();
        let presentation_steps = 200;
        for cycle in 0..50 {
            let mag_a = 1.8 + 0.4 * ((cycle * 7) % 5) as f64 / 5.0;
            let mag_b = 1.8 + 0.4 * ((cycle * 11) % 5) as f64 / 5.0;
            for _ in 0..presentation_steps {
                ms.step(&diagonal_pattern(mag_a), 0.1);
            }
            for _ in 0..presentation_steps {
                ms.step(&anti_diagonal_pattern(mag_b), 0.1);
            }
        }
        let test_mags = [1.85, 1.95, 2.05, 2.15];
        let class_a: Vec<Vec<f64>> = test_mags
            .iter()
            .map(|&m| collect_top_signature(&mut ms, &diagonal_pattern(m), 500))
            .collect();
        let class_b: Vec<Vec<f64>> = test_mags
            .iter()
            .map(|&m| collect_top_signature(&mut ms, &anti_diagonal_pattern(m), 500))
            .collect();
        let sep = separability(&class_a, &class_b).expect("separability must compute");
        assert!(sep.index > 2.0, "expected class separability > 2, got {}", sep.index);
    }

    #[test]
    fn coupling_reduces_pe_energy_at_comparable_separability() {
        let train_and_probe = |coupling: bool| -> (f64, usize) {
            let mut ms = two_stage();
            if !coupling {
                ms.disable_coupling();
            }
            let presentation_steps = 200;
            for cycle in 0..50 {
                let mag_a = 1.8 + 0.4 * ((cycle * 7) % 5) as f64 / 5.0;
                let mag_b = 1.8 + 0.4 * ((cycle * 11) % 5) as f64 / 5.0;
                for _ in 0..presentation_steps {
                    ms.step(&diagonal_pattern(mag_a), 0.1);
                }
                for _ in 0..presentation_steps {
                    ms.step(&anti_diagonal_pattern(mag_b), 0.1);
                }
            }
            let test_mags = [1.85, 1.95, 2.05, 2.15];
            let mut total_pe = 0usize;
            let top = ms.stages.len() - 1;
            let mut measure = |pattern: &[f64]| -> Vec<f64> {
                let n_up = ms.stages[top].units.len();
                let mut sig = vec![0.0; n_up];
                for _ in 0..500 {
                    let out = ms.step(pattern, 0.1);
                    total_pe += out.pe_total();
                    for i in 0..n_up {
                        sig[i] += out.pe_plus[top][i] as f64 - out.pe_minus[top][i] as f64;
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
        assert!(sep_on > 2.0);
        assert!(sep_off > 2.0);
        assert!(
            pe_on < pe_off,
            "coupling should reduce PE energy: on={}, off={} (sep_on={}, sep_off={})",
            pe_on, pe_off, sep_on, sep_off,
        );
    }

    // ------------- 3-stage tests -------------

    /// 3 stages: 8×8 → 4×4 → 2×2 with pool=2 at each interface. A hot input
    /// cell at lower position (5, 3) should activate the spatially-correct
    /// cell at the top stage: (5/4, 3/4) = (1, 0).
    #[test]
    fn retinotopy_preserved_through_three_stages() {
        let mut ms = MultiStage::with_defaults(&[(8, 8), (4, 4), (2, 2)], &[2, 2]);
        let input = one_hot(8, 8, 5, 3, 2.5);
        run_steady(&mut ms, &input, 10_000);
        let top_mu = ms.stages[2].predictions();
        let target_idx = 0 * 2 + 1; // top (1, 0)
        let target_mu = top_mu[target_idx];
        assert!(target_mu > 0.1, "top cell μ̂ should be positive, got {} (full: {:?})", target_mu, top_mu);
        for (i, &mu) in top_mu.iter().enumerate() {
            if i != target_idx {
                assert!(
                    mu.abs() < target_mu * 0.5,
                    "non-target top cell {} μ̂ {} too close to target {}",
                    i, mu, target_mu,
                );
            }
        }
    }

    /// Negative test from the design plan: with excessive pooling (one big
    /// jump from 8×8 → 1×1 instead of the staged 8 → 4 → 2), the top stage
    /// collapses to a single cell that cannot spatially discriminate.
    /// Class signatures from opposite quadrants should fail to separate.
    #[test]
    fn over_pooling_collapses_discrimination() {
        // Two-class spatial task: hot upper-left vs hot lower-right.
        fn ul_pattern(mag: f64) -> Vec<f64> {
            let mut v = vec![0.0; 64];
            for dy in 0..3 {
                for dx in 0..3 {
                    v[dy * 8 + dx] = mag;
                }
            }
            v
        }
        fn lr_pattern(mag: f64) -> Vec<f64> {
            let mut v = vec![0.0; 64];
            for dy in 5..8 {
                for dx in 5..8 {
                    v[dy * 8 + dx] = mag;
                }
            }
            v
        }

        fn train_and_measure(shapes: &[(usize, usize)], pools: &[usize]) -> f64 {
            let mut ms = MultiStage::with_defaults(shapes, pools);
            let presentation_steps = 200;
            for cycle in 0..40 {
                let mag_a = 1.8 + 0.4 * ((cycle * 7) % 5) as f64 / 5.0;
                let mag_b = 1.8 + 0.4 * ((cycle * 11) % 5) as f64 / 5.0;
                for _ in 0..presentation_steps {
                    ms.step(&ul_pattern(mag_a), 0.1);
                }
                for _ in 0..presentation_steps {
                    ms.step(&lr_pattern(mag_b), 0.1);
                }
            }

            let test_mags = [1.85, 1.95, 2.05, 2.15];
            let top = ms.stages.len() - 1;
            let n_up = ms.stages[top].units.len();
            let mut measure = |pattern: &[f64]| -> Vec<f64> {
                let mut sig = vec![0.0; n_up];
                for _ in 0..500 {
                    let out = ms.step(pattern, 0.1);
                    for i in 0..n_up {
                        sig[i] += out.pe_plus[top][i] as f64 - out.pe_minus[top][i] as f64;
                    }
                }
                sig
            };
            let class_a: Vec<Vec<f64>> = test_mags.iter().map(|&m| measure(&ul_pattern(m))).collect();
            let class_b: Vec<Vec<f64>> = test_mags.iter().map(|&m| measure(&lr_pattern(m))).collect();
            separability(&class_a, &class_b).map(|s| s.index).unwrap_or(0.0)
        }

        // Sensible: 8 → 4 → 2, two pool-2 steps.
        let sensible = train_and_measure(&[(8, 8), (4, 4), (2, 2)], &[2, 2]);
        // Over-pooled: 8 → 1, single pool-8 step. Top stage is one cell.
        let collapsed = train_and_measure(&[(8, 8), (1, 1)], &[8]);

        assert!(
            sensible > collapsed,
            "sensible pooling should outperform over-pool: sensible={}, collapsed={}",
            sensible,
            collapsed,
        );
    }

    /// Deterministic per-step pseudo-random noise (no rand dep needed).
    fn noise_sample(state: &mut u64) -> f64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        (*state as f64 / u64::MAX as f64) * 2.0 - 1.0
    }

    fn class_a_pattern(mag: f64) -> Vec<f64> {
        // Vertical stripe down the left third.
        let mut v = vec![0.0; 64];
        for y in 0..8 {
            for x in 0..2 {
                v[y * 8 + x] = mag;
            }
        }
        v
    }

    fn class_b_pattern(mag: f64) -> Vec<f64> {
        // Horizontal stripe across the top third.
        let mut v = vec![0.0; 64];
        for y in 0..2 {
            for x in 0..8 {
                v[y * 8 + x] = mag;
            }
        }
        v
    }

    fn jitter(base: &[f64], noise_amp: f64, state: &mut u64) -> Vec<f64> {
        base.iter().map(|&v| (v + noise_amp * noise_sample(state)).max(0.0)).collect()
    }

    /// Phase 2b: a timescale gradient through the stack produces noise-
    /// robust class discrimination at the top. With fast `eta` everywhere
    /// (flat profile), per-cell fast noise on the input contaminates the
    /// top-stage representation and reduces class separability. With a
    /// gradient (fast lower → slow upper), the upper stage filters out
    /// the noise as a fast component and retains the slow class identity.
    #[test]
    fn timescale_gradient_yields_noise_robust_class_separation() {
        fn train_and_measure(stage_etas: &[f64]) -> f64 {
            let mut ms = MultiStage::with_defaults(&[(8, 8), (4, 4), (2, 2)], &[2, 2]);
            for (i, &e) in stage_etas.iter().enumerate() {
                ms.set_stage_eta(i, e);
            }
            let mut state: u64 = 0x9E3779B97F4A7C15;

            let presentation_steps = 300;
            for cycle in 0..40 {
                let mag_a = 1.8 + 0.4 * ((cycle * 7) % 5) as f64 / 5.0;
                let mag_b = 1.8 + 0.4 * ((cycle * 11) % 5) as f64 / 5.0;
                let base_a = class_a_pattern(mag_a);
                let base_b = class_b_pattern(mag_b);
                for _ in 0..presentation_steps {
                    let s = jitter(&base_a, 0.8, &mut state);
                    ms.step(&s, 0.1);
                }
                for _ in 0..presentation_steps {
                    let s = jitter(&base_b, 0.8, &mut state);
                    ms.step(&s, 0.1);
                }
            }

            let test_mags = [1.85, 1.95, 2.05, 2.15];
            let top = ms.stages.len() - 1;
            let n_up = ms.stages[top].units.len();
            let mut measure_for = |base: Vec<f64>| -> Vec<f64> {
                let mut sig = vec![0.0; n_up];
                for _ in 0..1000 {
                    let s = jitter(&base, 0.8, &mut state);
                    let out = ms.step(&s, 0.1);
                    for i in 0..n_up {
                        sig[i] += out.pe_plus[top][i] as f64 - out.pe_minus[top][i] as f64;
                    }
                }
                sig
            };
            let class_a: Vec<Vec<f64>> =
                test_mags.iter().map(|&m| measure_for(class_a_pattern(m))).collect();
            let class_b: Vec<Vec<f64>> =
                test_mags.iter().map(|&m| measure_for(class_b_pattern(m))).collect();
            separability(&class_a, &class_b).map(|s| s.index).unwrap_or(0.0)
        }

        // Flat profile: same fast eta everywhere — no timescale gradient.
        let flat = train_and_measure(&[0.0005, 0.0005, 0.0005]);
        // Gradient: lower stages track fast input incl. noise; upper stage
        // has much smaller eta, so its prediction filters out the fast noise
        // and retains the slow class identity.
        let gradient = train_and_measure(&[0.0005, 0.0001, 0.00003]);

        assert!(
            gradient > flat,
            "timescale gradient should outperform flat profile: gradient={}, flat={}",
            gradient,
            flat,
        );
    }

    /// Save/load round-trip preserves learned weights and per-cell state.
    /// Demonstrates that "memorization" via Hebbian-learned weights persists
    /// across save/load — once the network learns, the knowledge can be
    /// frozen to disk and restored.
    #[test]
    fn save_load_round_trip_preserves_state() {
        let mut ms = MultiStage::with_defaults(&[(4, 4), (2, 2)], &[2]);
        ms.enable_plasticity();
        let input = one_hot(4, 4, 1, 1, 2.0);
        run_steady(&mut ms, &input, 20_000);

        let trained_w = ms.ff_weights[0][1 * 4 + 1].w;
        let trained_mu: Vec<f64> = ms.stages[1].predictions();

        // Round-trip through bytes.
        let bytes = ms.save();
        let restored = MultiStage::load(&bytes).expect("deserialize");
        assert_eq!(restored.ff_weights[0][1 * 4 + 1].w, trained_w);
        assert_eq!(restored.stages[1].predictions(), trained_mu);
    }

    /// File round-trip works the same way.
    #[test]
    fn save_load_via_file() {
        let mut ms = MultiStage::with_defaults(&[(4, 4), (2, 2)], &[2]);
        ms.enable_plasticity();
        let input = one_hot(4, 4, 2, 2, 2.0);
        run_steady(&mut ms, &input, 10_000);

        let tmp = tempfile::NamedTempFile::new().unwrap();
        ms.save_to_file(tmp.path()).expect("save");
        let restored = MultiStage::load_from_file(tmp.path()).expect("load");
        assert_eq!(
            restored.ff_weights[0][2 * 4 + 2].w,
            ms.ff_weights[0][2 * 4 + 2].w
        );
    }

    /// With plasticity on, connections whose child→parent PE+ activity
    /// frequently coincides get strengthened. Drive a single hot lower
    /// cell for many steps; its outgoing connection grows above baseline,
    /// while connections from quiet cells stay near baseline.
    #[test]
    fn feedforward_weights_strengthen_on_active_connections() {
        let mut ms = MultiStage::with_defaults(&[(4, 4), (2, 2)], &[2]);
        ms.enable_plasticity();
        let input = one_hot(4, 4, 1, 1, 2.0);

        let initial_active = ms.ff_weights[0][1 * 4 + 1].w;
        let initial_quiet = ms.ff_weights[0][3 * 4 + 3].w;
        assert_eq!(initial_active, 1.0);
        assert_eq!(initial_quiet, 1.0);

        run_steady(&mut ms, &input, 30_000);

        let final_active = ms.ff_weights[0][1 * 4 + 1].w;
        let final_quiet = ms.ff_weights[0][3 * 4 + 3].w;
        assert!(
            final_active > 1.05,
            "active connection should strengthen, got {} → {}",
            initial_active,
            final_active,
        );
        assert!(
            final_active > final_quiet + 0.05,
            "active connection should out-grow quiet one: active={}, quiet={}",
            final_active,
            final_quiet,
        );
    }

    /// PE total at the BOTTOM stage of a 3-stack should fall with learning,
    /// driven by the cascade of predictions descending from the top.
    #[test]
    fn three_stage_bottom_pe_falls_with_learning() {
        let mut ms = MultiStage::with_defaults(&[(8, 8), (4, 4), (2, 2)], &[2, 2]);
        let input = one_hot(8, 8, 2, 2, 2.0);
        let mut early_pe = 0usize;
        for _ in 0..2000 {
            early_pe += ms.step(&input, 0.1).stage_pe_total(0);
        }
        run_steady(&mut ms, &input, 40_000);
        let mut late_pe = 0usize;
        for _ in 0..2000 {
            late_pe += ms.step(&input, 0.1).stage_pe_total(0);
        }
        assert!(late_pe < early_pe, "bottom PE must fall: early={}, late={}", early_pe, late_pe);
    }
}
