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

#[derive(Clone, Debug)]
pub struct MultiStage {
    pub stages: Vec<StageGrid>,
    /// Pool factor between stage[i] and stage[i+1]. Length = stages.len() - 1.
    pub pools: Vec<usize>,
    /// Feedforward divisor: pooled PE+ spike count divided by this to feed
    /// the next stage's scalar input.
    pub ff_gain: f64,
    /// Feedback gain applied to higher stage's μ̂ before injecting as
    /// top-down to the lower stage's receptive-field cells.
    pub fb_gain: f64,
}

#[derive(Clone, Debug, Default)]
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
        Self { stages, pools: pools.to_vec(), ff_gain: 0.3, fb_gain: 1.0 }
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

            // Build next stage's input from pooled PE+ of this stage.
            if i + 1 < n_stages {
                let n_up = self.stages[i + 1].units.len();
                let mut pooled = vec![0.0f64; n_up];
                for (child_i, &pp_count) in pp.iter().enumerate() {
                    pooled[self.parent_idx(i, child_i)] += pp_count as f64;
                }
                for v in &mut pooled {
                    *v /= self.ff_gain;
                }
                current_input = pooled;
            }

            pe_plus[i] = pp;
            pe_minus[i] = pm;
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
