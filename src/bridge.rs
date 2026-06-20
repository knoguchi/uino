//! Bridge from [`retinula::RetinaOutput`] to cortex input vectors.
//!
//! Retinula's batched `simulate()` returns full spike trains for a
//! presentation. The bridge exposes two views of that output:
//!
//! - [`RetinaBridge::mean_rates`]: per-cell firing rate averaged over the
//!   whole presentation (Hz). The right view for static-image testing —
//!   constant input to the cortex for many timesteps.
//! - [`RetinaBridge::rates_at`]: per-step smoothed rate trace (binned spike
//!   counts with exponential decay). The right view for dynamic input
//!   (video, streaming).
//!
//! Output shape matches the retinula grid (cells_x × cells_y) row-major,
//! ON-channel only for v0.

use retinula::RetinaOutput;

/// Cortex-aligned view of retinula output.
pub struct RetinaBridge {
    cells_x: usize,
    cells_y: usize,
    /// Per-cell firing rate over the whole presentation (Hz).
    mean_rates_hz: Vec<f64>,
    /// `on_smoothed[step][cell]` = exponentially-smoothed spike trace at
    /// cortex step `step`. Use for dynamic/streaming input.
    on_smoothed: Vec<Vec<f64>>,
    /// Raw spike count per (step, cell).
    on_counts: Vec<Vec<usize>>,
    dt_s: f64,
}

impl RetinaBridge {
    /// `dt_ms`: cortex timestep. `tau_ms`: smoothing time constant
    /// (5 ms ≈ AMPA EPSC).
    pub fn from_retina_output(
        output: &RetinaOutput,
        cells_x: usize,
        cells_y: usize,
        dt_ms: f64,
        tau_ms: f64,
    ) -> Self {
        assert!(dt_ms > 0.0 && tau_ms > 0.0);
        assert_eq!(
            output.on_cells.len(),
            cells_x * cells_y,
            "retina output cell count ({}) doesn't match grid {}×{}",
            output.on_cells.len(),
            cells_x,
            cells_y,
        );
        let dt_s = dt_ms * 1e-3;
        let n_steps = (output.duration / dt_s).floor() as usize;
        let n_cells = cells_x * cells_y;

        let mut on_counts: Vec<Vec<usize>> = (0..n_steps).map(|_| vec![0usize; n_cells]).collect();
        for (cell_idx, cell) in output.on_cells.iter().enumerate() {
            for &t in &cell.spike_times {
                if t < 0.0 || t >= output.duration {
                    continue;
                }
                let step = (t / dt_s).floor() as usize;
                if step < n_steps {
                    on_counts[step][cell_idx] += 1;
                }
            }
        }

        let alpha = (-dt_ms / tau_ms).exp();
        let mut trace = vec![0.0f64; n_cells];
        let mut on_smoothed: Vec<Vec<f64>> = Vec::with_capacity(n_steps);
        for step in 0..n_steps {
            for c in 0..n_cells {
                trace[c] = alpha * trace[c] + on_counts[step][c] as f64;
            }
            on_smoothed.push(trace.clone());
        }

        let mean_rates_hz: Vec<f64> = output.on_cells.iter().map(|c| c.firing_rate).collect();

        Self { cells_x, cells_y, mean_rates_hz, on_smoothed, on_counts, dt_s }
    }

    pub fn cells_x(&self) -> usize {
        self.cells_x
    }
    pub fn cells_y(&self) -> usize {
        self.cells_y
    }
    pub fn n_steps(&self) -> usize {
        self.on_smoothed.len()
    }
    pub fn dt_s(&self) -> f64 {
        self.dt_s
    }

    /// Per-cell firing rate over the whole presentation (Hz), scaled.
    /// Right form for static-image cortex input.
    pub fn mean_rates(&self, scale: f64) -> Vec<f64> {
        self.mean_rates_hz.iter().map(|&r| r * scale).collect()
    }

    pub fn counts_at(&self, step: usize) -> &[usize] {
        &self.on_counts[step]
    }

    /// Per-step smoothed rate trace, scaled. Right form for dynamic input.
    pub fn rates_at(&self, step: usize, scale: f64) -> Vec<f64> {
        self.on_smoothed[step].iter().map(|&r| r * scale).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::microcircuit::MultiStage;
    use retinula::Retina;

    /// Drive a 5×5 bright patch in the upper-left quadrant. The cortex's
    /// top-stage cell at the spatially-corresponding position (top-left
    /// quadrant) should learn a larger prediction than its mirror.
    #[test]
    fn retina_drives_cortex_at_correct_position() {
        let img_w = 32;
        let img_h = 32;
        let cells = 8;
        let mut retina = Retina::for_image(img_w, img_h)
            .resolution(cells, cells)
            .on_cells_only()
            .no_eccentricity()
            .seed(42);

        // Bright patch in upper-left quadrant.
        let mut img = vec![0.0; img_w * img_h];
        for dy in 6..11 {
            for dx in 6..11 {
                img[dy * img_w + dx] = 1.0;
            }
        }
        let output = retina.simulate(&img, 0.5);
        assert!(output.total_spikes() > 100);

        let bridge = RetinaBridge::from_retina_output(&output, cells, cells, 0.1, 5.0);

        // Use mean-rates (per-cell Hz averaged over the presentation) as a
        // constant cortex input. Scale so cell firing rates land in the
        // cortex working range (~1–3).
        let scale = 1.0 / 30.0; // 70 Hz → 2.3; 90 Hz → 3.0
        let s = bridge.mean_rates(scale);

        let mut cortex = MultiStage::with_defaults(&[(8, 8), (4, 4), (2, 2)], &[2, 2]);
        for _ in 0..30_000 {
            cortex.step(&s, 0.1);
        }

        let top_mu = cortex.stages[2].predictions();
        let target_idx = 0; // top-left quadrant → (0,0)
        let anti_idx = 3; // bottom-right → (1,1)
        assert!(
            top_mu[target_idx] > top_mu[anti_idx],
            "target top cell μ̂ ({}) should exceed anti-target ({}). Full top μ̂: {:?}",
            top_mu[target_idx],
            top_mu[anti_idx],
            top_mu,
        );
        assert!(
            top_mu[target_idx] > 0.0,
            "target top cell μ̂ should be positive, got {}",
            top_mu[target_idx],
        );
    }

    /// Two images with hot patches in opposite quadrants should produce
    /// distinguishable cortex top-stage μ̂ patterns — the architecture
    /// discriminates retinal input by spatial structure.
    #[test]
    fn cortex_distinguishes_opposite_patches() {
        fn make_cortex_response(patch_origin: (usize, usize)) -> Vec<f64> {
            let img_w = 32;
            let img_h = 32;
            let cells = 8;
            let mut retina = Retina::for_image(img_w, img_h)
                .resolution(cells, cells)
                .on_cells_only()
                .no_eccentricity()
                .seed(42);

            let mut img = vec![0.0; img_w * img_h];
            for dy in 0..5 {
                for dx in 0..5 {
                    img[(patch_origin.1 + dy) * img_w + (patch_origin.0 + dx)] = 1.0;
                }
            }
            let output = retina.simulate(&img, 0.5);
            let bridge = RetinaBridge::from_retina_output(&output, cells, cells, 0.1, 5.0);
            let s = bridge.mean_rates(1.0 / 30.0);

            let mut cortex = MultiStage::with_defaults(&[(8, 8), (4, 4), (2, 2)], &[2, 2]);
            for _ in 0..30_000 {
                cortex.step(&s, 0.1);
            }
            cortex.stages[2].predictions()
        }

        let upper_left = make_cortex_response((6, 6));
        let lower_right = make_cortex_response((22, 22));

        // The patches are in opposite top-stage quadrants — idx 0 = (0,0),
        // idx 3 = (1,1). Retinula's spontaneous activity has spatial biases
        // that can dominate absolute μ̂ ranking, so the meaningful claim is
        // RELATIVE: moving the patch from UL to LR shifts the cortex
        // representation in the right direction. The asymmetry
        //   (μ̂_target − μ̂_anti)
        // must be larger for UL than for LR.
        let ul_asym = upper_left[0] - upper_left[3];
        let lr_asym = lower_right[0] - lower_right[3];
        assert!(
            ul_asym > lr_asym,
            "UL patch should produce more (0,0)-favoring asymmetry than LR: \
             ul_asym={}, lr_asym={}, ul={:?}, lr={:?}",
            ul_asym,
            lr_asym,
            upper_left,
            lower_right,
        );
    }
}
