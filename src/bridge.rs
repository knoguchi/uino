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
//! Output shape matches the retinula grid (cells_x × cells_y) row-major.
//! [`Channel`] selects which retina population the bridge bins from: ON
//! cells only, OFF cells only, or both concatenated (ON cells first, then
//! OFF cells — doubles the input dimension).

use retinula::RetinaOutput;

/// Which retina population(s) the bridge exposes to the cortex.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Channel {
    /// ON cells only. Output length = cells_x * cells_y.
    On,
    /// OFF cells only. Output length = cells_x * cells_y.
    Off,
    /// Both: ON cells first, then OFF cells, concatenated.
    /// Output length = 2 * cells_x * cells_y.
    OnOff,
}

/// Cortex-aligned view of retinula output.
pub struct RetinaBridge {
    cells_x: usize,
    cells_y: usize,
    channel: Channel,
    /// Per-cell firing rate over the whole presentation (Hz). Length matches `channel`.
    mean_rates_hz: Vec<f64>,
    /// `smoothed[step][cell]` = exponentially-smoothed spike trace. Length matches `channel`.
    smoothed: Vec<Vec<f64>>,
    /// Raw spike counts. Same shape as `smoothed`.
    counts: Vec<Vec<usize>>,
    dt_s: f64,
}

impl RetinaBridge {
    /// `dt_ms`: cortex timestep. `tau_ms`: smoothing time constant
    /// (5 ms ≈ AMPA EPSC). `channel`: which retina population(s) to expose.
    pub fn from_retina_output(
        output: &RetinaOutput,
        cells_x: usize,
        cells_y: usize,
        dt_ms: f64,
        tau_ms: f64,
        channel: Channel,
    ) -> Self {
        assert!(dt_ms > 0.0 && tau_ms > 0.0);
        let n_grid = cells_x * cells_y;
        match channel {
            Channel::On => assert_eq!(output.on_cells.len(), n_grid),
            Channel::Off => assert_eq!(output.off_cells.len(), n_grid),
            Channel::OnOff => {
                assert_eq!(output.on_cells.len(), n_grid);
                assert_eq!(output.off_cells.len(), n_grid);
            }
        }
        let dt_s = dt_ms * 1e-3;
        let n_steps = (output.duration / dt_s).floor() as usize;
        let out_len = match channel {
            Channel::On | Channel::Off => n_grid,
            Channel::OnOff => 2 * n_grid,
        };

        let mut counts: Vec<Vec<usize>> = (0..n_steps).map(|_| vec![0usize; out_len]).collect();

        let bin_into = |target: &mut [Vec<usize>], cells: &[retinula::CellOutput], offset: usize| {
            for (cell_idx, cell) in cells.iter().enumerate() {
                for &t in &cell.spike_times {
                    if t < 0.0 || t >= output.duration {
                        continue;
                    }
                    let step = (t / dt_s).floor() as usize;
                    if step < target.len() {
                        target[step][offset + cell_idx] += 1;
                    }
                }
            }
        };
        match channel {
            Channel::On => bin_into(&mut counts, &output.on_cells, 0),
            Channel::Off => bin_into(&mut counts, &output.off_cells, 0),
            Channel::OnOff => {
                bin_into(&mut counts, &output.on_cells, 0);
                bin_into(&mut counts, &output.off_cells, n_grid);
            }
        }

        let alpha = (-dt_ms / tau_ms).exp();
        let mut trace = vec![0.0f64; out_len];
        let mut smoothed: Vec<Vec<f64>> = Vec::with_capacity(n_steps);
        for step in 0..n_steps {
            for c in 0..out_len {
                trace[c] = alpha * trace[c] + counts[step][c] as f64;
            }
            smoothed.push(trace.clone());
        }

        let mean_rates_hz: Vec<f64> = match channel {
            Channel::On => output.on_cells.iter().map(|c| c.firing_rate).collect(),
            Channel::Off => output.off_cells.iter().map(|c| c.firing_rate).collect(),
            Channel::OnOff => output
                .on_cells
                .iter()
                .map(|c| c.firing_rate)
                .chain(output.off_cells.iter().map(|c| c.firing_rate))
                .collect(),
        };

        Self { cells_x, cells_y, channel, mean_rates_hz, smoothed, counts, dt_s }
    }

    pub fn cells_x(&self) -> usize {
        self.cells_x
    }
    pub fn cells_y(&self) -> usize {
        self.cells_y
    }
    pub fn channel(&self) -> Channel {
        self.channel
    }
    /// Length of each per-step input vector (`cells_x*cells_y` for single
    /// channel, doubled for `OnOff`).
    pub fn input_len(&self) -> usize {
        self.smoothed.first().map(|v| v.len()).unwrap_or(0)
    }
    pub fn n_steps(&self) -> usize {
        self.smoothed.len()
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
        &self.counts[step]
    }

    /// Per-step smoothed rate trace, scaled. Right form for dynamic input.
    pub fn rates_at(&self, step: usize, scale: f64) -> Vec<f64> {
        self.smoothed[step].iter().map(|&r| r * scale).collect()
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

        let bridge = RetinaBridge::from_retina_output(&output, cells, cells, 0.1, 5.0, Channel::On);

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
            let bridge = RetinaBridge::from_retina_output(&output, cells, cells, 0.1, 5.0, Channel::On);
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

    /// ON+OFF channel produces an input vector of length 2*cells*cells.
    /// A bright patch on a dark background drives ON cells preferentially;
    /// a dark patch on a bright background drives OFF cells. The bridge
    /// must route these to distinct halves of the output vector.
    #[test]
    fn on_off_channel_carries_polarity_information() {
        let img_w = 16;
        let img_h = 16;
        let cells = 4;
        let mut retina = Retina::for_image(img_w, img_h)
            .resolution(cells, cells)
            .no_eccentricity()
            .seed(42);

        // Both images sit on a gray background; only the central patch's
        // polarity differs. Same overall luminance — isolates the local
        // contrast effect that ON/OFF cells are designed to detect.
        let gray = 0.5;
        let mut bright_on_gray = vec![gray; img_w * img_h];
        let mut dark_on_gray = vec![gray; img_w * img_h];
        for dy in 4..9 {
            for dx in 4..9 {
                bright_on_gray[dy * img_w + dx] = 1.0;
                dark_on_gray[dy * img_w + dx] = 0.0;
            }
        }

        let out_bright = retina.simulate(&bright_on_gray, 0.2);
        let bridge_bright = RetinaBridge::from_retina_output(
            &out_bright, cells, cells, 0.1, 5.0, Channel::OnOff,
        );
        assert_eq!(bridge_bright.input_len(), 2 * cells * cells);
        let rates_bright = bridge_bright.mean_rates(1.0);
        let on_sum_bright: f64 = rates_bright[..cells * cells].iter().sum();
        let off_sum_bright: f64 = rates_bright[cells * cells..].iter().sum();

        retina.reset();
        let out_dark = retina.simulate(&dark_on_gray, 0.2);
        let bridge_dark = RetinaBridge::from_retina_output(
            &out_dark, cells, cells, 0.1, 5.0, Channel::OnOff,
        );
        let rates_dark = bridge_dark.mean_rates(1.0);
        let on_sum_dark: f64 = rates_dark[..cells * cells].iter().sum();
        let off_sum_dark: f64 = rates_dark[cells * cells..].iter().sum();

        // Bright-on-dark should drive ON cells MORE than dark-on-bright does;
        // dark-on-bright should drive OFF cells MORE than bright-on-dark.
        assert!(
            on_sum_bright > on_sum_dark,
            "ON should respond more to bright patch: bright={}, dark={}",
            on_sum_bright,
            on_sum_dark
        );
        assert!(
            off_sum_dark > off_sum_bright,
            "OFF should respond more to dark patch: bright={}, dark={}",
            off_sum_bright,
            off_sum_dark
        );
    }

    /// Position-invariance via spatial pooling: patches at multiple positions
    /// within the same image quadrant produce cortex top-stage signatures
    /// that cluster together, well separated from signatures of patches in
    /// the opposite quadrant. This is the within-receptive-field invariance
    /// the pooling architecture provides — not learned invariance, but the
    /// architectural baseline that learned invariance would build on.
    #[test]
    fn within_quadrant_position_invariance() {
        use crate::metrics::separability;

        fn top_signature_for_patch(patch_origin: (usize, usize)) -> Vec<f64> {
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
                    let x = patch_origin.0 + dx;
                    let y = patch_origin.1 + dy;
                    if x < img_w && y < img_h {
                        img[y * img_w + x] = 1.0;
                    }
                }
            }
            let output = retina.simulate(&img, 0.3);
            let bridge = RetinaBridge::from_retina_output(&output, cells, cells, 0.1, 5.0, Channel::On);
            let s = bridge.mean_rates(1.0 / 30.0);

            let mut cortex = MultiStage::with_defaults(&[(8, 8), (4, 4), (2, 2)], &[2, 2]);
            for _ in 0..20_000 {
                cortex.step(&s, 0.1);
            }
            cortex.stages[2].predictions()
        }

        // Class UL: patches at varied positions within the upper-left quadrant.
        // Class LR: same for lower-right.
        let ul_origins = [(2, 2), (4, 4), (6, 6), (3, 7), (7, 3)];
        let lr_origins = [(18, 18), (20, 20), (22, 22), (19, 23), (23, 19)];

        let class_ul: Vec<Vec<f64>> = ul_origins.iter().map(|&p| top_signature_for_patch(p)).collect();
        let class_lr: Vec<Vec<f64>> = lr_origins.iter().map(|&p| top_signature_for_patch(p)).collect();

        let sep = separability(&class_ul, &class_lr).expect("separability");
        assert!(
            sep.index > 1.0,
            "expected position-invariant class separability > 1, got {} (radius {}, centroid dist {})",
            sep.index,
            sep.mean_radius,
            sep.centroid_distance,
        );
    }
}
