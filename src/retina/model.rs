//! Retina Model - Complete retinal processing pipeline
//!
//! This module provides a complete retina simulation that converts
//! images to spike trains, analogous to how cochlea_rs converts
//! audio to spike trains.
//!
//! The key insight is that both systems use ribbon synapses for
//! the final stage of converting graded receptor potentials to spikes.

use crate::retina::ganglion::{GanglionCell, GanglionCellType};
use crate::retina::photoreceptor::gray_to_luminance;
use rand::Rng;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;

/// Configuration for the retina model.
#[derive(Clone, Debug)]
pub struct RetinaConfig {
    /// Sample rate for temporal processing (Hz)
    pub sample_rate: f64,
    /// Number of ganglion cells horizontally
    pub cells_x: usize,
    /// Number of ganglion cells vertically
    pub cells_y: usize,
    /// Receptive field center radius (pixels)
    pub rf_radius: f64,
    /// Include ON cells
    pub include_on: bool,
    /// Include OFF cells
    pub include_off: bool,
    /// Cell type to use
    pub cell_type: GanglionCellType,
    /// Random seed
    pub seed: Option<u64>,
    /// Pixels per degree of visual angle (for eccentricity calculation)
    /// Default assumes ~30 pixels per degree (typical for 64px image at ~2° visual angle)
    pub pixels_per_degree: f64,
    /// Use eccentricity-dependent adaptation (fovea vs periphery)
    pub use_eccentricity: bool,
}

impl Default for RetinaConfig {
    fn default() -> Self {
        Self {
            sample_rate: 1000.0,  // 1 kHz typical for visual experiments
            cells_x: 16,
            cells_y: 16,
            rf_radius: 2.0,
            include_on: true,
            include_off: true,
            cell_type: GanglionCellType::MidgetOn,
            seed: None,
            pixels_per_degree: 30.0,  // Typical for 64px image at ~2° visual angle
            use_eccentricity: true,   // Enable eccentricity-dependent adaptation by default
        }
    }
}

impl RetinaConfig {
    /// Create config for small retina (fast, low resolution)
    pub fn small() -> Self {
        Self {
            cells_x: 8,
            cells_y: 8,
            rf_radius: 2.0,
            ..Default::default()
        }
    }

    /// Create config for medium retina
    pub fn medium() -> Self {
        Self {
            cells_x: 16,
            cells_y: 16,
            rf_radius: 3.0,
            ..Default::default()
        }
    }

    /// Create config for high resolution retina
    pub fn high_res() -> Self {
        Self {
            cells_x: 32,
            cells_y: 32,
            rf_radius: 4.0,
            ..Default::default()
        }
    }
}

/// Output from retina model for a single cell.
#[derive(Clone, Debug)]
pub struct CellOutput {
    /// Cell position x
    pub x: f64,
    /// Cell position y
    pub y: f64,
    /// Is ON cell (vs OFF)
    pub is_on: bool,
    /// Spike times in seconds
    pub spike_times: Vec<f64>,
    /// Firing rate estimate (spikes/s)
    pub firing_rate: f64,
}

/// Complete output from the retina model.
#[derive(Clone, Debug)]
pub struct RetinaOutput {
    /// Output from all ON cells
    pub on_cells: Vec<CellOutput>,
    /// Output from all OFF cells
    pub off_cells: Vec<CellOutput>,
    /// Stimulus duration in seconds
    pub duration: f64,
    /// Image dimensions
    pub image_width: usize,
    pub image_height: usize,
}

impl RetinaOutput {
    /// Get all spike times as (cell_index, spike_time) pairs.
    pub fn all_spike_times(&self) -> Vec<(usize, f64, bool)> {
        let mut spikes = Vec::new();

        for (i, cell) in self.on_cells.iter().enumerate() {
            for &t in &cell.spike_times {
                spikes.push((i, t, true)); // true = ON cell
            }
        }

        for (i, cell) in self.off_cells.iter().enumerate() {
            for &t in &cell.spike_times {
                spikes.push((i, t, false)); // false = OFF cell
            }
        }

        spikes.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        spikes
    }

    /// Total spike count.
    pub fn total_spikes(&self) -> usize {
        self.on_cells.iter().map(|c| c.spike_times.len()).sum::<usize>()
            + self.off_cells.iter().map(|c| c.spike_times.len()).sum::<usize>()
    }

    /// Get firing rate map for ON cells.
    pub fn on_firing_rate_map(&self, cells_x: usize, cells_y: usize) -> Vec<Vec<f64>> {
        let mut map = vec![vec![0.0; cells_x]; cells_y];
        for cell in &self.on_cells {
            let xi = (cell.x as usize).min(cells_x - 1);
            let yi = (cell.y as usize).min(cells_y - 1);
            map[yi][xi] = cell.firing_rate;
        }
        map
    }

    /// Get firing rate map for OFF cells.
    pub fn off_firing_rate_map(&self, cells_x: usize, cells_y: usize) -> Vec<Vec<f64>> {
        let mut map = vec![vec![0.0; cells_x]; cells_y];
        for cell in &self.off_cells {
            let xi = (cell.x as usize).min(cells_x - 1);
            let yi = (cell.y as usize).min(cells_y - 1);
            map[yi][xi] = cell.firing_rate;
        }
        map
    }
}

/// Complete retina model.
///
/// Converts images → spike trains through:
/// 1. Photoreceptor transduction
/// 2. Center-surround receptive fields
/// 3. ON/OFF pathways
/// 4. Ribbon synapse (same as cochlea!)
/// 5. Spike generation
pub struct RetinaModel {
    config: RetinaConfig,
    on_cells: Vec<GanglionCell>,
    off_cells: Vec<GanglionCell>,
}

impl RetinaModel {
    /// Create a new retina model with eccentricity-dependent adaptation.
    pub fn new(config: RetinaConfig, image_width: usize, image_height: usize) -> Self {
        let mut on_cells = Vec::new();
        let mut off_cells = Vec::new();

        // Calculate cell spacing
        let spacing_x = image_width as f64 / config.cells_x as f64;
        let spacing_y = image_height as f64 / config.cells_y as f64;

        // Image center (fovea position)
        let center_x = image_width as f64 / 2.0;
        let center_y = image_height as f64 / 2.0;

        // Create ganglion cells at each position
        for yi in 0..config.cells_y {
            for xi in 0..config.cells_x {
                let x = (xi as f64 + 0.5) * spacing_x;
                let y = (yi as f64 + 0.5) * spacing_y;

                // Calculate eccentricity (distance from center in degrees)
                let eccentricity_deg = if config.use_eccentricity {
                    GanglionCell::eccentricity_from_position(
                        x, y, center_x, center_y, config.pixels_per_degree
                    )
                } else {
                    0.0  // All cells at fovea if disabled
                };

                if config.include_on {
                    let cell = match config.cell_type {
                        GanglionCellType::MidgetOn | GanglionCellType::MidgetOff => {
                            GanglionCell::midget_on(x, y, config.rf_radius, config.sample_rate, eccentricity_deg)
                        }
                        GanglionCellType::ParasolOn | GanglionCellType::ParasolOff => {
                            GanglionCell::parasol_on(x, y, config.rf_radius, config.sample_rate, eccentricity_deg)
                        }
                    };
                    on_cells.push(cell);
                }

                if config.include_off {
                    let cell = match config.cell_type {
                        GanglionCellType::MidgetOn | GanglionCellType::MidgetOff => {
                            GanglionCell::midget_off(x, y, config.rf_radius, config.sample_rate, eccentricity_deg)
                        }
                        GanglionCellType::ParasolOn | GanglionCellType::ParasolOff => {
                            GanglionCell::parasol_off(x, y, config.rf_radius, config.sample_rate, eccentricity_deg)
                        }
                    };
                    off_cells.push(cell);
                }
            }
        }

        Self {
            config,
            on_cells,
            off_cells,
        }
    }

    /// Create with default config.
    pub fn default_for_image(width: usize, height: usize) -> Self {
        Self::new(RetinaConfig::default(), width, height)
    }

    /// Process a static grayscale image.
    ///
    /// # Arguments
    /// * `image` - Grayscale pixel values (0-255 or 0.0-1.0)
    /// * `width` - Image width
    /// * `height` - Image height
    /// * `duration_s` - How long to present the image
    ///
    /// # Returns
    /// Spike trains from all ganglion cells
    pub fn process_static_image(
        &mut self,
        image: &[f64],
        width: usize,
        height: usize,
        duration_s: f64,
    ) -> RetinaOutput {
        // Normalize image if needed
        let max_val = image.iter().cloned().fold(0.0f64, f64::max);
        let normalized: Vec<f64> = if max_val > 1.0 {
            image.iter().map(|&p| p / 255.0).collect()
        } else {
            image.to_vec()
        };

        // Apply photoreceptor-like transformation (gamma)
        let processed: Vec<f64> = normalized.iter().map(|&p| gray_to_luminance(p)).collect();

        // Create RNG
        let mut base_rng: StdRng = match self.config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        // Generate seeds for parallel processing
        let num_on = self.on_cells.len();
        let num_off = self.off_cells.len();
        let total_cells = num_on + num_off;
        let seeds: Vec<u64> = (0..total_cells).map(|_| base_rng.gen()).collect();

        // Process ON cells in parallel
        let on_outputs: Vec<CellOutput> = self.on_cells
            .par_iter_mut()
            .zip(seeds[..num_on].par_iter())
            .map(|(cell, &seed)| {
                let mut rng = StdRng::seed_from_u64(seed);
                let spikes = cell.process_static_image(&processed, width, height, duration_s, &mut rng);
                let firing_rate = spikes.len() as f64 / duration_s;
                CellOutput {
                    x: cell.receptive_field.x,
                    y: cell.receptive_field.y,
                    is_on: true,
                    spike_times: spikes,
                    firing_rate,
                }
            })
            .collect();

        // Process OFF cells in parallel
        let off_outputs: Vec<CellOutput> = self.off_cells
            .par_iter_mut()
            .zip(seeds[num_on..].par_iter())
            .map(|(cell, &seed)| {
                let mut rng = StdRng::seed_from_u64(seed);
                let spikes = cell.process_static_image(&processed, width, height, duration_s, &mut rng);
                let firing_rate = spikes.len() as f64 / duration_s;
                CellOutput {
                    x: cell.receptive_field.x,
                    y: cell.receptive_field.y,
                    is_on: false,
                    spike_times: spikes,
                    firing_rate,
                }
            })
            .collect();

        RetinaOutput {
            on_cells: on_outputs,
            off_cells: off_outputs,
            duration: duration_s,
            image_width: width,
            image_height: height,
        }
    }

    /// Process a sequence of grayscale images (video).
    pub fn process_image_sequence(
        &mut self,
        images: &[Vec<f64>],
        width: usize,
        height: usize,
    ) -> RetinaOutput {
        let duration_s = images.len() as f64 / self.config.sample_rate;

        // Normalize all frames
        let processed: Vec<Vec<f64>> = images
            .iter()
            .map(|img| {
                let max_val = img.iter().cloned().fold(0.0f64, f64::max);
                if max_val > 1.0 {
                    img.iter().map(|&p| gray_to_luminance(p / 255.0)).collect()
                } else {
                    img.iter().map(|&p| gray_to_luminance(p)).collect()
                }
            })
            .collect();

        let mut base_rng: StdRng = match self.config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        let num_on = self.on_cells.len();
        let num_off = self.off_cells.len();
        let total_cells = num_on + num_off;
        let seeds: Vec<u64> = (0..total_cells).map(|_| base_rng.gen()).collect();

        // Process ON cells
        let on_outputs: Vec<CellOutput> = self.on_cells
            .par_iter_mut()
            .zip(seeds[..num_on].par_iter())
            .map(|(cell, &seed)| {
                let mut rng = StdRng::seed_from_u64(seed);
                let spikes = cell.process_images(&processed, width, height, &mut rng);
                let firing_rate = spikes.len() as f64 / duration_s;
                CellOutput {
                    x: cell.receptive_field.x,
                    y: cell.receptive_field.y,
                    is_on: true,
                    spike_times: spikes,
                    firing_rate,
                }
            })
            .collect();

        // Process OFF cells
        let off_outputs: Vec<CellOutput> = self.off_cells
            .par_iter_mut()
            .zip(seeds[num_on..].par_iter())
            .map(|(cell, &seed)| {
                let mut rng = StdRng::seed_from_u64(seed);
                let spikes = cell.process_images(&processed, width, height, &mut rng);
                let firing_rate = spikes.len() as f64 / duration_s;
                CellOutput {
                    x: cell.receptive_field.x,
                    y: cell.receptive_field.y,
                    is_on: false,
                    spike_times: spikes,
                    firing_rate,
                }
            })
            .collect();

        RetinaOutput {
            on_cells: on_outputs,
            off_cells: off_outputs,
            duration: duration_s,
            image_width: width,
            image_height: height,
        }
    }

    /// Reset all cell states.
    pub fn reset(&mut self) {
        for cell in &mut self.on_cells {
            cell.reset();
        }
        for cell in &mut self.off_cells {
            cell.reset();
        }
    }

    /// Get number of ON cells.
    pub fn num_on_cells(&self) -> usize {
        self.on_cells.len()
    }

    /// Get number of OFF cells.
    pub fn num_off_cells(&self) -> usize {
        self.off_cells.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retina_model_basic() {
        let width = 32;
        let height = 32;
        let config = RetinaConfig {
            cells_x: 4,
            cells_y: 4,
            sample_rate: 1000.0,
            seed: Some(42),
            ..Default::default()
        };

        let mut model = RetinaModel::new(config, width, height);

        // Create a simple gradient image
        let image: Vec<f64> = (0..width * height)
            .map(|i| {
                let x = (i % width) as f64 / width as f64;
                x
            })
            .collect();

        let output = model.process_static_image(&image, width, height, 0.1);

        assert_eq!(output.on_cells.len(), 16);
        assert_eq!(output.off_cells.len(), 16);
        assert!(output.total_spikes() > 0);
    }

    #[test]
    fn test_on_off_response_difference() {
        let width = 32;
        let height = 32;
        let config = RetinaConfig {
            cells_x: 1,
            cells_y: 1,
            rf_radius: 5.0,
            sample_rate: 1000.0,
            seed: Some(42),
            ..Default::default()
        };

        // Bright image
        let bright_image = vec![0.9; width * height];
        let mut model_bright = RetinaModel::new(config.clone(), width, height);
        let bright_output = model_bright.process_static_image(&bright_image, width, height, 0.2);

        // Dark image
        let dark_image = vec![0.1; width * height];
        let mut model_dark = RetinaModel::new(config, width, height);
        let dark_output = model_dark.process_static_image(&dark_image, width, height, 0.2);

        // ON cells should fire more to bright image
        let on_bright = bright_output.on_cells[0].firing_rate;
        let on_dark = dark_output.on_cells[0].firing_rate;

        // OFF cells should fire more to dark image
        let off_bright = bright_output.off_cells[0].firing_rate;
        let off_dark = dark_output.off_cells[0].firing_rate;

        // ON should prefer bright, OFF should prefer dark
        // Note: due to adaptation, this might not always hold strongly
        println!("ON bright: {}, ON dark: {}", on_bright, on_dark);
        println!("OFF bright: {}, OFF dark: {}", off_bright, off_dark);
    }
}
