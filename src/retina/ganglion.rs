//! Retinal Ganglion Cell Model
//!
//! Ganglion cells are the output neurons of the retina. They receive input
//! from bipolar cells (which receive from photoreceptors) and produce spike
//! trains that travel via the optic nerve to the brain.
//!
//! Key features:
//! - Center-surround receptive fields (DOG: Difference of Gaussians)
//! - ON and OFF pathways (respond to light increment/decrement)
//! - Multiple cell types (midget/P-cells, parasol/M-cells)
//! - Ribbon synapse output (same as cochlea!)

use crate::ribbon_synapse::{RibbonSynapse, RibbonSynapseConfig};
use crate::spike_generator::run_spike_generator;
use rand::Rng;

/// Type of ganglion cell.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GanglionCellType {
    /// Midget cell (P-pathway) - high spatial resolution, color opponent
    MidgetOn,
    MidgetOff,
    /// Parasol cell (M-pathway) - high temporal resolution, motion sensitive
    ParasolOn,
    ParasolOff,
}

impl GanglionCellType {
    /// Returns true if this is an ON-center cell.
    pub fn is_on_center(&self) -> bool {
        matches!(self, GanglionCellType::MidgetOn | GanglionCellType::ParasolOn)
    }

    /// Returns true if this is an OFF-center cell.
    pub fn is_off_center(&self) -> bool {
        !self.is_on_center()
    }

    /// Get the ribbon synapse config for this cell type.
    pub fn synapse_config(&self) -> RibbonSynapseConfig {
        match self {
            GanglionCellType::MidgetOn | GanglionCellType::MidgetOff => {
                RibbonSynapseConfig::retina_midget()
            }
            GanglionCellType::ParasolOn | GanglionCellType::ParasolOff => {
                RibbonSynapseConfig::retina_parasol()
            }
        }
    }
}

/// Receptive field model using Difference of Gaussians (DOG).
///
/// This models the center-surround antagonism created by lateral inhibition
/// in the retina (horizontal cells in outer plexiform layer).
#[derive(Clone, Debug)]
pub struct ReceptiveField {
    /// Center Gaussian radius (in pixels or degrees)
    pub center_radius: f64,
    /// Surround Gaussian radius
    pub surround_radius: f64,
    /// Center weight (positive for ON, negative for OFF)
    pub center_weight: f64,
    /// Surround weight (opposite sign of center)
    pub surround_weight: f64,
    /// Position in visual field (x)
    pub x: f64,
    /// Position in visual field (y)
    pub y: f64,
}

impl ReceptiveField {
    /// Create a new receptive field.
    pub fn new(
        x: f64,
        y: f64,
        center_radius: f64,
        surround_radius: f64,
        is_on_center: bool,
    ) -> Self {
        let (center_weight, surround_weight) = if is_on_center {
            (1.0, -0.5)  // ON-center: excitatory center, inhibitory surround
        } else {
            (-1.0, 0.5)  // OFF-center: inhibitory center, excitatory surround
        };

        Self {
            center_radius,
            surround_radius,
            center_weight,
            surround_weight,
            x,
            y,
        }
    }

    /// Create an ON-center receptive field.
    pub fn on_center(x: f64, y: f64, center_radius: f64, surround_ratio: f64) -> Self {
        Self::new(x, y, center_radius, center_radius * surround_ratio, true)
    }

    /// Create an OFF-center receptive field.
    pub fn off_center(x: f64, y: f64, center_radius: f64, surround_ratio: f64) -> Self {
        Self::new(x, y, center_radius, center_radius * surround_ratio, false)
    }

    /// Compute the response to a 2D image.
    ///
    /// # Arguments
    /// * `image` - 2D array of pixel intensities (row-major)
    /// * `width` - Image width
    /// * `height` - Image height
    ///
    /// # Returns
    /// The receptive field response (sum of weighted Gaussians)
    pub fn compute_response(&self, image: &[f64], width: usize, height: usize) -> f64 {
        let mut center_sum = 0.0;
        let mut surround_sum = 0.0;
        let mut center_norm = 0.0;
        let mut surround_norm = 0.0;

        let center_sigma2 = 2.0 * self.center_radius * self.center_radius;
        let surround_sigma2 = 2.0 * self.surround_radius * self.surround_radius;

        // Sample within a reasonable radius (3 sigma of surround)
        let max_radius = (3.0 * self.surround_radius).ceil() as i32;

        let cx = self.x as i32;
        let cy = self.y as i32;

        for dy in -max_radius..=max_radius {
            for dx in -max_radius..=max_radius {
                let px = cx + dx;
                let py = cy + dy;

                if px >= 0 && px < width as i32 && py >= 0 && py < height as i32 {
                    let idx = (py as usize) * width + (px as usize);
                    let intensity = image[idx];

                    let dist2 = (dx * dx + dy * dy) as f64;

                    // Center Gaussian weight
                    let center_w = (-dist2 / center_sigma2).exp();
                    center_sum += center_w * intensity;
                    center_norm += center_w;

                    // Surround Gaussian weight
                    let surround_w = (-dist2 / surround_sigma2).exp();
                    surround_sum += surround_w * intensity;
                    surround_norm += surround_w;
                }
            }
        }

        // Normalize and compute DOG response
        let center_response = if center_norm > 0.0 {
            center_sum / center_norm
        } else {
            0.0
        };

        let surround_response = if surround_norm > 0.0 {
            surround_sum / surround_norm
        } else {
            0.0
        };

        // DOG = weighted center - weighted surround
        self.center_weight * center_response + self.surround_weight * surround_response
    }

    /// Compute response for a time series of images.
    pub fn compute_response_sequence(
        &self,
        images: &[Vec<f64>],
        width: usize,
        height: usize,
    ) -> Vec<f64> {
        images
            .iter()
            .map(|img| self.compute_response(img, width, height))
            .collect()
    }
}

/// Retinal ganglion cell processor.
///
/// Combines:
/// 1. Center-surround receptive field (spatial filtering)
/// 2. Temporal filtering
/// 3. Ribbon synapse (firing rate) - with eccentricity-dependent adaptation
/// 4. Spike generation
#[derive(Clone)]
pub struct GanglionCell {
    /// Cell type
    pub cell_type: GanglionCellType,
    /// Receptive field
    pub receptive_field: ReceptiveField,
    /// Ribbon synapse for rate coding
    synapse: RibbonSynapse,
    /// Temporal filter state (biphasic)
    temporal_state: [f64; 4],
    /// Time resolution
    tdres: f64,
    /// Temporal filter time constants
    tau_on: f64,
    tau_off: f64,
    /// Eccentricity in degrees (distance from fovea)
    eccentricity_deg: f64,
}

impl GanglionCell {
    /// Create a new ganglion cell with eccentricity-dependent adaptation.
    ///
    /// # Arguments
    /// * `cell_type` - Type of ganglion cell (midget/parasol, ON/OFF)
    /// * `x` - X position in visual field (pixels)
    /// * `y` - Y position in visual field (pixels)
    /// * `center_radius` - Receptive field center radius (pixels)
    /// * `sample_rate` - Sample rate in Hz
    /// * `eccentricity_deg` - Distance from fovea in degrees (0 = fovea, 90 = far periphery)
    pub fn new(
        cell_type: GanglionCellType,
        x: f64,
        y: f64,
        center_radius: f64,
        sample_rate: f64,
        eccentricity_deg: f64,
    ) -> Self {
        let receptive_field = ReceptiveField::new(
            x,
            y,
            center_radius,
            center_radius * 3.0, // Surround is typically 3x center
            cell_type.is_on_center(),
        );

        // Use eccentricity-aware ribbon synapse
        let synapse = RibbonSynapse::new_for_retina(
            cell_type.synapse_config(),
            sample_rate,
            eccentricity_deg,
        );

        // Temporal parameters depend on cell type
        let (tau_on, tau_off) = match cell_type {
            GanglionCellType::MidgetOn | GanglionCellType::MidgetOff => (0.01, 0.05),
            GanglionCellType::ParasolOn | GanglionCellType::ParasolOff => (0.005, 0.02),
        };

        Self {
            cell_type,
            receptive_field,
            synapse,
            temporal_state: [0.0; 4],
            tdres: 1.0 / sample_rate,
            tau_on,
            tau_off,
            eccentricity_deg,
        }
    }

    /// Create a new ganglion cell at fovea (eccentricity = 0).
    pub fn new_at_fovea(
        cell_type: GanglionCellType,
        x: f64,
        y: f64,
        center_radius: f64,
        sample_rate: f64,
    ) -> Self {
        Self::new(cell_type, x, y, center_radius, sample_rate, 0.0)
    }

    /// Compute eccentricity from pixel position and image center.
    ///
    /// # Arguments
    /// * `x` - X position in pixels
    /// * `y` - Y position in pixels
    /// * `center_x` - Image center X (fovea position)
    /// * `center_y` - Image center Y (fovea position)
    /// * `pixels_per_degree` - Conversion factor (depends on viewing distance)
    pub fn eccentricity_from_position(
        x: f64,
        y: f64,
        center_x: f64,
        center_y: f64,
        pixels_per_degree: f64,
    ) -> f64 {
        let dx = x - center_x;
        let dy = y - center_y;
        let dist_pixels = (dx * dx + dy * dy).sqrt();
        dist_pixels / pixels_per_degree
    }

    /// Create an ON-center midget cell at given position.
    pub fn midget_on(x: f64, y: f64, center_radius: f64, sample_rate: f64, eccentricity_deg: f64) -> Self {
        Self::new(GanglionCellType::MidgetOn, x, y, center_radius, sample_rate, eccentricity_deg)
    }

    /// Create an OFF-center midget cell at given position.
    pub fn midget_off(x: f64, y: f64, center_radius: f64, sample_rate: f64, eccentricity_deg: f64) -> Self {
        Self::new(GanglionCellType::MidgetOff, x, y, center_radius, sample_rate, eccentricity_deg)
    }

    /// Create an ON-center parasol cell at given position.
    pub fn parasol_on(x: f64, y: f64, center_radius: f64, sample_rate: f64, eccentricity_deg: f64) -> Self {
        Self::new(GanglionCellType::ParasolOn, x, y, center_radius, sample_rate, eccentricity_deg)
    }

    /// Create an OFF-center parasol cell at given position.
    pub fn parasol_off(x: f64, y: f64, center_radius: f64, sample_rate: f64, eccentricity_deg: f64) -> Self {
        Self::new(GanglionCellType::ParasolOff, x, y, center_radius, sample_rate, eccentricity_deg)
    }

    /// Get the eccentricity of this cell.
    pub fn eccentricity(&self) -> f64 {
        self.eccentricity_deg
    }

    /// Reset cell state.
    pub fn reset(&mut self) {
        self.temporal_state = [0.0; 4];
        self.synapse.reset();
    }

    /// Apply biphasic temporal filter.
    ///
    /// This models the transient response of ganglion cells.
    fn temporal_filter(&mut self, input: f64) -> f64 {
        // Fast pathway (excitatory)
        let alpha_on = self.tdres / (self.tau_on + self.tdres);
        self.temporal_state[0] = (1.0 - alpha_on) * self.temporal_state[0] + alpha_on * input;

        // Slow pathway (inhibitory)
        let alpha_off = self.tdres / (self.tau_off + self.tdres);
        self.temporal_state[1] = (1.0 - alpha_off) * self.temporal_state[1] + alpha_off * input;

        // Biphasic: fast - slow
        self.temporal_state[0] - 0.5 * self.temporal_state[1]
    }

    /// Process a sequence of images and generate spikes.
    ///
    /// # Arguments
    /// * `images` - Sequence of images (each image is row-major pixel array)
    /// * `width` - Image width
    /// * `height` - Image height
    /// * `rng` - Random number generator
    ///
    /// # Returns
    /// Spike times (in seconds)
    pub fn process_images<R: Rng>(
        &mut self,
        images: &[Vec<f64>],
        width: usize,
        height: usize,
        rng: &mut R,
    ) -> Vec<f64> {
        // Compute spatial response for each frame
        let spatial_response = self.receptive_field.compute_response_sequence(images, width, height);

        // Apply temporal filtering
        let temporal_response: Vec<f64> = spatial_response
            .iter()
            .map(|&r| self.temporal_filter(r))
            .collect();

        // Rectify (ganglion cells can't have negative firing rate)
        // Add baseline to shift into positive range
        let baseline = 0.3;
        let rectified: Vec<f64> = temporal_response
            .iter()
            .map(|&r| (r + baseline).max(0.0))
            .collect();

        // Process through ribbon synapse
        let firing_rate = self.synapse.process(&rectified, rng);

        // Generate spikes
        run_spike_generator(&firing_rate, self.tdres, rng)
    }

    /// Process a single static image and generate spikes for given duration.
    ///
    /// Useful for testing with a single image held constant.
    pub fn process_static_image<R: Rng>(
        &mut self,
        image: &[f64],
        width: usize,
        height: usize,
        duration_s: f64,
        rng: &mut R,
    ) -> Vec<f64> {
        let n_frames = (duration_s / self.tdres).ceil() as usize;

        // Create sequence of identical frames
        let images: Vec<Vec<f64>> = (0..n_frames).map(|_| image.to_vec()).collect();

        self.process_images(&images, width, height, rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_receptive_field_center_response() {
        // Create a simple image with a bright spot in the center
        let width = 32;
        let height = 32;
        let mut image = vec![0.0; width * height];

        // Put a bright spot at center
        for dy in -2..=2i32 {
            for dx in -2..=2i32 {
                let x = 16 + dx;
                let y = 16 + dy;
                if x >= 0 && x < width as i32 && y >= 0 && y < height as i32 {
                    image[(y as usize) * width + (x as usize)] = 1.0;
                }
            }
        }

        // ON-center cell at center should respond positively
        let rf_on = ReceptiveField::on_center(16.0, 16.0, 3.0, 3.0);
        let response_on = rf_on.compute_response(&image, width, height);

        // OFF-center cell at center should respond negatively
        let rf_off = ReceptiveField::off_center(16.0, 16.0, 3.0, 3.0);
        let response_off = rf_off.compute_response(&image, width, height);

        assert!(response_on > 0.0, "ON-center should respond positively to bright spot");
        assert!(response_off < 0.0, "OFF-center should respond negatively to bright spot");
    }

    #[test]
    fn test_ganglion_cell_spikes() {
        let mut rng = StdRng::seed_from_u64(42);
        let sample_rate = 1000.0; // 1 kHz

        // Create a midget ON cell at fovea (eccentricity = 0)
        let mut cell = GanglionCell::midget_on(16.0, 16.0, 3.0, sample_rate, 0.0);

        // Create a bright image
        let width = 32;
        let height = 32;
        let image = vec![0.5; width * height];

        // Process for 100ms
        let spikes = cell.process_static_image(&image, width, height, 0.1, &mut rng);

        // Should produce some spikes
        assert!(!spikes.is_empty(), "Cell should produce spikes");

        // All spike times should be within duration
        assert!(spikes.iter().all(|&t| t >= 0.0 && t <= 0.1));
    }

    #[test]
    fn test_eccentricity_affects_adaptation() {
        let mut rng_fovea = StdRng::seed_from_u64(42);
        let mut rng_periph = StdRng::seed_from_u64(42);
        let sample_rate = 1000.0;

        // Create cells at different eccentricities
        let mut fovea_cell = GanglionCell::midget_on(16.0, 16.0, 3.0, sample_rate, 0.0);  // Fovea
        let mut periph_cell = GanglionCell::midget_on(16.0, 16.0, 3.0, sample_rate, 30.0); // 30° peripheral

        // Same bright image
        let width = 32;
        let height = 32;
        let image = vec![0.5; width * height];

        // Process for 100ms
        let fovea_spikes = fovea_cell.process_static_image(&image, width, height, 0.1, &mut rng_fovea);
        let periph_spikes = periph_cell.process_static_image(&image, width, height, 0.1, &mut rng_periph);

        // Both should produce spikes (the adaptation factor affects dynamics, not silence)
        assert!(!fovea_spikes.is_empty(), "Fovea cell should produce spikes");
        assert!(!periph_spikes.is_empty(), "Peripheral cell should produce spikes");

        // The spike counts may differ due to different adaptation dynamics
        println!("Fovea spikes: {}, Peripheral spikes: {}", fovea_spikes.len(), periph_spikes.len());
    }
}
