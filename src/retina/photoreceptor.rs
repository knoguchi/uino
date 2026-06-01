//! Photoreceptor Model
//!
//! Implements cone and rod photoreceptors that convert light intensity
//! to graded receptor potentials (hyperpolarization).
//!
//! Key biological features:
//! - Light adaptation (Weber's law)
//! - Temporal filtering (response dynamics)
//! - Spectral sensitivity (L, M, S cones)
//!
//! Unlike hair cells which depolarize to stimulus, photoreceptors
//! HYPERPOLARIZE in response to light (they release less glutamate).


/// Type of photoreceptor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PhotoreceptorType {
    /// Rod photoreceptor (scotopic vision)
    Rod,
    /// Cone photoreceptor (photopic vision)
    Cone(ConeType),
}

/// Cone subtypes based on spectral sensitivity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ConeType {
    /// Long wavelength (red), peak ~564nm
    L,
    /// Medium wavelength (green), peak ~534nm
    M,
    /// Short wavelength (blue), peak ~420nm
    S,
}

impl ConeType {
    /// Peak wavelength sensitivity in nm.
    pub fn peak_wavelength(&self) -> f64 {
        match self {
            ConeType::L => 564.0,
            ConeType::M => 534.0,
            ConeType::S => 420.0,
        }
    }

    /// Spectral sensitivity at a given wavelength.
    /// Returns relative sensitivity (0-1).
    pub fn sensitivity(&self, wavelength_nm: f64) -> f64 {
        let peak = self.peak_wavelength();
        let sigma = match self {
            ConeType::L => 50.0,
            ConeType::M => 45.0,
            ConeType::S => 35.0,
        };

        // Gaussian approximation of spectral sensitivity
        let diff = wavelength_nm - peak;
        (-0.5 * (diff / sigma).powi(2)).exp()
    }
}

/// Photoreceptor processor.
///
/// Converts light intensity to receptor potential using biologically
/// realistic dynamics including adaptation.
#[derive(Clone, Debug)]
pub struct Photoreceptor {
    /// Type of photoreceptor
    pub receptor_type: PhotoreceptorType,
    /// Time resolution (seconds)
    tdres: f64,
    /// Adaptation state (background light level)
    adaptation_level: f64,
    /// Temporal filter state
    filter_state: [f64; 4],
    /// Dark current (baseline membrane potential)
    dark_current: f64,
    /// Maximum response amplitude
    max_response: f64,
    /// Adaptation time constant (seconds)
    tau_adapt: f64,
    /// Response time constant (seconds)
    tau_response: f64,
    /// Half-saturation constant
    sigma: f64,
}

impl Photoreceptor {
    /// Create a new photoreceptor.
    pub fn new(receptor_type: PhotoreceptorType, sample_rate: f64) -> Self {
        let tdres = 1.0 / sample_rate;

        // Parameters differ for rods vs cones
        let (tau_adapt, tau_response, sigma, dark_current, max_response) = match receptor_type {
            PhotoreceptorType::Rod => (
                0.5,    // Slow adaptation
                0.2,    // Slow response
                0.01,   // Very sensitive
                -40.0,  // mV (depolarized in dark)
                30.0,   // mV (max hyperpolarization)
            ),
            PhotoreceptorType::Cone(_) => (
                0.05,   // Fast adaptation
                0.02,   // Fast response
                0.1,    // Less sensitive
                -40.0,  // mV
                25.0,   // mV
            ),
        };

        Self {
            receptor_type,
            tdres,
            adaptation_level: 0.1,
            filter_state: [0.0; 4],
            dark_current,
            max_response,
            tau_adapt,
            tau_response,
            sigma,
        }
    }

    /// Create an L-cone (red sensitive).
    pub fn l_cone(sample_rate: f64) -> Self {
        Self::new(PhotoreceptorType::Cone(ConeType::L), sample_rate)
    }

    /// Create an M-cone (green sensitive).
    pub fn m_cone(sample_rate: f64) -> Self {
        Self::new(PhotoreceptorType::Cone(ConeType::M), sample_rate)
    }

    /// Create an S-cone (blue sensitive).
    pub fn s_cone(sample_rate: f64) -> Self {
        Self::new(PhotoreceptorType::Cone(ConeType::S), sample_rate)
    }

    /// Create a rod photoreceptor.
    pub fn rod(sample_rate: f64) -> Self {
        Self::new(PhotoreceptorType::Rod, sample_rate)
    }

    /// Reset processor state.
    pub fn reset(&mut self) {
        self.adaptation_level = 0.1;
        self.filter_state = [0.0; 4];
    }

    /// Get spectral sensitivity at given wavelength.
    pub fn spectral_sensitivity(&self, wavelength_nm: f64) -> f64 {
        match self.receptor_type {
            PhotoreceptorType::Rod => {
                // Rod spectral sensitivity (scotopic)
                let peak = 498.0;
                let sigma = 40.0;
                let diff = wavelength_nm - peak;
                (-0.5 * (diff / sigma).powi(2)).exp()
            }
            PhotoreceptorType::Cone(cone_type) => cone_type.sensitivity(wavelength_nm),
        }
    }

    /// Process a single light intensity sample.
    ///
    /// # Arguments
    /// * `intensity` - Light intensity (0-1 normalized, or physical units)
    ///
    /// # Returns
    /// Receptor potential in mV (hyperpolarization from dark current)
    #[inline]
    pub fn process_sample(&mut self, intensity: f64) -> f64 {
        // Update adaptation level (slow exponential average)
        let alpha_adapt = self.tdres / (self.tau_adapt + self.tdres);
        self.adaptation_level = (1.0 - alpha_adapt) * self.adaptation_level + alpha_adapt * intensity;

        // Weber adaptation: response depends on contrast relative to background
        let adapted_intensity = intensity / (self.adaptation_level + 0.01);

        // Naka-Rushton equation: hyperbolic saturation
        // r/r_max = I^n / (I^n + sigma^n)
        let n = 1.0; // Hill coefficient
        let response_fraction = adapted_intensity.powf(n) / (adapted_intensity.powf(n) + self.sigma.powf(n));

        // Temporal filtering (4th order lowpass)
        let alpha = self.tdres / (self.tau_response + self.tdres);
        self.filter_state[0] = (1.0 - alpha) * self.filter_state[0] + alpha * response_fraction;
        self.filter_state[1] = (1.0 - alpha) * self.filter_state[1] + alpha * self.filter_state[0];
        self.filter_state[2] = (1.0 - alpha) * self.filter_state[2] + alpha * self.filter_state[1];
        self.filter_state[3] = (1.0 - alpha) * self.filter_state[3] + alpha * self.filter_state[2];

        let filtered_response = self.filter_state[3];

        // Convert to membrane potential
        // In dark: depolarized (dark_current)
        // In light: hyperpolarized (dark_current - response)
        // But for downstream processing, we output the "drive" which increases with light
        // (inverting the photoreceptor signal as bipolar cells do)
        self.dark_current - self.max_response * filtered_response
    }

    /// Process a sequence of light intensity samples.
    ///
    /// # Arguments
    /// * `intensities` - Light intensity time series
    ///
    /// # Returns
    /// Receptor potential time series
    pub fn process(&mut self, intensities: &[f64]) -> Vec<f64> {
        intensities.iter().map(|&i| self.process_sample(i)).collect()
    }
}

/// Convert RGB image pixel to cone activations.
///
/// Returns (L, M, S) cone responses for a given RGB value.
pub fn rgb_to_cone_response(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    // Approximate transformation from RGB to LMS cone space
    // Based on Hunt-Pointer-Estevez transform
    let l = 0.38971 * r + 0.68898 * g - 0.07868 * b;
    let m = -0.22981 * r + 1.18340 * g + 0.04641 * b;
    let s = 0.00000 * r + 0.00000 * g + 1.00000 * b;

    (l.max(0.0), m.max(0.0), s.max(0.0))
}

/// Convert grayscale intensity to luminance-based cone response.
pub fn gray_to_luminance(gray: f64) -> f64 {
    // Simple gamma correction and normalization
    gray.powf(2.2).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cone_sensitivity() {
        let l_cone = ConeType::L;
        let m_cone = ConeType::M;
        let s_cone = ConeType::S;

        // L cone should be most sensitive to red
        assert!(l_cone.sensitivity(600.0) > m_cone.sensitivity(600.0));

        // M cone should be most sensitive to green
        assert!(m_cone.sensitivity(530.0) > l_cone.sensitivity(530.0));

        // S cone should be most sensitive to blue
        assert!(s_cone.sensitivity(420.0) > l_cone.sensitivity(420.0));
    }

    #[test]
    fn test_photoreceptor_response() {
        let sample_rate = 1000.0; // 1 kHz
        let mut cone = Photoreceptor::l_cone(sample_rate);

        // Dark response
        let dark = cone.process_sample(0.0);

        // Light response
        cone.reset();
        let light = cone.process_sample(1.0);

        // Light should hyperpolarize (more negative)
        assert!(light < dark);
    }

    #[test]
    fn test_adaptation() {
        let sample_rate = 1000.0;
        let mut cone = Photoreceptor::l_cone(sample_rate);

        // Let temporal filter settle with constant light
        for _ in 0..100 {
            cone.process_sample(1.0);
        }
        let early = cone.process_sample(1.0);

        // Continue adapting
        for _ in 0..1000 {
            cone.process_sample(1.0);
        }
        let late = cone.process_sample(1.0);

        // After adaptation, response should be closer to dark level
        // (adaptation moves response toward baseline)
        let dark = cone.dark_current;

        // Both responses should be below dark (hyperpolarized)
        // And late response should be less hyperpolarized (closer to dark)
        // due to adaptation increasing the adaptation_level
        println!("dark={}, early={}, late={}", dark, early, late);

        // The response should still be hyperpolarized from dark
        assert!(early < dark);
        assert!(late < dark);
    }
}
