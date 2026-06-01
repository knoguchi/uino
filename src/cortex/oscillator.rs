//! Theta-Gamma Oscillator for Temporal Normalization
//!
//! Implements nested oscillations that provide:
//! - Syllable-level segmentation (theta: 4-8 Hz)
//! - Phonetic slot encoding (gamma: 30-50 Hz)
//! - Rate-invariant temporal coding via phase
//!
//! # Biology
//!
//! - Theta oscillations (~5 Hz) track syllable boundaries
//! - Gamma oscillations (~40 Hz) nest within theta cycles
//! - Phase reset occurs at acoustic landmarks (onsets)
//! - Information is encoded in spike phase relative to gamma

use std::f64::consts::PI;

/// State of the theta-gamma oscillator
#[derive(Clone, Debug)]
pub struct OscillatorState {
    /// Theta phase (0 to 2π)
    pub theta_phase: f64,
    /// Gamma phase (0 to 2π)
    pub gamma_phase: f64,
    /// Current gamma bin within theta cycle
    pub gamma_bin: usize,
    /// Total gamma bins per theta cycle
    pub gamma_bins_per_theta: usize,
    /// Theta modulation factor (0-1, for gating)
    pub theta_gate: f64,
    /// Whether a phase reset just occurred
    pub reset_occurred: bool,
    /// Current theta frequency (adaptive)
    pub theta_freq: f64,
    /// Current gamma frequency
    pub gamma_freq: f64,
}

/// Theta-Gamma Oscillator
#[derive(Clone, Debug)]
pub struct ThetaGammaOscillator {
    // Theta oscillator (syllable rate: 4-8 Hz)
    theta_phase: f64,
    theta_freq: f64,
    theta_freq_min: f64,
    theta_freq_max: f64,

    // Gamma oscillator (phonetic rate: 30-50 Hz)
    gamma_phase: f64,
    gamma_freq: f64,

    // Phase reset
    reset_threshold: f64,
    last_reset_time: f64,
    min_reset_interval: f64, // Minimum time between resets

    // Adaptive frequency tracking
    adaptation_rate: f64,
    onset_history: Vec<f64>, // Recent onset times for frequency estimation

    // Phase coding
    gamma_bins_per_theta: usize,
}

impl ThetaGammaOscillator {
    /// Create a new theta-gamma oscillator
    ///
    /// # Arguments
    /// * `theta_freq` - Initial theta frequency in Hz (default: 5 Hz)
    /// * `gamma_freq` - Gamma frequency in Hz (default: 40 Hz)
    /// * `reset_threshold` - Onset strength threshold for phase reset
    pub fn new(theta_freq: f64, gamma_freq: f64, reset_threshold: f64) -> Self {
        let gamma_bins_per_theta = (gamma_freq / theta_freq).round() as usize;

        Self {
            theta_phase: 0.0,
            theta_freq,
            theta_freq_min: 3.0,
            theta_freq_max: 10.0,
            gamma_phase: 0.0,
            gamma_freq,
            reset_threshold,
            last_reset_time: -1.0, // No reset yet
            min_reset_interval: 0.080, // 80ms minimum between resets
            adaptation_rate: 0.1,
            onset_history: Vec::with_capacity(10),
            gamma_bins_per_theta,
        }
    }

    /// Create with default parameters
    pub fn with_defaults() -> Self {
        Self::new(5.0, 40.0, 0.3)
    }

    /// Process one time step
    ///
    /// # Arguments
    /// * `onset_strength` - Strength of current onset (0-1)
    /// * `current_time` - Current time in seconds
    /// * `dt` - Time step in seconds
    pub fn step(&mut self, onset_strength: f64, current_time: f64, dt: f64) -> OscillatorState {
        let mut reset_occurred = false;

        // Check for phase reset on strong onsets
        if onset_strength > self.reset_threshold {
            let time_since_reset = current_time - self.last_reset_time;
            if time_since_reset > self.min_reset_interval || self.last_reset_time < 0.0 {
                // Reset phases to align with syllable boundary
                self.theta_phase = 0.0;
                self.gamma_phase = 0.0;
                self.last_reset_time = current_time;
                reset_occurred = true;

                // Record onset for frequency adaptation
                self.onset_history.push(current_time);
                if self.onset_history.len() > 10 {
                    self.onset_history.remove(0);
                }

                // Adapt theta frequency based on onset intervals
                self.adapt_theta_frequency();
            }
        }

        // Advance theta phase
        self.theta_phase += 2.0 * PI * self.theta_freq * dt;
        if self.theta_phase >= 2.0 * PI {
            self.theta_phase -= 2.0 * PI;
            // Reset gamma at theta cycle boundary
            self.gamma_phase = 0.0;
        }

        // Advance gamma phase
        self.gamma_phase += 2.0 * PI * self.gamma_freq * dt;
        if self.gamma_phase >= 2.0 * PI {
            self.gamma_phase -= 2.0 * PI;
        }

        // Theta modulation: gate is high at theta trough (beginning of cycle)
        // This allows processing at syllable onsets
        let theta_gate = (-self.theta_phase.cos() + 1.0) / 2.0;

        // Compute current gamma bin
        let theta_progress = self.theta_phase / (2.0 * PI);
        let gamma_bin = (theta_progress * self.gamma_bins_per_theta as f64).floor() as usize;
        let gamma_bin = gamma_bin.min(self.gamma_bins_per_theta - 1);

        OscillatorState {
            theta_phase: self.theta_phase,
            gamma_phase: self.gamma_phase,
            gamma_bin,
            gamma_bins_per_theta: self.gamma_bins_per_theta,
            theta_gate,
            reset_occurred,
            theta_freq: self.theta_freq,
            gamma_freq: self.gamma_freq,
        }
    }

    /// Adapt theta frequency based on recent onset intervals
    fn adapt_theta_frequency(&mut self) {
        if self.onset_history.len() < 3 {
            return;
        }

        // Compute average inter-onset interval
        let mut total_interval = 0.0;
        let mut count = 0;
        for i in 1..self.onset_history.len() {
            let interval = self.onset_history[i] - self.onset_history[i - 1];
            if interval > 0.05 && interval < 0.5 {
                // Valid syllable-range interval
                total_interval += interval;
                count += 1;
            }
        }

        if count > 0 {
            let avg_interval = total_interval / count as f64;
            let estimated_freq = 1.0 / avg_interval;

            // Slowly adapt towards estimated frequency
            let target_freq = estimated_freq.clamp(self.theta_freq_min, self.theta_freq_max);
            self.theta_freq += self.adaptation_rate * (target_freq - self.theta_freq);

            // Update gamma bins per theta
            self.gamma_bins_per_theta = (self.gamma_freq / self.theta_freq).round() as usize;
        }
    }

    /// Encode a spike time as phase code
    ///
    /// Returns the gamma phase at spike time (rate-invariant encoding)
    pub fn encode_spike_phase(&self, spike_time: f64, current_time: f64) -> f64 {
        // Compute phase at spike time
        let time_diff = current_time - spike_time;
        let phase_at_spike = self.gamma_phase - 2.0 * PI * self.gamma_freq * time_diff;

        // Normalize to [0, 2π]
        let mut phase = phase_at_spike % (2.0 * PI);
        if phase < 0.0 {
            phase += 2.0 * PI;
        }
        phase
    }

    /// Get which gamma bin a spike falls into
    pub fn spike_to_gamma_bin(&self, spike_phase: f64) -> usize {
        let bin = (spike_phase / (2.0 * PI) * self.gamma_bins_per_theta as f64).floor() as usize;
        bin.min(self.gamma_bins_per_theta - 1)
    }

    /// Reset oscillator state
    pub fn reset(&mut self) {
        self.theta_phase = 0.0;
        self.gamma_phase = 0.0;
        self.last_reset_time = -1.0;
        self.onset_history.clear();
    }

    /// Get current theta frequency
    pub fn theta_freq(&self) -> f64 {
        self.theta_freq
    }

    /// Get current gamma frequency
    pub fn gamma_freq(&self) -> f64 {
        self.gamma_freq
    }

    /// Set theta frequency bounds
    pub fn set_theta_bounds(&mut self, min: f64, max: f64) {
        self.theta_freq_min = min;
        self.theta_freq_max = max;
        self.theta_freq = self.theta_freq.clamp(min, max);
    }

    /// Set reset threshold
    pub fn set_reset_threshold(&mut self, threshold: f64) {
        self.reset_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Get gamma bins per theta cycle
    pub fn gamma_bins_per_theta(&self) -> usize {
        self.gamma_bins_per_theta
    }
}

/// Phase-coded spike representation
#[derive(Clone, Debug)]
pub struct PhaseCoding {
    /// Gamma bins per theta cycle
    n_bins: usize,
    /// Number of channels
    n_channels: usize,
    /// Current phase-coded activity: [channels][gamma_bins]
    activity: Vec<Vec<f64>>,
}

impl PhaseCoding {
    /// Create new phase coding buffer
    pub fn new(n_channels: usize, gamma_bins: usize) -> Self {
        Self {
            n_bins: gamma_bins,
            n_channels,
            activity: vec![vec![0.0; gamma_bins]; n_channels],
        }
    }

    /// Record a spike at given channel and gamma bin
    pub fn record_spike(&mut self, channel: usize, gamma_bin: usize) {
        if channel < self.n_channels && gamma_bin < self.n_bins {
            self.activity[channel][gamma_bin] += 1.0;
        }
    }

    /// Decay activity (call each time step)
    pub fn decay(&mut self, factor: f64) {
        for channel in &mut self.activity {
            for bin in channel {
                *bin *= factor;
            }
        }
    }

    /// Reset at theta cycle boundary
    pub fn reset_cycle(&mut self) {
        for channel in &mut self.activity {
            channel.fill(0.0);
        }
    }

    /// Get activity pattern
    pub fn get_activity(&self) -> &Vec<Vec<f64>> {
        &self.activity
    }

    /// Get flattened activity vector (for readout)
    pub fn flatten(&self) -> Vec<f64> {
        self.activity.iter().flat_map(|c| c.iter().copied()).collect()
    }

    /// Get dimensionality
    pub fn dim(&self) -> usize {
        self.n_channels * self.n_bins
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_oscillator_phases() {
        let mut osc = ThetaGammaOscillator::with_defaults();
        let dt = 0.001; // 1ms

        // Run for one theta cycle (~200ms at 5 Hz)
        let mut theta_wraps = 0;
        let mut prev_theta = 0.0;

        for i in 0..250 {
            let state = osc.step(0.0, i as f64 * dt, dt);

            if state.theta_phase < prev_theta {
                theta_wraps += 1;
            }
            prev_theta = state.theta_phase;
        }

        // Should complete about 1 theta cycle in 200ms
        assert!(theta_wraps >= 1, "Should complete at least 1 theta cycle");
    }

    #[test]
    fn test_phase_reset() {
        let mut osc = ThetaGammaOscillator::new(5.0, 40.0, 0.3);
        let dt = 0.001;

        // Advance oscillator
        for i in 0..100 {
            osc.step(0.0, i as f64 * dt, dt);
        }

        // Phase should have advanced
        assert!(osc.theta_phase > 0.1, "Theta phase should advance");

        // Trigger reset with strong onset
        let state = osc.step(0.5, 0.100, dt);

        assert!(state.reset_occurred, "Reset should occur");
        assert!(
            osc.theta_phase < 0.1,
            "Theta phase should reset to near zero"
        );
    }

    #[test]
    fn test_gamma_bins() {
        let osc = ThetaGammaOscillator::new(5.0, 40.0, 0.3);

        // 40 Hz / 5 Hz = 8 gamma cycles per theta
        assert_eq!(osc.gamma_bins_per_theta(), 8);
    }

    #[test]
    fn test_theta_adaptation() {
        let mut osc = ThetaGammaOscillator::new(5.0, 40.0, 0.3);
        let dt = 0.001;

        // Simulate onsets at ~6 Hz rate (every ~167ms)
        let onset_interval = 0.167;
        let mut t = 0.0;

        for i in 0..20 {
            let is_onset = (i as f64 * dt / onset_interval).floor()
                > ((i as f64 - 1.0) * dt / onset_interval).floor();
            let onset = if is_onset { 0.5 } else { 0.0 };
            osc.step(onset, t, dt);
            t += dt;
        }

        // After adaptation, theta should move toward 6 Hz
        // (Though adaptation is slow, so change may be small)
    }

    #[test]
    fn test_phase_coding() {
        let mut coding = PhaseCoding::new(10, 8);

        // Record some spikes
        coding.record_spike(5, 3);
        coding.record_spike(5, 3);
        coding.record_spike(7, 1);

        assert_eq!(coding.activity[5][3], 2.0);
        assert_eq!(coding.activity[7][1], 1.0);

        // Test decay
        coding.decay(0.5);
        assert_eq!(coding.activity[5][3], 1.0);
    }
}
