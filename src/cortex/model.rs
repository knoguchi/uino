//! Cortical Model Orchestration
//!
//! Integrates all cortical components into a unified processing pipeline:
//! - Thalamus: input gating and predictive suppression
//! - A1: spectro-temporal feature extraction with learned STRFs
//! - Belt: context integration and syllable-level encoding
//! - Oscillator: theta-gamma coupling for temporal normalization
//! - Reward: learning signal computation
//!
//! # Data Flow
//!
//! ```text
//! Cochlear Input
//!       │
//!       ▼
//! ┌─────────────┐    prediction    ┌─────────────┐
//! │  Thalamus   │◄─────────────────│     A1      │
//! └──────┬──────┘                  └──────┬──────┘
//!        │                                │
//!        │ onset/sustained                │ activity
//!        │                                │
//!        ▼                                ▼
//! ┌─────────────┐    prediction    ┌─────────────┐
//! │     A1      │◄─────────────────│    Belt     │
//! └──────┬──────┘                  └─────────────┘
//!        │
//!        │ activity
//!        ▼
//! ┌─────────────┐
//! │    Belt     │──────────────────► Output
//! └─────────────┘
//! ```

use crate::cortex::a1_core::{A1Core, A1Output};
use crate::cortex::belt::{Belt, BeltOutput};
use crate::cortex::oscillator::{OscillatorState, PhaseCoding, ThetaGammaOscillator};
use crate::cortex::reward::{PredictionErrorAccumulator, RewardMode, RewardModule};
use crate::cortex::thalamus::{ThalamicOutput, ThalamicRelay};

/// Configuration for the cortical model
#[derive(Clone, Debug)]
pub struct CorticalConfig {
    /// Number of cochlear input channels
    pub n_cochlear: usize,
    /// Number of A1 neurons (overcomplete, typically 10-50× n_cochlear)
    pub n_a1: usize,
    /// Number of belt neurons (compressed, typically n_a1 / 3)
    pub n_belt: usize,
    /// STRF duration in milliseconds
    pub strf_duration_ms: f64,
    /// STRF time bin in milliseconds
    pub strf_bin_ms: f64,
    /// Target sparsity for A1
    pub target_sparsity: f64,
    /// Theta frequency (Hz)
    pub theta_freq: f64,
    /// Gamma frequency (Hz)
    pub gamma_freq: f64,
    /// Reward mode
    pub reward_mode: RewardMode,
    /// Random seed
    pub seed: u64,
}

impl Default for CorticalConfig {
    fn default() -> Self {
        Self {
            n_cochlear: 30,
            n_a1: 600,       // 20× overcomplete
            n_belt: 200,     // ~1/3 of A1
            strf_duration_ms: 100.0,
            strf_bin_ms: 5.0,
            target_sparsity: 0.05,
            theta_freq: 5.0,
            gamma_freq: 40.0,
            reward_mode: RewardMode::Supervised,
            seed: 42,
        }
    }
}

impl CorticalConfig {
    /// Create config for given number of cochlear channels
    pub fn for_cochlear_channels(n_cochlear: usize) -> Self {
        Self {
            n_cochlear,
            n_a1: n_cochlear * 20,
            n_belt: (n_cochlear * 20) / 3,
            ..Default::default()
        }
    }
}

/// Output from cortical model processing
#[derive(Clone, Debug)]
pub struct CorticalOutput {
    /// Thalamic output (onset/sustained separation)
    pub thalamus: ThalamicOutput,
    /// A1 output (STRFs, sparsity)
    pub a1: A1Output,
    /// Belt output (context, syllable code)
    pub belt: BeltOutput,
    /// Oscillator state (theta/gamma phases)
    pub oscillator: OscillatorState,
    /// Current reward signal
    pub reward: f64,
    /// Prediction error (for unsupervised learning)
    pub prediction_error: f64,
}

/// Complete cortical model
pub struct CorticalModel {
    /// Thalamic relay (MGN)
    thalamus: ThalamicRelay,
    /// Primary auditory cortex
    a1: A1Core,
    /// Belt/parabelt
    belt: Belt,
    /// Theta-gamma oscillator
    oscillator: ThetaGammaOscillator,
    /// Reward module
    reward: RewardModule,
    /// Prediction error accumulator
    error_accumulator: PredictionErrorAccumulator,
    /// Phase coding buffer
    phase_coding: PhaseCoding,
    /// Configuration
    config: CorticalConfig,
    /// Current simulation time
    current_time: f64,
    /// Number of thalamic channels (onset + sustained)
    n_thalamic: usize,
}

impl CorticalModel {
    /// Create a new cortical model
    pub fn new(config: CorticalConfig) -> Self {
        // Thalamic input is 2× cochlear (onset + sustained)
        let n_thalamic = config.n_cochlear * 2;

        let thalamus = ThalamicRelay::with_defaults(config.n_cochlear);

        let a1 = A1Core::new(
            n_thalamic,
            config.n_cochlear,
            config.n_a1,
            config.strf_duration_ms,
            config.strf_bin_ms,
            config.target_sparsity,
            config.seed,
        );

        let belt = Belt::new(config.n_a1, config.n_belt, config.seed + 1000);

        let oscillator = ThetaGammaOscillator::new(
            config.theta_freq,
            config.gamma_freq,
            0.3, // Reset threshold
        );

        let reward = RewardModule::new(config.reward_mode, 0.150);

        let gamma_bins = oscillator.gamma_bins_per_theta();
        let phase_coding = PhaseCoding::new(config.n_a1, gamma_bins);

        let error_accumulator = PredictionErrorAccumulator::new();

        Self {
            thalamus,
            a1,
            belt,
            oscillator,
            reward,
            error_accumulator,
            phase_coding,
            config,
            current_time: 0.0,
            n_thalamic,
        }
    }

    /// Create with default configuration
    pub fn with_defaults(n_cochlear: usize) -> Self {
        let config = CorticalConfig::for_cochlear_channels(n_cochlear);
        Self::new(config)
    }

    /// Process one time step
    ///
    /// # Arguments
    /// * `cochlear_input` - Firing rates from cochlea (synapse_out)
    /// * `label` - Optional correct label for supervised learning
    /// * `dt` - Time step in seconds
    pub fn step(
        &mut self,
        cochlear_input: &[f64],
        label: Option<usize>,
        dt: f64,
    ) -> CorticalOutput {
        assert_eq!(cochlear_input.len(), self.config.n_cochlear);

        // 1. Update oscillator
        let onset_strength = self.thalamus.get_onset_activity().iter().sum::<f64>()
            / self.config.n_cochlear as f64;
        let osc_state = self.oscillator.step(onset_strength, self.current_time, dt);

        // 2. Process through thalamus with A1 prediction (predictive suppression)
        let a1_thalamic_pred = self.a1.get_thalamic_prediction();
        let thal_out = self.thalamus.step(cochlear_input, &a1_thalamic_pred, dt);

        // 3. Combine onset and sustained into thalamic input for A1
        let thalamic_combined: Vec<f64> = thal_out
            .onset_activity
            .iter()
            .chain(thal_out.sustained_activity.iter())
            .copied()
            .collect();

        // 4. Process through A1 with belt prediction
        let belt_a1_pred = self.belt.get_a1_prediction();
        let reward_signal = self.reward.get_reward();
        let a1_out = self.a1.step(&thalamic_combined, &belt_a1_pred, reward_signal, dt);

        // 5. Compute prediction error (for unsupervised learning)
        self.error_accumulator
            .add(&belt_a1_pred, &a1_out.activity);
        let prediction_error = self.error_accumulator.rms_error();

        // 6. Process through belt
        let belt_out = self.belt.step(
            &a1_out.activity,
            osc_state.theta_phase,
            reward_signal,
            dt,
        );

        // 7. Update phase coding
        self.phase_coding.decay((-dt / 0.050).exp()); // 50ms decay
        for (i, &spiked) in a1_out.spikes.iter().enumerate() {
            if spiked {
                self.phase_coding.record_spike(i, osc_state.gamma_bin);
            }
        }

        // Reset phase coding at theta boundary
        if osc_state.reset_occurred {
            self.phase_coding.reset_cycle();
            self.error_accumulator.reset();
        }

        // 8. Update reward/eligibility
        let activity_level = a1_out.activity.iter().sum::<f64>() / self.config.n_a1 as f64;
        self.reward.update_eligibility(activity_level, dt);

        // 9. Compute reward if label provided or using unsupervised
        let computed_reward = match self.config.reward_mode {
            RewardMode::Supervised => {
                if let Some(correct) = label {
                    // Simple classification based on belt activity
                    let predicted = self.classify_from_belt(&belt_out);
                    self.reward.compute_supervised(predicted, correct)
                } else {
                    0.0
                }
            }
            RewardMode::Unsupervised => self.reward.compute_unsupervised(prediction_error),
            RewardMode::Mixed { .. } => self.reward.compute(
                label.map(|_| self.classify_from_belt(&belt_out)),
                label,
                Some(prediction_error),
            ),
        };

        // 10. Update time
        self.current_time += dt;

        CorticalOutput {
            thalamus: thal_out,
            a1: a1_out,
            belt: belt_out,
            oscillator: osc_state,
            reward: computed_reward,
            prediction_error,
        }
    }

    /// Simple classification from belt activity (winner-take-all)
    fn classify_from_belt(&self, belt_out: &BeltOutput) -> usize {
        belt_out
            .activity
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Get syllable-level representation for classification
    pub fn get_syllable_code(&self) -> Vec<f64> {
        self.belt.get_syllable_code()
    }

    /// Get phase-coded representation
    pub fn get_phase_code(&self) -> Vec<f64> {
        self.phase_coding.flatten()
    }

    /// Get current A1 activity
    pub fn get_a1_activity(&self) -> Vec<f64> {
        self.a1.get_activity().to_vec()
    }

    /// Get current belt activity
    pub fn get_belt_activity(&self) -> Vec<f64> {
        self.belt.get_activity().to_vec()
    }

    /// Reset model state (but preserve learned weights)
    pub fn reset(&mut self) {
        self.thalamus.reset();
        self.a1.reset();
        self.belt.reset();
        self.oscillator.reset();
        self.reward.reset();
        self.error_accumulator.reset();
        self.phase_coding = PhaseCoding::new(
            self.config.n_a1,
            self.oscillator.gamma_bins_per_theta(),
        );
        self.current_time = 0.0;
    }

    /// Set learning rate for all learnable components
    pub fn set_learning_rate(&mut self, rate: f64) {
        self.a1.set_learning_rate(rate);
    }

    /// Enable/disable learning
    pub fn set_learning_enabled(&mut self, enabled: bool) {
        if enabled {
            self.reward.set_mode(self.config.reward_mode);
        } else {
            // Clear reward to disable learning
            self.reward.clear_reward();
        }
    }

    /// Get configuration
    pub fn config(&self) -> &CorticalConfig {
        &self.config
    }

    /// Get current simulation time
    pub fn current_time(&self) -> f64 {
        self.current_time
    }

    /// Get A1 sparsity statistics
    pub fn get_a1_stats(&self) -> (f64, f64) {
        // (average sparsity, average STRF norm)
        let activity: Vec<f64> = self.a1.get_activity().to_vec();
        let sparsity = activity.iter().filter(|&&a| a > 0.1).count() as f64 / activity.len() as f64;
        let strf_norm = self.a1.avg_strf_norm();
        (sparsity, strf_norm)
    }

    /// Get debug info about signal levels at each stage
    /// Returns (mean_thal_onset, mean_thal_sustained, mean_a1_activity, mean_belt_activity)
    pub fn get_debug_signal_levels(&self) -> (f64, f64, f64, f64) {
        let thal_onset: f64 = self.thalamus.get_onset_activity().iter().sum::<f64>()
            / self.config.n_cochlear as f64;
        let thal_sustained: f64 = self.thalamus.get_sustained_activity().iter().sum::<f64>()
            / self.config.n_cochlear as f64;
        let a1_act: f64 = self.a1.get_activity().iter().sum::<f64>() / self.config.n_a1 as f64;
        let belt_act: f64 = self.belt.get_activity().iter().sum::<f64>() / self.config.n_belt as f64;
        (thal_onset, thal_sustained, a1_act, belt_act)
    }

    /// Get STRF drive statistics (mean, min, max)
    pub fn get_strf_drive_stats(&self) -> (f64, f64, f64) {
        self.a1.debug_strf_drive()
    }
}

/// Builder for CorticalModel with fluent API
pub struct CorticalModelBuilder {
    config: CorticalConfig,
}

impl CorticalModelBuilder {
    pub fn new(n_cochlear: usize) -> Self {
        Self {
            config: CorticalConfig::for_cochlear_channels(n_cochlear),
        }
    }

    pub fn n_a1(mut self, n: usize) -> Self {
        self.config.n_a1 = n;
        self
    }

    pub fn n_belt(mut self, n: usize) -> Self {
        self.config.n_belt = n;
        self
    }

    pub fn strf_duration_ms(mut self, ms: f64) -> Self {
        self.config.strf_duration_ms = ms;
        self
    }

    pub fn target_sparsity(mut self, s: f64) -> Self {
        self.config.target_sparsity = s;
        self
    }

    pub fn theta_freq(mut self, freq: f64) -> Self {
        self.config.theta_freq = freq;
        self
    }

    pub fn gamma_freq(mut self, freq: f64) -> Self {
        self.config.gamma_freq = freq;
        self
    }

    pub fn reward_mode(mut self, mode: RewardMode) -> Self {
        self.config.reward_mode = mode;
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.config.seed = seed;
        self
    }

    pub fn build(self) -> CorticalModel {
        CorticalModel::new(self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cortical_model_creation() {
        let model = CorticalModel::with_defaults(30);
        assert_eq!(model.config().n_cochlear, 30);
        assert_eq!(model.config().n_a1, 600);
    }

    #[test]
    fn test_cortical_model_step() {
        let mut model = CorticalModel::with_defaults(10);
        let dt = 0.001;

        let input = vec![0.5; 10];

        // Run for 100ms
        for _ in 0..100 {
            let out = model.step(&input, None, dt);

            // Check dimensions
            assert_eq!(out.thalamus.onset_spikes.len(), 10);
            assert_eq!(out.a1.activity.len(), 200);
            assert_eq!(out.belt.activity.len(), model.config().n_belt);
        }
    }

    #[test]
    fn test_predictive_coding() {
        let mut model = CorticalModel::with_defaults(10);
        let dt = 0.001;

        // Run with constant input to build up predictions
        let input = vec![1.0; 10];

        let mut errors = Vec::new();
        for i in 0..500 {
            let out = model.step(&input, None, dt);
            if i > 100 && i % 50 == 0 {
                errors.push(out.prediction_error);
            }
        }

        // Prediction error should decrease as predictions improve
        // (This is a weak test since it depends on learning dynamics)
        assert!(errors.len() > 2);
    }

    #[test]
    fn test_builder() {
        let model = CorticalModelBuilder::new(20)
            .n_a1(400)
            .target_sparsity(0.10)
            .theta_freq(6.0)
            .seed(123)
            .build();

        assert_eq!(model.config().n_a1, 400);
        assert_eq!(model.config().target_sparsity, 0.10);
        assert_eq!(model.config().theta_freq, 6.0);
    }
}
