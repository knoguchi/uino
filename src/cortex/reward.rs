//! Reward Module for Gated Plasticity
//!
//! Computes global reward/error signals that gate synaptic plasticity.
//! Supports both:
//! - Supervised: reward based on correct/incorrect classification
//! - Unsupervised: reward based on prediction error magnitude
//!
//! # Biology
//!
//! - Neuromodulators (dopamine, acetylcholine) gate plasticity
//! - Reward prediction error drives learning
//! - Eligibility traces allow delayed credit assignment

/// Reward signal with eligibility trace
#[derive(Clone, Debug)]
pub struct RewardSignal {
    /// Current reward value
    pub value: f64,
    /// Eligibility trace (decaying memory of recent activity)
    pub eligibility: f64,
    /// Modulated reward (value * eligibility)
    pub modulated: f64,
}

/// Reward computation mode
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RewardMode {
    /// Supervised: +1 for correct, -1 for incorrect
    Supervised,
    /// Unsupervised: negative prediction error
    Unsupervised,
    /// Mixed: combines both signals
    Mixed { supervised_weight: f64 },
}

/// Reward module for computing learning signals
#[derive(Clone, Debug)]
pub struct RewardModule {
    /// Current reward value
    reward: f64,
    /// Eligibility trace
    eligibility: f64,
    /// Eligibility time constant (seconds)
    eligibility_tau: f64,
    /// Reward mode
    mode: RewardMode,
    /// Reward baseline (for computing prediction error)
    baseline: f64,
    /// Baseline adaptation rate
    baseline_rate: f64,
    /// Positive reward magnitude
    positive_reward: f64,
    /// Negative reward magnitude (typically smaller)
    negative_reward: f64,
    /// Minimum time between reward updates
    min_interval: f64,
    /// Time since last reward
    time_since_reward: f64,
}

impl RewardModule {
    /// Create a new reward module
    ///
    /// # Arguments
    /// * `mode` - Reward computation mode
    /// * `eligibility_tau` - Eligibility trace time constant (default: 100-200ms)
    pub fn new(mode: RewardMode, eligibility_tau: f64) -> Self {
        Self {
            reward: 0.0,
            eligibility: 0.0,
            eligibility_tau,
            mode,
            baseline: 0.0,
            baseline_rate: 0.01,
            positive_reward: 1.0,
            negative_reward: 0.5, // Asymmetric: negative weaker
            min_interval: 0.050,  // 50ms minimum
            time_since_reward: 1.0,
        }
    }

    /// Create with default parameters for supervised learning
    pub fn supervised() -> Self {
        Self::new(RewardMode::Supervised, 0.150)
    }

    /// Create with default parameters for unsupervised learning
    pub fn unsupervised() -> Self {
        Self::new(RewardMode::Unsupervised, 0.200)
    }

    /// Update eligibility trace (call every time step)
    pub fn update_eligibility(&mut self, activity_level: f64, dt: f64) {
        // Decay existing eligibility
        self.eligibility *= (-dt / self.eligibility_tau).exp();

        // Accumulate new activity
        self.eligibility += activity_level * (1.0 - (-dt / self.eligibility_tau).exp());

        // Update time since reward
        self.time_since_reward += dt;
    }

    /// Compute supervised reward signal
    ///
    /// # Arguments
    /// * `predicted` - Predicted class index
    /// * `correct` - Correct class index
    pub fn compute_supervised(&mut self, predicted: usize, correct: usize) -> f64 {
        if self.time_since_reward < self.min_interval {
            return 0.0;
        }

        self.time_since_reward = 0.0;

        if predicted == correct {
            self.reward = self.positive_reward;
        } else {
            self.reward = -self.negative_reward;
        }

        self.reward
    }

    /// Compute unsupervised reward signal based on prediction error
    ///
    /// # Arguments
    /// * `prediction_error` - L2 norm of prediction error
    pub fn compute_unsupervised(&mut self, prediction_error: f64) -> f64 {
        // Update baseline (moving average of error)
        self.baseline += self.baseline_rate * (prediction_error - self.baseline);

        // Reward = negative of error relative to baseline
        // Low error (good prediction) = positive reward
        // High error (bad prediction) = negative reward
        let relative_error = prediction_error - self.baseline;
        self.reward = -relative_error.tanh();

        self.reward
    }

    /// Compute reward based on current mode
    pub fn compute(
        &mut self,
        predicted: Option<usize>,
        correct: Option<usize>,
        prediction_error: Option<f64>,
    ) -> f64 {
        match self.mode {
            RewardMode::Supervised => {
                if let (Some(pred), Some(corr)) = (predicted, correct) {
                    self.compute_supervised(pred, corr)
                } else {
                    0.0
                }
            }
            RewardMode::Unsupervised => {
                if let Some(error) = prediction_error {
                    self.compute_unsupervised(error)
                } else {
                    0.0
                }
            }
            RewardMode::Mixed { supervised_weight } => {
                let sup = if let (Some(pred), Some(corr)) = (predicted, correct) {
                    self.compute_supervised(pred, corr)
                } else {
                    0.0
                };
                let unsup = if let Some(error) = prediction_error {
                    self.compute_unsupervised(error)
                } else {
                    0.0
                };
                self.reward = supervised_weight * sup + (1.0 - supervised_weight) * unsup;
                self.reward
            }
        }
    }

    /// Get current reward signal
    pub fn get_reward(&self) -> f64 {
        self.reward
    }

    /// Get modulated reward (reward * eligibility)
    pub fn get_modulated_reward(&self) -> f64 {
        self.reward * self.eligibility
    }

    /// Get full reward signal
    pub fn get_signal(&self) -> RewardSignal {
        RewardSignal {
            value: self.reward,
            eligibility: self.eligibility,
            modulated: self.reward * self.eligibility,
        }
    }

    /// Reset reward to zero (call after learning update)
    pub fn clear_reward(&mut self) {
        self.reward = 0.0;
    }

    /// Reset all state
    pub fn reset(&mut self) {
        self.reward = 0.0;
        self.eligibility = 0.0;
        self.time_since_reward = 1.0;
    }

    /// Set reward magnitudes
    pub fn set_magnitudes(&mut self, positive: f64, negative: f64) {
        self.positive_reward = positive;
        self.negative_reward = negative;
    }

    /// Set eligibility time constant
    pub fn set_eligibility_tau(&mut self, tau: f64) {
        self.eligibility_tau = tau;
    }

    /// Get current mode
    pub fn mode(&self) -> RewardMode {
        self.mode
    }

    /// Set mode
    pub fn set_mode(&mut self, mode: RewardMode) {
        self.mode = mode;
    }
}

/// Accumulator for computing prediction error across the network
#[derive(Clone, Debug, Default)]
pub struct PredictionErrorAccumulator {
    /// Sum of squared errors
    sum_squared: f64,
    /// Number of samples
    count: usize,
}

impl PredictionErrorAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add prediction error sample
    pub fn add(&mut self, predicted: &[f64], actual: &[f64]) {
        assert_eq!(predicted.len(), actual.len());

        let error: f64 = predicted
            .iter()
            .zip(actual.iter())
            .map(|(p, a)| (p - a).powi(2))
            .sum();

        self.sum_squared += error;
        self.count += predicted.len();
    }

    /// Get RMS prediction error
    pub fn rms_error(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            (self.sum_squared / self.count as f64).sqrt()
        }
    }

    /// Get mean squared error
    pub fn mse(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum_squared / self.count as f64
        }
    }

    /// Reset accumulator
    pub fn reset(&mut self) {
        self.sum_squared = 0.0;
        self.count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_supervised_reward() {
        let mut reward = RewardModule::supervised();

        // Correct prediction
        let r = reward.compute_supervised(5, 5);
        assert!(r > 0.0, "Correct should give positive reward");

        // Wait for interval
        reward.time_since_reward = 1.0;

        // Incorrect prediction
        let r = reward.compute_supervised(3, 5);
        assert!(r < 0.0, "Incorrect should give negative reward");
    }

    #[test]
    fn test_unsupervised_reward() {
        let mut reward = RewardModule::unsupervised();

        // Establish baseline with moderate errors
        for _ in 0..100 {
            reward.compute_unsupervised(0.5);
        }

        // Low error should give positive reward
        let r = reward.compute_unsupervised(0.1);
        assert!(r > 0.0, "Low error should give positive reward: {}", r);

        // High error should give negative reward
        let r = reward.compute_unsupervised(0.9);
        assert!(r < 0.0, "High error should give negative reward: {}", r);
    }

    #[test]
    fn test_eligibility_trace() {
        let mut reward = RewardModule::supervised();
        let dt = 0.001;

        // Build up eligibility
        for _ in 0..100 {
            reward.update_eligibility(1.0, dt);
        }

        let e1 = reward.eligibility;
        assert!(e1 > 0.0, "Eligibility should accumulate");

        // Let it decay
        for _ in 0..200 {
            reward.update_eligibility(0.0, dt);
        }

        let e2 = reward.eligibility;
        assert!(e2 < e1, "Eligibility should decay");
    }

    #[test]
    fn test_prediction_error_accumulator() {
        let mut acc = PredictionErrorAccumulator::new();

        let predicted = vec![1.0, 2.0, 3.0];
        let actual = vec![1.1, 2.0, 2.9];

        acc.add(&predicted, &actual);

        let rms = acc.rms_error();
        assert!(rms > 0.0 && rms < 0.1, "RMS should be small: {}", rms);
    }
}
