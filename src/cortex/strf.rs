//! Spectro-Temporal Receptive Fields (STRFs)
//!
//! STRFs are the learned filters that A1 neurons use to detect patterns
//! in the thalamic input. They integrate over both frequency (spectral)
//! and time (temporal) dimensions.
//!
//! # Biology
//!
//! - Real A1 neurons have STRFs spanning 50-200ms
//! - Asymmetric in time: strong onset response, weak offset
//! - Learned through experience via Hebbian/STDP mechanisms
//! - Different neurons specialize for different spectro-temporal patterns

use std::collections::VecDeque;

/// Spectro-Temporal Receptive Field
#[derive(Clone, Debug)]
pub struct STRF {
    /// Weights: [time_bins][channels]
    /// Organized as time-major for efficient convolution
    weights: Vec<Vec<f64>>,
    /// Number of frequency channels (from thalamus)
    n_channels: usize,
    /// Number of time bins in the STRF
    n_time_bins: usize,
    /// Duration of each time bin in seconds
    bin_duration: f64,
    /// Total STRF duration in seconds
    total_duration: f64,
    /// Eligibility trace for learning: [time_bins][channels]
    eligibility: Vec<Vec<f64>>,
    /// Eligibility trace time constant (seconds)
    eligibility_tau: f64,
    /// Learning rate
    learning_rate: f64,
    /// Weight decay (L2 regularization)
    weight_decay: f64,
    /// Maximum absolute weight (for stability)
    max_weight: f64,
}

impl STRF {
    /// Create a new STRF with random initialization
    ///
    /// # Arguments
    /// * `n_channels` - Number of frequency channels
    /// * `duration_ms` - Total STRF duration in milliseconds
    /// * `bin_size_ms` - Size of each time bin in milliseconds
    /// * `seed` - Random seed for initialization
    pub fn new(n_channels: usize, duration_ms: f64, bin_size_ms: f64, seed: u64) -> Self {
        let n_time_bins = (duration_ms / bin_size_ms).ceil() as usize;
        let bin_duration = bin_size_ms / 1000.0;
        let total_duration = duration_ms / 1000.0;

        // Initialize weights with small positive random values
        // Non-negative weights ensure positive activation for any input
        let mut rng_state = seed;
        let mut rand = || {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            (rng_state as f64) / (u64::MAX as f64) // [0, 1] range
        };

        let weights: Vec<Vec<f64>> = (0..n_time_bins)
            .map(|t| {
                // Time-dependent scaling: more recent = stronger
                let time_scale = ((n_time_bins - t) as f64 / n_time_bins as f64).sqrt();
                (0..n_channels)
                    .map(|_| {
                        // Sparse initialization: only some weights are non-zero
                        if rand() < 0.2 {
                            rand() * 0.3 * time_scale // [0, 0.3] for non-zero weights
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect();

        let eligibility = vec![vec![0.0; n_channels]; n_time_bins];

        Self {
            weights,
            n_channels,
            n_time_bins,
            bin_duration,
            total_duration,
            eligibility,
            eligibility_tau: 0.100, // 100ms eligibility trace
            learning_rate: 0.001,
            weight_decay: 0.0001,
            max_weight: 2.0,
        }
    }

    /// Create STRF with specific initialization pattern
    pub fn with_pattern(
        n_channels: usize,
        duration_ms: f64,
        bin_size_ms: f64,
        center_channel: usize,
        bandwidth: usize,
    ) -> Self {
        let n_time_bins = (duration_ms / bin_size_ms).ceil() as usize;
        let bin_duration = bin_size_ms / 1000.0;
        let total_duration = duration_ms / 1000.0;

        // Initialize with Gabor-like pattern centered on a frequency
        // Stronger weights to ensure reliable activation
        let weights: Vec<Vec<f64>> = (0..n_time_bins)
            .map(|t| {
                let t_norm = t as f64 / n_time_bins as f64;
                // Temporal envelope: onset-weighted, peak near recent times
                let temporal = (-(t_norm - 0.7).powi(2) / 0.15).exp();

                (0..n_channels)
                    .map(|c| {
                        let c_dist = (c as i32 - center_channel as i32).abs() as f64;
                        // Spectral envelope: Gaussian around center
                        let spectral = (-c_dist.powi(2) / (2.0 * bandwidth as f64).powi(2)).exp();
                        temporal * spectral * 1.0 // Increased from 0.5 to 1.0
                    })
                    .collect()
            })
            .collect();

        let eligibility = vec![vec![0.0; n_channels]; n_time_bins];

        Self {
            weights,
            n_channels,
            n_time_bins,
            bin_duration,
            total_duration,
            eligibility,
            eligibility_tau: 0.100,
            learning_rate: 0.001,
            weight_decay: 0.0001,
            max_weight: 2.0,
        }
    }

    /// Compute STRF response to input history
    ///
    /// # Arguments
    /// * `history` - Input history buffer [time_bins][channels], oldest first
    ///
    /// # Returns
    /// Scalar activation value
    pub fn compute(&self, history: &[Vec<f64>]) -> f64 {
        assert!(history.len() >= self.n_time_bins);

        let mut activation = 0.0;
        let offset = history.len() - self.n_time_bins;

        for t in 0..self.n_time_bins {
            for c in 0..self.n_channels {
                activation += self.weights[t][c] * history[offset + t][c];
            }
        }

        activation
    }

    /// Compute response and update eligibility trace
    ///
    /// # Arguments
    /// * `history` - Input history buffer
    /// * `dt` - Time step in seconds
    ///
    /// # Returns
    /// Scalar activation value
    pub fn compute_with_eligibility(&mut self, history: &[Vec<f64>], dt: f64) -> f64 {
        assert!(history.len() >= self.n_time_bins);

        let mut activation = 0.0;
        let offset = history.len() - self.n_time_bins;

        // Decay eligibility trace
        let decay = (-dt / self.eligibility_tau).exp();

        for t in 0..self.n_time_bins {
            for c in 0..self.n_channels {
                // Decay existing eligibility
                self.eligibility[t][c] *= decay;

                // Accumulate activation
                let input = history[offset + t][c];
                activation += self.weights[t][c] * input;

                // Update eligibility: how much this synapse contributed
                self.eligibility[t][c] += input * (1.0 - decay);
            }
        }

        activation
    }

    /// Update weights based on reward signal (reward-gated Hebbian)
    ///
    /// # Arguments
    /// * `reward` - Global reward signal (positive = correct, negative = incorrect)
    /// * `post_activity` - Post-synaptic neuron activity (0-1)
    pub fn update_weights(&mut self, reward: f64, post_activity: f64) {
        if reward.abs() < 0.001 {
            return; // No update for near-zero reward
        }

        for t in 0..self.n_time_bins {
            for c in 0..self.n_channels {
                // Three-factor learning rule:
                // dw = learning_rate * reward * eligibility * post_activity
                let dw = self.learning_rate * reward * self.eligibility[t][c] * post_activity;

                // Apply weight update with decay
                self.weights[t][c] += dw - self.weight_decay * self.weights[t][c];

                // Clamp weights
                self.weights[t][c] = self.weights[t][c].clamp(-self.max_weight, self.max_weight);
            }
        }
    }

    /// Reset eligibility traces
    pub fn reset_eligibility(&mut self) {
        for t in 0..self.n_time_bins {
            for c in 0..self.n_channels {
                self.eligibility[t][c] = 0.0;
            }
        }
    }

    /// Get the weight matrix (for visualization/analysis)
    pub fn get_weights(&self) -> &Vec<Vec<f64>> {
        &self.weights
    }

    /// Get STRF parameters
    pub fn n_channels(&self) -> usize {
        self.n_channels
    }

    pub fn n_time_bins(&self) -> usize {
        self.n_time_bins
    }

    pub fn total_duration(&self) -> f64 {
        self.total_duration
    }

    /// Set learning rate
    pub fn set_learning_rate(&mut self, rate: f64) {
        self.learning_rate = rate;
    }

    /// Compute L2 norm of weights (for regularization monitoring)
    pub fn weight_norm(&self) -> f64 {
        let mut norm = 0.0;
        for t in 0..self.n_time_bins {
            for c in 0..self.n_channels {
                norm += self.weights[t][c].powi(2);
            }
        }
        norm.sqrt()
    }
}

/// History buffer for efficient STRF computation
#[derive(Clone, Debug)]
pub struct STRFHistoryBuffer {
    /// Ring buffer of input frames
    buffer: VecDeque<Vec<f64>>,
    /// Maximum history length (in frames)
    max_length: usize,
    /// Number of channels
    n_channels: usize,
}

impl STRFHistoryBuffer {
    /// Create a new history buffer
    pub fn new(max_length: usize, n_channels: usize) -> Self {
        let mut buffer = VecDeque::with_capacity(max_length);
        // Pre-fill with zeros
        for _ in 0..max_length {
            buffer.push_back(vec![0.0; n_channels]);
        }
        Self {
            buffer,
            max_length,
            n_channels,
        }
    }

    /// Add a new frame to the buffer
    pub fn push(&mut self, frame: Vec<f64>) {
        assert_eq!(frame.len(), self.n_channels);
        if self.buffer.len() >= self.max_length {
            self.buffer.pop_front();
        }
        self.buffer.push_back(frame);
    }

    /// Get the buffer as a slice for STRF computation
    pub fn as_slice(&self) -> Vec<Vec<f64>> {
        self.buffer.iter().cloned().collect()
    }

    /// Reset buffer to zeros
    pub fn reset(&mut self) {
        for frame in &mut self.buffer {
            for v in frame {
                *v = 0.0;
            }
        }
    }

    /// Get current length
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strf_creation() {
        let strf = STRF::new(30, 100.0, 5.0, 42);
        assert_eq!(strf.n_channels(), 30);
        assert_eq!(strf.n_time_bins(), 20); // 100ms / 5ms
    }

    #[test]
    fn test_strf_response() {
        let strf = STRF::with_pattern(10, 50.0, 5.0, 5, 2);
        let n_bins = strf.n_time_bins();

        // Create input that matches the pattern (activity at center channel)
        let mut history: Vec<Vec<f64>> = vec![vec![0.0; 10]; n_bins];
        for t in 0..n_bins {
            history[t][5] = 1.0; // Center channel active
        }

        let response = strf.compute(&history);
        assert!(response > 0.0, "Should have positive response to matching input");

        // Create non-matching input (activity at edge)
        let mut off_history: Vec<Vec<f64>> = vec![vec![0.0; 10]; n_bins];
        for t in 0..n_bins {
            off_history[t][0] = 1.0; // Edge channel active
        }

        let off_response = strf.compute(&off_history);
        assert!(
            response > off_response,
            "Center input should give stronger response"
        );
    }

    #[test]
    fn test_history_buffer() {
        let mut buffer = STRFHistoryBuffer::new(10, 5);

        for i in 0..15 {
            buffer.push(vec![i as f64; 5]);
        }

        let slice = buffer.as_slice();
        assert_eq!(slice.len(), 10);
        // Should contain frames 5-14 (most recent 10)
        assert_eq!(slice[0][0], 5.0);
        assert_eq!(slice[9][0], 14.0);
    }

    #[test]
    fn test_reward_gated_learning() {
        let mut strf = STRF::new(10, 50.0, 5.0, 42);
        strf.set_learning_rate(0.1); // High learning rate for testing

        let n_bins = strf.n_time_bins();
        let mut history: Vec<Vec<f64>> = vec![vec![0.0; 10]; n_bins];
        history[n_bins - 1][5] = 1.0; // Recent activity at channel 5

        // Compute with eligibility
        let _ = strf.compute_with_eligibility(&history, 0.001);

        // Get initial weight
        let initial_weight = strf.weights[n_bins - 1][5];

        // Positive reward should increase weight
        strf.update_weights(1.0, 1.0);
        let new_weight = strf.weights[n_bins - 1][5];

        assert!(
            new_weight > initial_weight,
            "Positive reward should increase weight: {} -> {}",
            initial_weight,
            new_weight
        );
    }
}
