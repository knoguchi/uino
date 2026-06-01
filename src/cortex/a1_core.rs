//! Primary Auditory Cortex (A1) Core
//!
//! A1 is the first cortical stage of auditory processing. It:
//! - Learns spectro-temporal receptive fields (STRFs)
//! - Implements sparse overcomplete representations
//! - Computes prediction errors for predictive coding
//! - Generates feedback predictions for the thalamus
//!
//! # Architecture
//!
//! - Overcomplete: 10-50× more neurons than thalamic inputs
//! - Sparse: only ~5-10% of neurons active at any time
//! - Lateral inhibition enforces sparsity
//! - Recurrent connections for local integration

use crate::cortex::strf::{STRF, STRFHistoryBuffer};

/// Output from A1 processing
#[derive(Clone, Debug)]
pub struct A1Output {
    /// Binary spike output for each neuron
    pub spikes: Vec<bool>,
    /// Continuous activity level (membrane potential normalized)
    pub activity: Vec<f64>,
    /// Prediction sent to thalamus (for predictive suppression)
    pub thalamic_prediction: Vec<f64>,
    /// Current sparsity (fraction of active neurons)
    pub sparsity: f64,
    /// Average firing rate
    pub mean_rate: f64,
}

/// A1 neuron with adaptive threshold for homeostatic sparsity
#[derive(Clone, Debug)]
pub struct A1Neuron {
    /// Membrane potential
    membrane: f64,
    /// Current firing threshold
    threshold: f64,
    /// Base threshold
    base_threshold: f64,
    /// Membrane time constant (50ms for phonetic integration)
    tau: f64,
    /// Adaptive threshold for sparsity control
    threshold_adapt: f64,
    /// Target firing rate (for homeostasis)
    target_rate: f64,
    /// Homeostatic adaptation rate
    homeostatic_rate: f64,
    /// Refractory period remaining
    refractory: f64,
    /// Refractory period duration
    ref_period: f64,
    /// Recent spike history (for rate estimation)
    recent_spikes: f64,
    /// Spike history decay
    spike_history_tau: f64,
}

impl A1Neuron {
    pub fn new(tau: f64, target_rate: f64) -> Self {
        Self {
            membrane: 0.0,
            threshold: 50.0,     // Threshold around mean STRF drive
            base_threshold: 50.0,
            tau,
            threshold_adapt: 0.0,
            target_rate,
            homeostatic_rate: 0.5, // Fast adaptation for quick sparsity control
            refractory: 0.0,
            ref_period: 0.001, // 1ms refractory
            recent_spikes: 0.0,
            spike_history_tau: 0.100, // 100ms for faster rate estimation
        }
    }

    /// Process one time step
    ///
    /// # Arguments
    /// * `excitation` - Excitatory input (from STRF + recurrent)
    /// * `inhibition` - Inhibitory input (lateral inhibition)
    /// * `dt` - Time step
    ///
    /// # Returns
    /// True if neuron spiked
    pub fn step(&mut self, excitation: f64, inhibition: f64, dt: f64) -> bool {
        // Update spike history estimate
        self.recent_spikes *= (-dt / self.spike_history_tau).exp();

        // Refractory period
        if self.refractory > 0.0 {
            self.refractory -= dt;
            self.membrane *= (-dt / self.tau).exp();
            return false;
        }

        // Net input
        let net_input = (excitation - inhibition).max(0.0);

        // Leaky integration
        let decay = (-dt / self.tau).exp();
        self.membrane = self.membrane * decay + net_input * (1.0 - decay);

        // Effective threshold with homeostatic adaptation
        let effective_threshold = self.base_threshold + self.threshold_adapt;

        // Spike detection
        if self.membrane >= effective_threshold {
            self.membrane = 0.0;
            self.refractory = self.ref_period;
            self.recent_spikes += 1.0;

            // Homeostatic threshold adaptation
            // If firing too much, increase threshold
            let rate_error = self.recent_spikes / self.spike_history_tau - self.target_rate;
            self.threshold_adapt += self.homeostatic_rate * rate_error;
            self.threshold_adapt = self.threshold_adapt.max(0.0); // Can't go below base

            true
        } else {
            // Slowly decrease threshold if not firing enough
            let rate_error = self.recent_spikes / self.spike_history_tau - self.target_rate;
            self.threshold_adapt += self.homeostatic_rate * rate_error * 0.1; // Slower decrease
            self.threshold_adapt = self.threshold_adapt.max(0.0);

            false
        }
    }

    pub fn get_activity(&self) -> f64 {
        self.membrane / (self.base_threshold + self.threshold_adapt)
    }

    pub fn reset(&mut self) {
        self.membrane = 0.0;
        self.refractory = 0.0;
        self.recent_spikes = 0.0;
        // Don't reset threshold_adapt - keep learned homeostasis
    }

    pub fn get_rate_estimate(&self) -> f64 {
        self.recent_spikes / self.spike_history_tau
    }
}

/// Primary Auditory Cortex (A1) Core
#[derive(Clone)]
pub struct A1Core {
    /// A1 neurons
    neurons: Vec<A1Neuron>,
    /// Number of neurons (overcomplete)
    n_neurons: usize,
    /// Number of thalamic input channels (onset + sustained)
    n_thalamic: usize,
    /// Number of cochlear channels (for feedback prediction)
    n_cochlear: usize,

    /// STRFs for each neuron (learnable)
    strfs: Vec<STRF>,
    /// Input history buffer for STRF computation
    history_buffer: STRFHistoryBuffer,

    /// Lateral inhibition weights (sparse)
    /// Only store non-zero connections
    inhibition: Vec<Vec<(usize, f64)>>,
    /// Inhibition strength
    inhibition_strength: f64,

    /// Recurrent excitatory weights (sparse, local)
    recurrent: Vec<Vec<(usize, f64)>>,
    /// Recurrent strength
    recurrent_strength: f64,

    /// Feedback weights to cochlea (for prediction) - size n_cochlear
    feedback_weights: Vec<Vec<f64>>,

    /// Target sparsity
    target_sparsity: f64,
    /// Integration time constant
    tau: f64,

    /// Current spike state
    current_spikes: Vec<bool>,
    /// Previous activity (for recurrence)
    prev_activity: Vec<f64>,
}

impl A1Core {
    /// Create a new A1 core
    ///
    /// # Arguments
    /// * `n_thalamic` - Number of thalamic input channels (onset + sustained = 2 × cochlear)
    /// * `n_cochlear` - Number of cochlear channels (for feedback prediction)
    /// * `n_neurons` - Number of A1 neurons (should be 10-50× n_cochlear)
    /// * `strf_duration_ms` - STRF duration in milliseconds
    /// * `strf_bin_ms` - STRF time bin size in milliseconds
    /// * `target_sparsity` - Target fraction of active neurons (e.g., 0.05)
    /// * `seed` - Random seed
    pub fn new(
        n_thalamic: usize,
        n_cochlear: usize,
        n_neurons: usize,
        strf_duration_ms: f64,
        strf_bin_ms: f64,
        target_sparsity: f64,
        seed: u64,
    ) -> Self {
        let tau = 0.050; // 50ms
        let target_rate = 20.0; // 20 Hz target rate

        // Create neurons
        let neurons: Vec<A1Neuron> = (0..n_neurons)
            .map(|_| A1Neuron::new(tau, target_rate))
            .collect();

        // Create STRFs with diverse initializations
        let strfs: Vec<STRF> = (0..n_neurons)
            .map(|i| {
                // Distribute center frequencies across channels
                let center = (i * n_thalamic / n_neurons) % n_thalamic;
                let bandwidth = 2 + (i % 4); // Varying bandwidth
                if i % 3 == 0 {
                    // Some random, some structured
                    STRF::new(n_thalamic, strf_duration_ms, strf_bin_ms, seed + i as u64)
                } else {
                    STRF::with_pattern(n_thalamic, strf_duration_ms, strf_bin_ms, center, bandwidth)
                }
            })
            .collect();

        // History buffer
        let n_history = (strf_duration_ms / strf_bin_ms).ceil() as usize + 5;
        let history_buffer = STRFHistoryBuffer::new(n_history, n_thalamic);

        // Create sparse lateral inhibition (global, all-to-all but weak)
        let mut rng_state = seed;
        let mut rand = || {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            (rng_state as f64) / (u64::MAX as f64)
        };

        let inhibition_density = 0.1; // 10% connectivity
        let inhibition: Vec<Vec<(usize, f64)>> = (0..n_neurons)
            .map(|i| {
                (0..n_neurons)
                    .filter_map(|j| {
                        if i != j && rand() < inhibition_density {
                            Some((j, rand() * 0.5 + 0.1))
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .collect();

        // Sparse local recurrent excitation
        let recurrent_density = 0.05;
        let recurrent: Vec<Vec<(usize, f64)>> = (0..n_neurons)
            .map(|i| {
                (0..n_neurons)
                    .filter_map(|j| {
                        if i != j && rand() < recurrent_density {
                            Some((j, rand() * 0.3))
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .collect();

        // Feedback weights to cochlea (learned, initialized to spread across channels)
        let feedback_weights: Vec<Vec<f64>> = (0..n_neurons)
            .map(|i| {
                // Simple initialization: spread weight across cochlear channels
                let mut weights = vec![0.0; n_cochlear];
                let center = (i * n_cochlear / n_neurons) % n_cochlear;
                for c in 0..n_cochlear {
                    let dist = (c as i32 - center as i32).abs() as f64;
                    weights[c] = (-dist.powi(2) / 8.0).exp() * 0.1;
                }
                weights
            })
            .collect();

        Self {
            neurons,
            n_neurons,
            n_thalamic,
            n_cochlear,
            strfs,
            history_buffer,
            inhibition,
            inhibition_strength: 50.0, // Strong inhibition to enforce sparsity
            recurrent,
            recurrent_strength: 0.5,
            feedback_weights,
            target_sparsity,
            tau,
            current_spikes: vec![false; n_neurons],
            prev_activity: vec![0.0; n_neurons],
        }
    }

    /// Create with default parameters
    ///
    /// # Arguments
    /// * `n_cochlear` - Number of cochlear channels
    pub fn with_defaults(n_cochlear: usize) -> Self {
        let n_thalamic = n_cochlear * 2; // onset + sustained
        let n_neurons = n_cochlear * 20; // 20× overcomplete
        Self::new(
            n_thalamic,
            n_cochlear,
            n_neurons,
            100.0, // 100ms STRF
            5.0,   // 5ms bins
            0.05,  // 5% sparsity
            42,
        )
    }

    /// Process one time step
    ///
    /// # Arguments
    /// * `thalamic_input` - Combined onset and sustained activity from thalamus
    /// * `belt_prediction` - Top-down prediction from belt (per A1 neuron)
    /// * `reward` - Global reward signal for learning
    /// * `dt` - Time step in seconds
    pub fn step(
        &mut self,
        thalamic_input: &[f64],
        belt_prediction: &[f64],
        reward: f64,
        dt: f64,
    ) -> A1Output {
        assert_eq!(thalamic_input.len(), self.n_thalamic);
        assert_eq!(belt_prediction.len(), self.n_neurons);

        // 1. Update history buffer
        self.history_buffer.push(thalamic_input.to_vec());

        // 2. Compute STRF responses for all neurons
        let history = self.history_buffer.as_slice();
        let strf_drive: Vec<f64> = self
            .strfs
            .iter_mut()
            .map(|strf| strf.compute_with_eligibility(&history, dt))
            .collect();

        // 3. Compute recurrent input from previous activity
        let recurrent_input: Vec<f64> = (0..self.n_neurons)
            .map(|i| {
                self.recurrent[i]
                    .iter()
                    .map(|&(j, w)| w * self.prev_activity[j])
                    .sum::<f64>()
                    * self.recurrent_strength
            })
            .collect();

        // 4. Compute total excitation (STRF + recurrent - belt prediction)
        let excitation: Vec<f64> = (0..self.n_neurons)
            .map(|i| {
                let strf = strf_drive[i];
                let recur = recurrent_input[i];
                let pred = belt_prediction[i];
                // Prediction error: subtract expected activity
                (strf + recur - pred * 0.5).max(0.0)
            })
            .collect();

        // 5. Compute lateral inhibition
        let inhibition: Vec<f64> = (0..self.n_neurons)
            .map(|i| {
                self.inhibition[i]
                    .iter()
                    .map(|&(j, w)| w * self.prev_activity[j])
                    .sum::<f64>()
                    * self.inhibition_strength
            })
            .collect();

        // 6. Update neurons and collect spikes
        let mut spikes = Vec::with_capacity(self.n_neurons);
        let mut activity = Vec::with_capacity(self.n_neurons);
        let mut spike_count = 0;

        for i in 0..self.n_neurons {
            let spiked = self.neurons[i].step(excitation[i], inhibition[i], dt);
            if spiked {
                spike_count += 1;
            }
            spikes.push(spiked);
            activity.push(self.neurons[i].get_activity());
        }

        // 7. Update previous activity
        self.prev_activity = activity.clone();
        self.current_spikes = spikes.clone();

        // 8. Compute sparsity
        let sparsity = spike_count as f64 / self.n_neurons as f64;

        // 9. Update STRF weights if reward is non-zero
        if reward.abs() > 0.001 {
            for (i, strf) in self.strfs.iter_mut().enumerate() {
                // Only update if neuron was recently active
                let post_activity = activity[i];
                strf.update_weights(reward, post_activity);
            }
        }

        // 10. Generate thalamic prediction (feedback)
        let thalamic_prediction = self.generate_thalamic_prediction(&activity);

        // 11. Compute mean rate
        let mean_rate: f64 = self.neurons.iter().map(|n| n.get_rate_estimate()).sum::<f64>()
            / self.n_neurons as f64;

        A1Output {
            spikes,
            activity,
            thalamic_prediction,
            sparsity,
            mean_rate,
        }
    }

    /// Generate prediction for thalamus based on current A1 activity
    /// Returns a cochlear-sized prediction (what cochlear activity is expected)
    fn generate_thalamic_prediction(&self, activity: &[f64]) -> Vec<f64> {
        let mut prediction = vec![0.0; self.n_cochlear];

        for (i, &act) in activity.iter().enumerate() {
            if act > 0.01 {
                for c in 0..self.n_cochlear {
                    prediction[c] += self.feedback_weights[i][c] * act;
                }
            }
        }

        prediction
    }

    /// Get current thalamic prediction
    pub fn get_thalamic_prediction(&self) -> Vec<f64> {
        self.generate_thalamic_prediction(&self.prev_activity)
    }

    /// Get current activity levels
    pub fn get_activity(&self) -> &[f64] {
        &self.prev_activity
    }

    /// Get current spikes
    pub fn get_spikes(&self) -> &[bool] {
        &self.current_spikes
    }

    /// Reset state (but preserve learned weights)
    pub fn reset(&mut self) {
        for neuron in &mut self.neurons {
            neuron.reset();
        }
        for strf in &mut self.strfs {
            strf.reset_eligibility();
        }
        self.history_buffer.reset();
        self.current_spikes.fill(false);
        self.prev_activity.fill(0.0);
    }

    /// Get number of neurons
    pub fn n_neurons(&self) -> usize {
        self.n_neurons
    }

    /// Get number of thalamic inputs
    pub fn n_thalamic(&self) -> usize {
        self.n_thalamic
    }

    /// Set inhibition strength
    pub fn set_inhibition_strength(&mut self, strength: f64) {
        self.inhibition_strength = strength;
    }

    /// Set learning rate for all STRFs
    pub fn set_learning_rate(&mut self, rate: f64) {
        for strf in &mut self.strfs {
            strf.set_learning_rate(rate);
        }
    }

    /// Get average STRF weight norm (for monitoring learning)
    pub fn avg_strf_norm(&self) -> f64 {
        self.strfs.iter().map(|s| s.weight_norm()).sum::<f64>() / self.n_neurons as f64
    }

    /// Debug: compute STRF drive statistics for current input
    pub fn debug_strf_drive(&self) -> (f64, f64, f64) {
        let history = self.history_buffer.as_slice();
        if history.len() < self.strfs[0].n_time_bins() {
            return (0.0, 0.0, 0.0);
        }

        let drives: Vec<f64> = self.strfs.iter()
            .map(|strf| strf.compute(&history))
            .collect();

        let mean = drives.iter().sum::<f64>() / drives.len() as f64;
        let max = drives.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let min = drives.iter().copied().fold(f64::INFINITY, f64::min);
        (mean, min, max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_a1_creation() {
        let a1 = A1Core::with_defaults(30);
        assert_eq!(a1.n_thalamic(), 60); // 2× cochlear (onset + sustained)
        assert_eq!(a1.n_neurons(), 600); // 20× overcomplete
    }

    #[test]
    fn test_a1_sparsity() {
        // n_thalamic=30, n_cochlear=15, n_neurons=300
        let mut a1 = A1Core::new(30, 15, 300, 50.0, 5.0, 0.10, 42);
        let dt = 0.001; // 1ms

        let thalamic_input = vec![1.0; 30]; // Uniform input (onset + sustained)
        let belt_prediction = vec![0.0; 300]; // No prediction

        // Run for 100ms
        let mut total_sparsity = 0.0;
        let mut n_steps = 0;
        for _ in 0..100 {
            let out = a1.step(&thalamic_input, &belt_prediction, 0.0, dt);
            total_sparsity += out.sparsity;
            n_steps += 1;
        }

        let avg_sparsity = total_sparsity / n_steps as f64;
        // Sparsity should be in reasonable range (homeostasis takes time)
        assert!(
            avg_sparsity < 0.5,
            "Sparsity should be reasonably low: {}",
            avg_sparsity
        );
    }

    #[test]
    fn test_a1_learning() {
        // n_thalamic=10, n_cochlear=5, n_neurons=50
        let mut a1 = A1Core::new(10, 5, 50, 50.0, 5.0, 0.10, 42);
        a1.set_learning_rate(0.1); // High learning rate for testing
        let dt = 0.001;

        // Get initial STRF norm
        let initial_norm = a1.avg_strf_norm();

        // Run with consistent input and positive reward
        let thalamic_input = vec![2.0; 10];
        let belt_prediction = vec![0.0; 50];

        for _ in 0..100 {
            a1.step(&thalamic_input, &belt_prediction, 1.0, dt);
        }

        let final_norm = a1.avg_strf_norm();

        // Weights should change with learning
        assert!(
            (final_norm - initial_norm).abs() > 0.001,
            "Weights should change: {} -> {}",
            initial_norm,
            final_norm
        );
    }
}
