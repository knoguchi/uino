//! Belt/Parabelt Auditory Cortex
//!
//! Higher-level auditory areas that:
//! - Integrate over longer timescales (syllable-level, ~200ms)
//! - Maintain context through recurrent activity
//! - Generate top-down predictions for A1
//! - Provide the substrate for phoneme/syllable classification
//!
//! # Biology
//!
//! - Belt areas receive input from A1 and integrate over larger spatial/temporal scales
//! - Heavy recurrence maintains working memory of recent context
//! - Generates predictions based on learned sequences

use std::collections::VecDeque;

/// Output from belt processing
#[derive(Clone, Debug)]
pub struct BeltOutput {
    /// Binary spike output
    pub spikes: Vec<bool>,
    /// Continuous activity (for classification)
    pub activity: Vec<f64>,
    /// Prediction for A1 (what A1 pattern is expected next)
    pub a1_prediction: Vec<f64>,
    /// Syllable-level encoding (compressed representation)
    pub syllable_code: Vec<f64>,
}

/// Belt neuron with long integration time constant
#[derive(Clone, Debug)]
pub struct BeltNeuron {
    /// Membrane potential
    membrane: f64,
    /// Firing threshold
    threshold: f64,
    /// Long time constant (~200ms)
    tau: f64,
    /// Refractory period
    refractory: f64,
    ref_period: f64,
    /// Slow adaptation for sustained context encoding
    adaptation: f64,
    adapt_tau: f64,
}

impl BeltNeuron {
    pub fn new(tau: f64) -> Self {
        Self {
            membrane: 0.0,
            threshold: 0.03,  // Very low threshold
            tau,
            refractory: 0.0,
            ref_period: 0.002, // 2ms
            adaptation: 0.0,
            adapt_tau: 0.300, // 300ms
        }
    }

    pub fn step(&mut self, input: f64, dt: f64) -> bool {
        // Refractory period
        if self.refractory > 0.0 {
            self.refractory -= dt;
            self.membrane *= (-dt / self.tau).exp();
            return false;
        }

        // Slow adaptation decay
        self.adaptation *= (-dt / self.adapt_tau).exp();

        // Effective threshold
        let effective_threshold = self.threshold + self.adaptation * 0.2;

        // Long leaky integration
        let decay = (-dt / self.tau).exp();
        self.membrane = self.membrane * decay + input * (1.0 - decay);

        if self.membrane >= effective_threshold {
            self.membrane = 0.0;
            self.refractory = self.ref_period;
            self.adaptation += 1.0;
            true
        } else {
            false
        }
    }

    pub fn get_activity(&self) -> f64 {
        self.membrane / self.threshold
    }

    pub fn reset(&mut self) {
        self.membrane = 0.0;
        self.refractory = 0.0;
        // Keep adaptation for context continuity
    }
}

/// Belt/Parabelt cortical area
#[derive(Clone)]
pub struct Belt {
    /// Belt neurons
    neurons: Vec<BeltNeuron>,
    /// Number of neurons
    n_neurons: usize,
    /// Number of A1 inputs
    n_a1: usize,

    /// Feedforward weights from A1
    a1_weights: Vec<Vec<f64>>,

    /// Recurrent weights (dense, for working memory)
    recurrent_weights: Vec<Vec<f64>>,
    recurrent_strength: f64,

    /// Feedback weights to A1 (for prediction)
    feedback_weights: Vec<Vec<f64>>,

    /// Activity history for context
    activity_history: VecDeque<Vec<f64>>,
    history_length: usize,

    /// Current state
    current_spikes: Vec<bool>,
    prev_activity: Vec<f64>,

    /// Long time constant
    tau: f64,
}

impl Belt {
    /// Create a new belt area
    ///
    /// # Arguments
    /// * `n_a1` - Number of A1 input neurons
    /// * `n_neurons` - Number of belt neurons (typically smaller than A1)
    /// * `seed` - Random seed
    pub fn new(n_a1: usize, n_neurons: usize, seed: u64) -> Self {
        let tau = 0.200; // 200ms - syllabic integration

        let neurons: Vec<BeltNeuron> = (0..n_neurons).map(|_| BeltNeuron::new(tau)).collect();

        // Random number generator
        let mut rng_state = seed;
        let mut rand = || {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            (rng_state as f64) / (u64::MAX as f64)
        };

        // Sparse feedforward weights from A1
        let input_density = 0.2;
        let a1_weights: Vec<Vec<f64>> = (0..n_neurons)
            .map(|_| {
                (0..n_a1)
                    .map(|_| {
                        if rand() < input_density {
                            rand() * 0.3
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect();

        // Dense recurrent weights for working memory
        let recurrent_density = 0.4;
        let recurrent_weights: Vec<Vec<f64>> = (0..n_neurons)
            .map(|i| {
                (0..n_neurons)
                    .map(|j| {
                        if i == j {
                            0.0 // No self-connections
                        } else if rand() < recurrent_density {
                            // Mix of excitatory and inhibitory
                            if rand() < 0.8 {
                                rand() * 0.2 // Excitatory
                            } else {
                                -rand() * 0.3 // Inhibitory
                            }
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect();

        // Feedback weights to A1 (for generating predictions)
        let feedback_weights: Vec<Vec<f64>> = (0..n_neurons)
            .map(|i| {
                (0..n_a1)
                    .map(|j| {
                        // Initialize with rough inverse of a1_weights
                        let base = if rand() < input_density {
                            rand() * 0.2
                        } else {
                            0.0
                        };
                        base
                    })
                    .collect()
            })
            .collect();

        Self {
            neurons,
            n_neurons,
            n_a1,
            a1_weights,
            recurrent_weights,
            recurrent_strength: 0.7,
            feedback_weights,
            activity_history: VecDeque::with_capacity(20),
            history_length: 20,
            current_spikes: vec![false; n_neurons],
            prev_activity: vec![0.0; n_neurons],
            tau,
        }
    }

    /// Create with default sizing
    pub fn with_defaults(n_a1: usize) -> Self {
        let n_neurons = n_a1 / 3; // Compression
        Self::new(n_a1, n_neurons.max(50), 42)
    }

    /// Process one time step
    ///
    /// # Arguments
    /// * `a1_activity` - Activity from A1 neurons
    /// * `theta_phase` - Current theta phase (for gating)
    /// * `reward` - Global reward signal
    /// * `dt` - Time step
    pub fn step(
        &mut self,
        a1_activity: &[f64],
        theta_phase: f64,
        reward: f64,
        dt: f64,
    ) -> BeltOutput {
        assert_eq!(a1_activity.len(), self.n_a1);

        // 1. Compute feedforward drive from A1
        let ff_drive: Vec<f64> = (0..self.n_neurons)
            .map(|i| {
                self.a1_weights[i]
                    .iter()
                    .zip(a1_activity.iter())
                    .map(|(&w, &a)| w * a)
                    .sum()
            })
            .collect();

        // 2. Compute recurrent drive from previous activity
        let recurrent_drive: Vec<f64> = (0..self.n_neurons)
            .map(|i| {
                self.recurrent_weights[i]
                    .iter()
                    .zip(self.prev_activity.iter())
                    .map(|(&w, &a)| w * a)
                    .sum::<f64>()
                    * self.recurrent_strength
            })
            .collect();

        // 3. Theta-phase gating: process more at theta trough
        let theta_gate = (-theta_phase.cos() + 1.0) / 2.0;
        let gated_ff: Vec<f64> = ff_drive
            .iter()
            .map(|&d| d * (0.5 + 0.5 * theta_gate))
            .collect();

        // 4. Total drive
        let total_drive: Vec<f64> = gated_ff
            .iter()
            .zip(recurrent_drive.iter())
            .map(|(&ff, &rec)| (ff + rec).max(0.0))
            .collect();

        // 5. Update neurons
        let mut spikes = Vec::with_capacity(self.n_neurons);
        let mut activity = Vec::with_capacity(self.n_neurons);

        for i in 0..self.n_neurons {
            let spiked = self.neurons[i].step(total_drive[i], dt);
            spikes.push(spiked);
            activity.push(self.neurons[i].get_activity());
        }

        // 6. Update history
        self.activity_history.push_back(activity.clone());
        if self.activity_history.len() > self.history_length {
            self.activity_history.pop_front();
        }

        // 7. Generate A1 prediction based on current state
        let a1_prediction = self.generate_a1_prediction(&activity);

        // 8. Generate syllable code (compressed representation)
        let syllable_code = self.compute_syllable_code();

        // 9. Update state
        self.current_spikes = spikes.clone();
        self.prev_activity = activity.clone();

        // 10. Learning (if reward present)
        if reward.abs() > 0.001 {
            self.update_weights(reward, a1_activity);
        }

        BeltOutput {
            spikes,
            activity,
            a1_prediction,
            syllable_code,
        }
    }

    /// Generate prediction for A1 based on current belt activity
    fn generate_a1_prediction(&self, activity: &[f64]) -> Vec<f64> {
        let mut prediction = vec![0.0; self.n_a1];

        for (i, &act) in activity.iter().enumerate() {
            if act > 0.01 {
                for j in 0..self.n_a1 {
                    prediction[j] += self.feedback_weights[i][j] * act;
                }
            }
        }

        prediction
    }

    /// Compute syllable-level code from activity history
    fn compute_syllable_code(&self) -> Vec<f64> {
        if self.activity_history.is_empty() {
            return vec![0.0; self.n_neurons];
        }

        // Average activity over history (weighted by recency)
        let mut code = vec![0.0; self.n_neurons];
        let mut total_weight = 0.0;

        for (i, hist) in self.activity_history.iter().enumerate() {
            let weight = (i + 1) as f64; // More recent = higher weight
            total_weight += weight;
            for (j, &act) in hist.iter().enumerate() {
                code[j] += weight * act;
            }
        }

        if total_weight > 0.0 {
            for v in &mut code {
                *v /= total_weight;
            }
        }

        code
    }

    /// Update weights based on reward
    fn update_weights(&mut self, reward: f64, a1_input: &[f64]) {
        let learning_rate = 0.0001;

        // Simple Hebbian update gated by reward
        for i in 0..self.n_neurons {
            let post = self.prev_activity[i];
            if post > 0.1 {
                // Only update for active neurons
                for j in 0..self.n_a1 {
                    let pre = a1_input[j];
                    if pre > 0.1 {
                        // Reward-gated Hebbian
                        self.a1_weights[i][j] += learning_rate * reward * pre * post;
                        // Clamp weights
                        self.a1_weights[i][j] = self.a1_weights[i][j].clamp(-0.5, 0.5);
                    }
                }
            }
        }
    }

    /// Get current A1 prediction
    pub fn get_a1_prediction(&self) -> Vec<f64> {
        self.generate_a1_prediction(&self.prev_activity)
    }

    /// Get current activity
    pub fn get_activity(&self) -> &[f64] {
        &self.prev_activity
    }

    /// Get syllable code
    pub fn get_syllable_code(&self) -> Vec<f64> {
        self.compute_syllable_code()
    }

    /// Reset state
    pub fn reset(&mut self) {
        for neuron in &mut self.neurons {
            neuron.reset();
        }
        self.activity_history.clear();
        self.current_spikes.fill(false);
        self.prev_activity.fill(0.0);
    }

    /// Get number of neurons
    pub fn n_neurons(&self) -> usize {
        self.n_neurons
    }

    /// Get number of A1 inputs
    pub fn n_a1(&self) -> usize {
        self.n_a1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_belt_creation() {
        let belt = Belt::with_defaults(300);
        assert_eq!(belt.n_a1(), 300);
        assert!(belt.n_neurons() >= 50);
    }

    #[test]
    fn test_belt_processing() {
        let mut belt = Belt::new(100, 50, 42);
        let dt = 0.001;

        let a1_activity = vec![0.5; 100];
        let theta_phase = 0.0;

        // Run for 200ms
        for _ in 0..200 {
            let out = belt.step(&a1_activity, theta_phase, 0.0, dt);

            // Check output dimensions
            assert_eq!(out.activity.len(), 50);
            assert_eq!(out.a1_prediction.len(), 100);
            assert_eq!(out.syllable_code.len(), 50);
        }
    }

    #[test]
    fn test_belt_context_memory() {
        let mut belt = Belt::new(50, 30, 42);
        let dt = 0.001;

        // Feed distinct patterns
        let pattern_a: Vec<f64> = (0..50).map(|i| if i < 25 { 1.0 } else { 0.0 }).collect();
        let pattern_b: Vec<f64> = (0..50).map(|i| if i >= 25 { 1.0 } else { 0.0 }).collect();

        // Show pattern A for 100ms
        for _ in 0..100 {
            belt.step(&pattern_a, 0.0, 0.0, dt);
        }
        let code_after_a = belt.get_syllable_code();

        // Reset and show pattern B
        belt.reset();
        for _ in 0..100 {
            belt.step(&pattern_b, 0.0, 0.0, dt);
        }
        let code_after_b = belt.get_syllable_code();

        // Codes should be different
        let diff: f64 = code_after_a
            .iter()
            .zip(code_after_b.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        assert!(diff > 0.1, "Different patterns should produce different codes");
    }
}
