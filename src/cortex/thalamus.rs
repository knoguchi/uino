//! Thalamic Relay (Medial Geniculate Nucleus)
//!
//! The thalamus serves as a gateway between the auditory nerve and cortex.
//! It performs:
//! - Separation of onset (transient) vs sustained (tonic) responses
//! - Predictive suppression of expected input from A1 feedback
//! - Gain control and gating

use std::f64::consts::PI;

/// Output from the thalamic relay
#[derive(Clone, Debug)]
pub struct ThalamicOutput {
    /// Onset population spikes (one per channel)
    pub onset_spikes: Vec<bool>,
    /// Sustained population spikes (one per channel)
    pub sustained_spikes: Vec<bool>,
    /// Onset population membrane potentials (for continuous readout)
    pub onset_activity: Vec<f64>,
    /// Sustained population membrane potentials
    pub sustained_activity: Vec<f64>,
    /// Total onset strength (for oscillator reset detection)
    pub onset_strength: f64,
}

/// Onset neuron with strong adaptation (transient response)
#[derive(Clone, Debug)]
pub struct OnsetNeuron {
    /// Membrane potential
    membrane: f64,
    /// Firing threshold
    threshold: f64,
    /// Base threshold (for reset)
    base_threshold: f64,
    /// Membrane time constant (fast: ~10ms)
    tau: f64,
    /// Adaptation variable
    adaptation: f64,
    /// Adaptation time constant (~50ms)
    adapt_tau: f64,
    /// Adaptation strength
    adapt_strength: f64,
    /// Refractory period remaining
    refractory: f64,
    /// Refractory period duration
    ref_period: f64,
}

impl OnsetNeuron {
    pub fn new(tau: f64) -> Self {
        Self {
            membrane: 0.0,
            threshold: 0.5,       // Lower threshold
            base_threshold: 0.5,
            tau,
            adaptation: 0.0,
            adapt_tau: 0.050, // 50ms adaptation
            adapt_strength: 1.5, // Moderate adaptation for onset detection
            refractory: 0.0,
            ref_period: 0.002, // 2ms refractory
        }
    }

    /// Process one time step
    /// Returns true if neuron spiked
    pub fn step(&mut self, input: f64, dt: f64) -> bool {
        // Refractory period
        if self.refractory > 0.0 {
            self.refractory -= dt;
            // Decay membrane during refractory
            self.membrane *= (-dt / self.tau).exp();
            return false;
        }

        // Decay adaptation
        self.adaptation *= (-dt / self.adapt_tau).exp();

        // Effective threshold increases with adaptation
        let effective_threshold = self.base_threshold + self.adapt_strength * self.adaptation;

        // Leaky integration
        let decay = (-dt / self.tau).exp();
        self.membrane = self.membrane * decay + input * (1.0 - decay);

        // Spike detection
        if self.membrane >= effective_threshold {
            self.membrane = 0.0;
            self.refractory = self.ref_period;
            self.adaptation += 1.0; // Increase adaptation on spike
            true
        } else {
            false
        }
    }

    pub fn reset(&mut self) {
        self.membrane = 0.0;
        self.adaptation = 0.0;
        self.refractory = 0.0;
    }

    pub fn get_activity(&self) -> f64 {
        self.membrane
    }
}

/// Sustained neuron with weak adaptation (tonic response)
#[derive(Clone, Debug)]
pub struct SustainedNeuron {
    /// Membrane potential
    membrane: f64,
    /// Firing threshold
    threshold: f64,
    /// Membrane time constant (slower: ~30ms)
    tau: f64,
    /// Weak adaptation
    adaptation: f64,
    /// Adaptation time constant
    adapt_tau: f64,
    /// Adaptation strength (weak)
    adapt_strength: f64,
    /// Refractory period remaining
    refractory: f64,
    /// Refractory period duration
    ref_period: f64,
}

impl SustainedNeuron {
    pub fn new(tau: f64) -> Self {
        Self {
            membrane: 0.0,
            threshold: 0.4,   // Lower threshold
            tau,
            adaptation: 0.0,
            adapt_tau: 0.200, // Slower adaptation decay
            adapt_strength: 0.2, // Weak adaptation for sustained response
            refractory: 0.0,
            ref_period: 0.002,
        }
    }

    pub fn step(&mut self, input: f64, dt: f64) -> bool {
        if self.refractory > 0.0 {
            self.refractory -= dt;
            self.membrane *= (-dt / self.tau).exp();
            return false;
        }

        // Slow adaptation decay
        self.adaptation *= (-dt / self.adapt_tau).exp();

        let effective_threshold = self.threshold + self.adapt_strength * self.adaptation;

        // Leaky integration with longer time constant
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

    pub fn reset(&mut self) {
        self.membrane = 0.0;
        self.adaptation = 0.0;
        self.refractory = 0.0;
    }

    pub fn get_activity(&self) -> f64 {
        self.membrane
    }
}

/// Thalamic relay with onset and sustained populations
#[derive(Clone, Debug)]
pub struct ThalamicRelay {
    /// Onset neurons (one per cochlear channel)
    onset_neurons: Vec<OnsetNeuron>,
    /// Sustained neurons (one per cochlear channel)
    sustained_neurons: Vec<SustainedNeuron>,
    /// Number of channels
    n_channels: usize,
    /// Predictive suppression gain (0 = no suppression, 1 = full suppression)
    suppression_gain: f64,
    /// Input gain (scales cochlear input)
    input_gain: f64,
    /// Onset time constant
    onset_tau: f64,
    /// Sustained time constant
    sustained_tau: f64,
}

impl ThalamicRelay {
    /// Create a new thalamic relay
    ///
    /// # Arguments
    /// * `n_channels` - Number of cochlear channels
    /// * `onset_tau` - Time constant for onset neurons (default: 10ms)
    /// * `sustained_tau` - Time constant for sustained neurons (default: 30ms)
    /// * `suppression_gain` - How much to suppress predicted input (default: 0.5)
    pub fn new(
        n_channels: usize,
        onset_tau: f64,
        sustained_tau: f64,
        suppression_gain: f64,
    ) -> Self {
        let onset_neurons = (0..n_channels)
            .map(|_| OnsetNeuron::new(onset_tau))
            .collect();
        let sustained_neurons = (0..n_channels)
            .map(|_| SustainedNeuron::new(sustained_tau))
            .collect();

        Self {
            onset_neurons,
            sustained_neurons,
            n_channels,
            suppression_gain,
            input_gain: 0.01, // Scale down cochlear firing rates (typically 0-200 spikes/s)
            onset_tau,
            sustained_tau,
        }
    }

    /// Create with default parameters
    pub fn with_defaults(n_channels: usize) -> Self {
        Self::new(
            n_channels,
            0.010, // 10ms onset
            0.030, // 30ms sustained
            0.5,   // 50% suppression
        )
    }

    /// Process one time step
    ///
    /// # Arguments
    /// * `cochlear_input` - Firing rates from cochlea (one per channel)
    /// * `a1_prediction` - Top-down prediction from A1 (one per channel)
    /// * `dt` - Time step in seconds
    pub fn step(
        &mut self,
        cochlear_input: &[f64],
        a1_prediction: &[f64],
        dt: f64,
    ) -> ThalamicOutput {
        assert_eq!(cochlear_input.len(), self.n_channels);
        assert_eq!(a1_prediction.len(), self.n_channels);

        let mut onset_spikes = Vec::with_capacity(self.n_channels);
        let mut sustained_spikes = Vec::with_capacity(self.n_channels);
        let mut onset_activity = Vec::with_capacity(self.n_channels);
        let mut sustained_activity = Vec::with_capacity(self.n_channels);

        let mut total_onset = 0.0;

        for i in 0..self.n_channels {
            // Compute prediction error (input - expected)
            // Clamp to non-negative (can't have negative firing)
            let prediction_error = (cochlear_input[i] * self.input_gain
                - self.suppression_gain * a1_prediction[i])
                .max(0.0);

            // Drive both populations with prediction error
            // Onset neurons get the raw error (respond to changes)
            // Sustained neurons get smoothed error (respond to steady-state)
            let onset_spike = self.onset_neurons[i].step(prediction_error, dt);
            let sustained_spike = self.sustained_neurons[i].step(prediction_error, dt);

            if onset_spike {
                total_onset += 1.0;
            }

            onset_spikes.push(onset_spike);
            sustained_spikes.push(sustained_spike);
            onset_activity.push(self.onset_neurons[i].get_activity());
            sustained_activity.push(self.sustained_neurons[i].get_activity());
        }

        // Normalize onset strength by number of channels
        let onset_strength = total_onset / self.n_channels as f64;

        ThalamicOutput {
            onset_spikes,
            sustained_spikes,
            onset_activity,
            sustained_activity,
            onset_strength,
        }
    }

    /// Reset all neurons to initial state
    pub fn reset(&mut self) {
        for neuron in &mut self.onset_neurons {
            neuron.reset();
        }
        for neuron in &mut self.sustained_neurons {
            neuron.reset();
        }
    }

    /// Get current onset activity (membrane potentials)
    pub fn get_onset_activity(&self) -> Vec<f64> {
        self.onset_neurons.iter().map(|n| n.get_activity()).collect()
    }

    /// Get current sustained activity
    pub fn get_sustained_activity(&self) -> Vec<f64> {
        self.sustained_neurons.iter().map(|n| n.get_activity()).collect()
    }

    /// Set suppression gain
    pub fn set_suppression_gain(&mut self, gain: f64) {
        self.suppression_gain = gain.clamp(0.0, 1.0);
    }

    /// Set input gain
    pub fn set_input_gain(&mut self, gain: f64) {
        self.input_gain = gain;
    }

    /// Get number of channels
    pub fn n_channels(&self) -> usize {
        self.n_channels
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onset_neuron_adaptation() {
        let mut neuron = OnsetNeuron::new(0.010);
        let dt = 0.0001; // 0.1ms steps

        // Strong sustained input
        let input = 5.0;
        let mut spike_count = 0;
        let mut spike_times = Vec::new();

        for i in 0..1000 {
            // 100ms
            if neuron.step(input, dt) {
                spike_count += 1;
                spike_times.push(i as f64 * dt);
            }
        }

        // Onset neuron should spike frequently at first, then adapt
        assert!(spike_count > 0, "Should spike at least once");

        // Check that inter-spike intervals increase (adaptation)
        if spike_times.len() >= 3 {
            let isi1 = spike_times[1] - spike_times[0];
            let isi_last = spike_times[spike_times.len() - 1] - spike_times[spike_times.len() - 2];
            assert!(
                isi_last > isi1,
                "ISI should increase due to adaptation"
            );
        }
    }

    #[test]
    fn test_sustained_neuron_tonic() {
        let mut neuron = SustainedNeuron::new(0.030);
        let dt = 0.0001;

        let input = 3.0;
        let mut spike_count = 0;

        for _ in 0..2000 {
            // 200ms
            if neuron.step(input, dt) {
                spike_count += 1;
            }
        }

        // Sustained neuron should maintain firing
        assert!(spike_count > 5, "Sustained neuron should fire multiple times");
    }

    #[test]
    fn test_predictive_suppression() {
        let n_channels = 10;
        let mut relay = ThalamicRelay::with_defaults(n_channels);
        let dt = 0.0001;

        // Input with no prediction (realistic cochlear firing rates ~100-200 spikes/s)
        let input: Vec<f64> = vec![150.0; n_channels];
        let no_prediction: Vec<f64> = vec![0.0; n_channels];

        // Run for 50ms without prediction
        let mut spikes_no_pred = 0;
        for _ in 0..500 {
            let out = relay.step(&input, &no_prediction, dt);
            spikes_no_pred += out.onset_spikes.iter().filter(|&&s| s).count();
            spikes_no_pred += out.sustained_spikes.iter().filter(|&&s| s).count();
        }

        // Reset
        relay.reset();

        // Same input but with matching prediction (scaled to match input after gain)
        let prediction: Vec<f64> = vec![1.5; n_channels]; // After input_gain, input is 1.5
        let mut spikes_with_pred = 0;
        for _ in 0..500 {
            let out = relay.step(&input, &prediction, dt);
            spikes_with_pred += out.onset_spikes.iter().filter(|&&s| s).count();
            spikes_with_pred += out.sustained_spikes.iter().filter(|&&s| s).count();
        }

        // Prediction should reduce spiking
        assert!(
            spikes_with_pred < spikes_no_pred,
            "Predictive suppression should reduce spikes: {} vs {}",
            spikes_with_pred,
            spikes_no_pred
        );
    }
}
