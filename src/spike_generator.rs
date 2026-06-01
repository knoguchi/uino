//! Spike generator implementation.
//!
//! Ported from model_Synapse.c lines 341-413.
//! Implements an inhomogeneous Poisson process with refractory period.

use rand::Rng;

/// Spike generator parameters.
#[derive(Clone, Debug)]
pub struct SpikeGeneratorParams {
    /// First exponential coefficient
    pub c0: f64,
    /// First exponential time constant
    pub s0: f64,
    /// Second exponential coefficient
    pub c1: f64,
    /// Second exponential time constant
    pub s1: f64,
    /// Absolute refractory period (dead time) in seconds
    pub dead: f64,
}

impl Default for SpikeGeneratorParams {
    fn default() -> Self {
        Self {
            c0: 0.5,
            s0: 0.001,
            c1: 0.5,
            s1: 0.0125,
            dead: 0.00075, // 0.75 ms
        }
    }
}

/// Spike generator state.
pub struct SpikeGenerator {
    params: SpikeGeneratorParams,
}

impl Default for SpikeGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl SpikeGenerator {
    /// Create a new spike generator with default parameters.
    pub fn new() -> Self {
        Self {
            params: SpikeGeneratorParams::default(),
        }
    }

    /// Create a spike generator with custom parameters.
    pub fn with_params(params: SpikeGeneratorParams) -> Self {
        Self { params }
    }

    /// Generate spikes from synapse output.
    ///
    /// Uses the algorithm from B. Scott Jackson.
    ///
    /// # Arguments
    /// * `synout` - Synapse output (instantaneous firing rate)
    /// * `tdres` - Time resolution (1/fs)
    /// * `rng` - Random number generator
    ///
    /// # Returns
    /// Vector of spike times in seconds
    pub fn generate<R: Rng>(&self, synout: &[f64], tdres: f64, rng: &mut R) -> Vec<f64> {
        let totalstim = synout.len();
        if totalstim == 0 {
            return Vec::new();
        }

        let dt = totalstim as f64 * tdres; // Total duration
        let nout_max = (dt / self.params.dead).ceil() as usize;

        // Pre-generate random numbers for efficiency
        let rand_nums: Vec<f64> = (0..nout_max + 1).map(|_| rng.gen::<f64>()).collect();
        let mut rand_buf_index = 0;

        let mut spike_times = Vec::with_capacity(nout_max);

        // Calculate useful constants
        let deadtime_index = (self.params.dead / tdres).floor() as usize;
        let deadtime_rnd = deadtime_index as f64 * tdres;

        let refrac_mult0 = 1.0 - tdres / self.params.s0;
        let refrac_mult1 = 1.0 - tdres / self.params.s1;

        // Calculate effects of a random spike before t=0 on refractoriness
        let end_of_last_deadtime = if synout[0] == 0.0 {
            0.0
        } else {
            if rand_buf_index >= rand_nums.len() {
                return spike_times;
            }
            let r = rand_nums[rand_buf_index];
            rand_buf_index += 1;
            r.ln() / synout[0]
        };

        let mut refrac_value0 = self.params.c0 * (end_of_last_deadtime / self.params.s0).exp();
        let mut refrac_value1 = self.params.c1 * (end_of_last_deadtime / self.params.s1).exp();

        // Value of time-warping sum (normalized by tdres)
        let mut xsum = synout[0]
            * (-end_of_last_deadtime
                + self.params.c0 * self.params.s0 * ((end_of_last_deadtime / self.params.s0).exp() - 1.0)
                + self.params.c1 * self.params.s1 * ((end_of_last_deadtime / self.params.s1).exp() - 1.0));

        // Calculate first interspike interval in unit-rate Poisson process
        if rand_buf_index >= rand_nums.len() {
            return spike_times;
        }
        let mut unit_rate_intrvl = -rand_nums[rand_buf_index].ln() / tdres;
        rand_buf_index += 1;

        let mut count_time = tdres;
        let mut k = 0usize;

        while k < totalstim && count_time < dt {
            if synout[k] > 0.0 {
                // Add synout*(refractory value) to time-warping sum
                xsum += synout[k] * (1.0 - refrac_value0 - refrac_value1);

                // Spike occurs when time-warping sum exceeds interspike "time"
                if xsum >= unit_rate_intrvl {
                    spike_times.push(count_time);

                    if rand_buf_index >= rand_nums.len() {
                        break;
                    }
                    unit_rate_intrvl = -rand_nums[rand_buf_index].ln() / tdres;
                    rand_buf_index += 1;
                    xsum = 0.0;

                    // Skip deadtime and reset refractory function
                    k += deadtime_index;
                    count_time += deadtime_rnd;
                    refrac_value0 = self.params.c0;
                    refrac_value1 = self.params.c1;
                }
            }

            k += 1;
            count_time += tdres;
            refrac_value0 *= refrac_mult0;
            refrac_value1 *= refrac_mult1;
        }

        spike_times
    }
}

/// Convenience function to run spike generation.
///
/// # Arguments
/// * `synout` - Synapse output (instantaneous firing rate)
/// * `tdres` - Time resolution (1/fs)
/// * `rng` - Random number generator
///
/// # Returns
/// Vector of spike times in seconds
pub fn run_spike_generator<R: Rng>(synout: &[f64], tdres: f64, rng: &mut R) -> Vec<f64> {
    let generator = SpikeGenerator::new();
    generator.generate(synout, tdres, rng)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_spike_generator() {
        let mut rng = StdRng::seed_from_u64(42);
        let fs = 100e3;
        let tdres = 1.0 / fs;

        // Create a constant firing rate signal
        let rate = 100.0; // 100 spikes/s
        let duration = 0.1; // 100 ms
        let n_samples = (duration * fs) as usize;
        let synout = vec![rate; n_samples];

        let spike_times = run_spike_generator(&synout, tdres, &mut rng);

        // Check that we got some spikes
        assert!(!spike_times.is_empty());

        // Check that all spike times are within the duration
        for &t in &spike_times {
            assert!(t >= 0.0);
            assert!(t <= duration);
        }

        // Check that spike times are monotonically increasing
        for i in 1..spike_times.len() {
            assert!(spike_times[i] > spike_times[i - 1]);
        }

        // Check that interspike intervals respect the dead time
        let dead_time = 0.00075;
        for i in 1..spike_times.len() {
            let isi = spike_times[i] - spike_times[i - 1];
            assert!(isi >= dead_time - 1e-10);
        }
    }

    #[test]
    fn test_spike_generator_zero_rate() {
        let mut rng = StdRng::seed_from_u64(42);
        let fs = 100e3;
        let tdres = 1.0 / fs;

        // Zero firing rate should produce no spikes
        let n_samples = 1000;
        let synout = vec![0.0; n_samples];

        let spike_times = run_spike_generator(&synout, tdres, &mut rng);

        assert!(spike_times.is_empty());
    }

    #[test]
    fn test_spike_generator_high_rate() {
        let mut rng = StdRng::seed_from_u64(42);
        let fs = 100e3;
        let tdres = 1.0 / fs;

        // Very high firing rate
        let rate = 10000.0; // 10000 spikes/s (very high)
        let duration = 0.01; // 10 ms
        let n_samples = (duration * fs) as usize;
        let synout = vec![rate; n_samples];

        let spike_times = run_spike_generator(&synout, tdres, &mut rng);

        // Should be rate-limited by dead time
        // Maximum possible rate = 1/dead_time = 1/0.00075 ≈ 1333 spikes/s
        let expected_max_spikes = (duration / 0.00075).ceil() as usize;
        assert!(spike_times.len() <= expected_max_spikes);
    }
}
