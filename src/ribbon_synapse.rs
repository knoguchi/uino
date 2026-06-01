//! Generic Ribbon Synapse Model
//!
//! Ribbon synapses are specialized structures found in both the cochlea (inner hair cells)
//! and retina (photoreceptors, bipolar cells). They enable sustained, graded neurotransmitter
//! release and have similar adaptation properties across sensory systems.
//!
//! Reference: "Sensory Processing at Ribbon Synapses in the Retina and the Cochlea"
//! - Matthews & Bhave (2008)
//!
//! This module provides a generic ribbon synapse that can be used by:
//! - Cochlea (IHC → auditory nerve)
//! - Retina (photoreceptor → bipolar → ganglion)

use rand::Rng;
use rand_distr::{Distribution, StandardNormal};
use rustfft::num_complex::Complex as FftComplex;
use rustfft::FftPlanner;

/// Ribbon synapse configuration parameters.
///
/// These parameters control the adaptation dynamics and can be tuned
/// for different sensory contexts (cochlea vs retina).
#[derive(Clone, Debug)]
pub struct RibbonSynapseConfig {
    /// Spontaneous release rate (vesicles/s)
    pub spontaneous_rate: f64,

    /// Maximum release rate (vesicles/s)
    pub max_rate: f64,

    /// Fast adaptation time constant (ms)
    pub tau_fast: f64,

    /// Slow adaptation time constant (ms)
    pub tau_slow: f64,

    /// Vesicle replenishment time constant (ms)
    pub tau_replenish: f64,

    /// Power-law exponent for slow adaptation
    pub power_law_exponent: f64,

    /// Whether to use fractional Gaussian noise
    pub use_noise: bool,

    /// Noise scaling factor
    pub noise_sigma: f64,
}

impl Default for RibbonSynapseConfig {
    fn default() -> Self {
        Self {
            spontaneous_rate: 50.0,  // Typical for retinal ganglion cells
            max_rate: 500.0,
            tau_fast: 10.0,          // ms
            tau_slow: 100.0,         // ms
            tau_replenish: 60.0,     // ms
            power_law_exponent: 0.9, // Hurst parameter for fGn
            use_noise: true,
            noise_sigma: 30.0,
        }
    }
}

impl RibbonSynapseConfig {
    /// Create config for cochlear IHC synapse (HSR fiber)
    pub fn cochlea_hsr() -> Self {
        Self {
            spontaneous_rate: 100.0,
            max_rate: 800.0,
            tau_fast: 2.0,
            tau_slow: 60.0,
            tau_replenish: 60.0,
            power_law_exponent: 0.9,
            use_noise: true,
            noise_sigma: 200.0,
        }
    }

    /// Create config for cochlear IHC synapse (MSR fiber)
    pub fn cochlea_msr() -> Self {
        Self {
            spontaneous_rate: 4.0,
            max_rate: 800.0,
            tau_fast: 2.0,
            tau_slow: 60.0,
            tau_replenish: 60.0,
            power_law_exponent: 0.9,
            use_noise: true,
            noise_sigma: 30.0,
        }
    }

    /// Create config for cochlear IHC synapse (LSR fiber)
    pub fn cochlea_lsr() -> Self {
        Self {
            spontaneous_rate: 0.1,
            max_rate: 800.0,
            tau_fast: 2.0,
            tau_slow: 60.0,
            tau_replenish: 60.0,
            power_law_exponent: 0.9,
            use_noise: true,
            noise_sigma: 3.0,
        }
    }

    /// Create config for retinal ganglion cell (midget/P-cell)
    pub fn retina_midget() -> Self {
        Self {
            spontaneous_rate: 10.0,
            max_rate: 300.0,
            tau_fast: 5.0,
            tau_slow: 200.0,
            tau_replenish: 100.0,
            power_law_exponent: 0.9,
            use_noise: true,
            noise_sigma: 20.0,
        }
    }

    /// Create config for retinal ganglion cell (parasol/M-cell)
    pub fn retina_parasol() -> Self {
        Self {
            spontaneous_rate: 30.0,
            max_rate: 500.0,
            tau_fast: 3.0,
            tau_slow: 150.0,
            tau_replenish: 80.0,
            power_law_exponent: 0.9,
            use_noise: true,
            noise_sigma: 40.0,
        }
    }
}

/// Power-law adaptation filter state (IIR approximation).
///
/// This implements the power-law dynamics observed in ribbon synapses
/// using an efficient O(n) IIR filter approximation.
#[derive(Clone, Debug, Default)]
pub struct PowerLawState {
    // Slow adaptation cascaded IIR state
    pub n1: [f64; 3],
    pub n2: [f64; 3],
    pub n3: [f64; 3],
    // Fast adaptation cascaded IIR state
    pub m1: [f64; 3],
    pub m2: [f64; 3],
    pub m3: [f64; 3],
    pub m4: [f64; 3],
    pub m5: [f64; 3],
}

impl PowerLawState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Process fast adaptation pathway.
    #[inline]
    pub fn process_fast(&mut self, k: usize, input: f64, history: &[f64]) -> f64 {
        if k == 0 {
            self.m1[0] = 0.2 * input;
            self.m2[0] = self.m1[0];
            self.m3[0] = self.m2[0];
            self.m4[0] = self.m3[0];
            self.m5[0] = self.m4[0];
        } else if k == 1 {
            let prev = history[0];
            self.m1[1] = 0.491115852967412 * self.m1[0]
                + 0.2 * (input - 0.173492003319319 * prev);
            self.m2[1] = 1.084520302502860 * self.m2[0] + self.m1[1] - 0.803462163297112 * self.m1[0];
            self.m3[1] = 1.588427084535629 * self.m3[0] + self.m2[1] - 1.416084732997016 * self.m2[0];
            self.m4[1] = 1.886287488516458 * self.m4[0] + self.m3[1] - 1.830362725074550 * self.m3[0];
            self.m5[1] = 1.989549282714008 * self.m5[0] + self.m4[1] - 1.983165053215032 * self.m4[0];

            self.m1[2] = self.m1[1];
            self.m1[1] = self.m1[0];
            self.m2[2] = self.m2[1];
            self.m2[1] = self.m2[0];
            self.m3[2] = self.m3[1];
            self.m3[1] = self.m3[0];
            self.m4[2] = self.m4[1];
            self.m4[1] = self.m4[0];
            self.m5[2] = self.m5[1];
            self.m5[1] = self.m5[0];
        } else {
            let prev1 = history[0];
            let prev2 = history[1];

            let m1_new = 0.491115852967412 * self.m1[0] - 0.055050209956838 * self.m1[1]
                + 0.2 * (input - 0.173492003319319 * prev1 + 0.000000172983796 * prev2);
            let m2_new = 1.084520302502860 * self.m2[0] - 0.288760329320566 * self.m2[1]
                + m1_new - 0.803462163297112 * self.m1[0] + 0.154962026341513 * self.m1[1];
            let m3_new = 1.588427084535629 * self.m3[0] - 0.628138993662508 * self.m3[1]
                + m2_new - 1.416084732997016 * self.m2[0] + 0.496615555008723 * self.m2[1];
            let m4_new = 1.886287488516458 * self.m4[0] - 0.888972875389923 * self.m4[1]
                + m3_new - 1.830362725074550 * self.m3[0] + 0.836399964176882 * self.m3[1];
            let m5_new = 1.989549282714008 * self.m5[0] - 0.989558985673023 * self.m5[1]
                + m4_new - 1.983165053215032 * self.m4[0] + 0.983193027347456 * self.m4[1];

            self.m1[1] = self.m1[0];
            self.m1[0] = m1_new;
            self.m2[1] = self.m2[0];
            self.m2[0] = m2_new;
            self.m3[1] = self.m3[0];
            self.m3[0] = m3_new;
            self.m4[1] = self.m4[0];
            self.m4[0] = m4_new;
            self.m5[1] = self.m5[0];
            self.m5[0] = m5_new;
        }

        self.m5[0]
    }

    /// Process slow adaptation pathway.
    #[inline]
    pub fn process_slow(&mut self, k: usize, input: f64, history: &[f64]) -> f64 {
        if k == 0 {
            self.n1[0] = 1.0e-3 * input;
            self.n2[0] = self.n1[0];
            self.n3[0] = self.n2[0];
        } else if k == 1 {
            let prev = history[0];
            self.n1[1] = 1.992127932802320 * self.n1[0]
                + 1.0e-3 * (input - 0.994466986569624 * prev);
            self.n2[1] = 1.999195329360981 * self.n2[0] + self.n1[1] - 1.997855276593802 * self.n1[0];
            self.n3[1] = -0.798261718183851 * self.n3[0] + self.n2[1] + 0.798261718184977 * self.n2[0];

            self.n1[2] = self.n1[1];
            self.n1[1] = self.n1[0];
            self.n2[2] = self.n2[1];
            self.n2[1] = self.n2[0];
            self.n3[2] = self.n3[1];
            self.n3[1] = self.n3[0];
        } else {
            let prev1 = history[0];
            let prev2 = history[1];

            let n1_new = 1.992127932802320 * self.n1[0] - 0.992140616993846 * self.n1[1]
                + 1.0e-3 * (input - 0.994466986569624 * prev1 + 0.000000000002347 * prev2);
            let n2_new = 1.999195329360981 * self.n2[0] - 0.999195402928777 * self.n2[1]
                + n1_new - 1.997855276593802 * self.n1[0] + 0.997855827934345 * self.n1[1];
            let n3_new = -0.798261718183851 * self.n3[0] - 0.199131619873480 * self.n3[1]
                + n2_new + 0.798261718184977 * self.n2[0] + 0.199131619874064 * self.n2[1];

            self.n1[1] = self.n1[0];
            self.n1[0] = n1_new;
            self.n2[1] = self.n2[0];
            self.n2[0] = n2_new;
            self.n3[1] = self.n3[0];
            self.n3[0] = n3_new;
        }

        self.n3[0]
    }
}

/// Generate fractional Gaussian noise for stochastic vesicle release.
pub fn generate_fgn<R: Rng>(n: usize, sample_rate: f64, hurst: f64, sigma: f64, rng: &mut R) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }

    // Determine internal sampling rate (downsample for efficiency)
    let internal_rate = 10e3;
    let resamp = (sample_rate / internal_rate).ceil() as usize;
    let mut n_internal = (n as f64 / resamp as f64).ceil() as usize + 1;
    if n_internal < 10 {
        n_internal = 10;
    }

    let y = if (hurst - 0.5).abs() < 1e-10 {
        // H = 0.5 is white noise
        (0..n_internal)
            .map(|_| StandardNormal.sample(rng))
            .collect::<Vec<f64>>()
    } else {
        // Davies-Harte method for fGn
        let nfft = (2 * (n_internal - 1)).next_power_of_two();
        let nfft_half = nfft / 2;

        // Autocorrelation sequence
        let mut autocov: Vec<f64> = vec![0.0; nfft];
        for i in 0..=nfft_half {
            let ki = i as f64;
            autocov[i] = 0.5 * ((ki + 1.0).powf(2.0 * hurst) - 2.0 * ki.powf(2.0 * hurst)
                + (ki - 1.0).abs().powf(2.0 * hurst));
        }
        for i in (nfft_half + 1)..nfft {
            autocov[i] = autocov[nfft - i];
        }

        // FFT
        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(nfft);

        let mut spectrum: Vec<FftComplex<f64>> = autocov
            .iter()
            .map(|&x| FftComplex::new(x, 0.0))
            .collect();
        fft.process(&mut spectrum);

        // sqrt of eigenvalues
        let eigenval_sqrt: Vec<f64> = spectrum
            .iter()
            .map(|c| c.re.max(0.0).sqrt())
            .collect();

        // Generate weighted random
        let mut z: Vec<FftComplex<f64>> = eigenval_sqrt
            .iter()
            .map(|&ev| {
                let re: f64 = StandardNormal.sample(rng);
                let im: f64 = StandardNormal.sample(rng);
                FftComplex::new(ev * re, ev * im)
            })
            .collect();

        // IFFT
        let ifft = planner.plan_fft_inverse(nfft);
        ifft.process(&mut z);

        let scale = 1.0 / (nfft as f64).sqrt();
        z.iter()
            .take(n_internal)
            .map(|c| c.re * scale)
            .collect()
    };

    // Resample to output rate
    let y_resampled = resample_linear(&y, resamp);

    // Scale
    y_resampled.iter().take(n).map(|&yi| yi * sigma).collect()
}

/// Linear interpolation resampling.
fn resample_linear(signal: &[f64], factor: usize) -> Vec<f64> {
    let n = signal.len();
    let out_len = n * factor;
    let mut result = Vec::with_capacity(out_len);

    for i in 0..n - 1 {
        let start = signal[i];
        let end = signal[i + 1];
        for j in 0..factor {
            let t = j as f64 / factor as f64;
            result.push(start + t * (end - start));
        }
    }

    for _ in 0..factor {
        result.push(signal[n - 1]);
    }

    result
}

/// Simple decimation with averaging.
fn decimate(signal: &[f64], factor: usize) -> Vec<f64> {
    if factor <= 1 {
        return signal.to_vec();
    }

    let mut result = Vec::with_capacity(signal.len() / factor + 1);
    let mut i = 0;

    while i < signal.len() {
        let end = (i + factor).min(signal.len());
        let sum: f64 = signal[i..end].iter().sum();
        result.push(sum / (end - i) as f64);
        i += factor;
    }

    result
}

/// Generic ribbon synapse processor.
///
/// This processes receptor potential → instantaneous firing rate
/// using the ribbon synapse dynamics shared between cochlea and retina.
#[derive(Clone)]
pub struct RibbonSynapse {
    config: RibbonSynapseConfig,
    power_law: PowerLawState,
    /// Internal sampling rate for power-law (10 kHz)
    internal_rate: f64,
    /// Resampling factor
    resamp: usize,
    /// Time resolution
    tdres: f64,
    /// Double-exponential state
    ci: f64,
    cl: f64,
    /// Parameters derived from config
    pi_max: f64,
    synstrength: f64,
    synslope: f64,
    vi: f64,
    vl: f64,
    pg: f64,
    pl: f64,
    cg: f64,
    /// Power-law coefficients
    alpha1: f64,
    alpha2: f64,
    /// History for IIR
    sout1_history: [f64; 2],
    sout2_history: [f64; 2],
}

impl RibbonSynapse {
    /// Create a new ribbon synapse processor with optional position-dependent factor.
    ///
    /// The `adaptation_factor` scales the adaptation dynamics:
    /// - For cochlea: derived from CF (characteristic frequency)
    /// - For retina: derived from eccentricity (distance from fovea)
    /// - Use 1.0 for generic/default behavior
    fn new_internal(config: RibbonSynapseConfig, sample_rate: f64, adaptation_factor: f64) -> Self {
        let tdres = 1.0 / sample_rate;
        let internal_rate = 10e3;
        let resamp = (1.0 / (tdres * internal_rate)).ceil() as usize;

        // Derive double-exponential parameters
        let spont = config.spontaneous_rate;
        let ass = config.max_rate;
        let pi_max = 0.6;

        let asp = spont * 2.75;
        let tau_r = config.tau_fast / 1000.0;
        let tau_st = config.tau_slow / 1000.0;
        let ar_ast = 6.0;
        let pts = 3.0;

        let aon = pts * ass;
        let ar = (aon - ass) * ar_ast / (1.0 + ar_ast);
        let _ast = aon - ass - ar;
        let prest = pi_max / aon * asp;
        let cg = (asp * (aon - asp)) / (aon * prest * (1.0 - asp / ass));
        let gamma1 = cg / asp;
        let gamma2 = cg / ass;
        let k1 = -1.0 / tau_r;
        let k2 = -1.0 / tau_st;

        let vi0 = (1.0 - pi_max / prest)
            / (gamma1 * (ar * (k1 - k2) / cg / pi_max + k2 / prest / gamma1 - k2 / pi_max / gamma2));
        let vi1 = (1.0 - pi_max / prest)
            / (gamma1 * (_ast * (k2 - k1) / cg / pi_max + k1 / prest / gamma1 - k1 / pi_max / gamma2));
        let vi = (vi0 + vi1) / 2.0;
        let alpha = gamma2 / k1 / k2;
        let beta = -(k1 + k2) * alpha;
        let theta1 = alpha * pi_max / vi;
        let theta2 = vi / pi_max;
        let theta3 = gamma2 - 1.0 / pi_max;

        let pl = ((beta - theta2 * theta3) / theta1 - 1.0) * pi_max;
        let pg = 1.0 / (theta3 - 1.0 / pl);
        let vl = theta1 * pl * pg;
        let ci = asp / prest;
        let cl = ci * (prest + pl) / pl;

        // adaptation_factor affects kslope (CF-dependent in cochlea, eccentricity-dependent in retina)
        let kslope = (1.0 + 50.0) / (5.0 + 50.0) * adaptation_factor * 20.0 * pi_max;
        let vsat = if kslope >= 0.0 { kslope + prest } else { prest };
        let tmpst = vsat / prest * 2.0_f64.ln();
        let synstrength = if tmpst < 400.0 {
            (tmpst.exp() - 1.0).ln()
        } else {
            tmpst
        };
        let synslope = prest / 2.0_f64.ln() * synstrength;

        // Power-law coefficients
        let alpha1 = 2.5e-6 * 100e3;
        let alpha2 = 1e-2 * 100e3;

        Self {
            config,
            power_law: PowerLawState::new(),
            internal_rate,
            resamp,
            tdres,
            ci,
            cl,
            pi_max,
            synstrength,
            synslope,
            vi,
            vl: vl.abs(),
            pg,
            pl,
            cg,
            alpha1,
            alpha2,
            sout1_history: [0.0; 2],
            sout2_history: [0.0; 2],
        }
    }

    /// Create a new ribbon synapse processor (generic, no position-dependent adaptation).
    pub fn new(config: RibbonSynapseConfig, sample_rate: f64) -> Self {
        Self::new_internal(config, sample_rate, 1.0)
    }

    /// Create ribbon synapse for cochlea with CF-dependent adaptation.
    ///
    /// # Arguments
    /// * `config` - Synapse configuration (use cochlea_hsr/msr/lsr)
    /// * `sample_rate` - Sample rate in Hz
    /// * `cf` - Characteristic frequency in Hz (20-20000 typical)
    ///
    /// The CF affects adaptation dynamics - high CF fibers adapt differently than low CF.
    pub fn new_for_cochlea(config: RibbonSynapseConfig, sample_rate: f64, cf: f64) -> Self {
        let cf_factor = Self::compute_cf_factor(config.spontaneous_rate, cf);
        Self::new_internal(config, sample_rate, cf_factor)
    }

    /// Compute CF-dependent adaptation factor (from original Zilany model).
    ///
    /// This factor varies with both CF and fiber type (via spontaneous rate).
    fn compute_cf_factor(spont: f64, cf: f64) -> f64 {
        if (spont - 100.0).abs() < 1e-10 {
            // HSR fiber
            (10.0_f64.powf(0.29 * cf / 1e3 + 0.7)).min(800.0)
        } else if (spont - 4.0).abs() < 1e-10 {
            // MSR fiber
            (2.5e-4 * cf * 4.0 + 0.2).min(50.0)
        } else {
            // LSR fiber (spont = 0.1)
            (2.5e-4 * cf * 0.1 + 0.15).min(1.0)
        }
    }

    /// Create ribbon synapse for retina with eccentricity-dependent adaptation.
    ///
    /// # Arguments
    /// * `config` - Synapse configuration (use retina_midget/parasol)
    /// * `sample_rate` - Sample rate in Hz
    /// * `eccentricity_deg` - Distance from fovea in degrees (0-90)
    ///
    /// Foveal cells (low eccentricity) may adapt differently than peripheral cells.
    pub fn new_for_retina(config: RibbonSynapseConfig, sample_rate: f64, eccentricity_deg: f64) -> Self {
        let ecc_factor = Self::compute_eccentricity_factor(eccentricity_deg);
        Self::new_internal(config, sample_rate, ecc_factor)
    }

    /// Compute eccentricity-dependent adaptation factor for retina.
    ///
    /// TODO: This is a placeholder. Need to find proper values from retinal physiology literature.
    /// For now, assumes fovea (0°) has factor ~1.0 and periphery has higher factor.
    fn compute_eccentricity_factor(eccentricity_deg: f64) -> f64 {
        // Placeholder: linear scaling from 1.0 at fovea to ~5.0 at 90° periphery
        // Real values should come from literature on retinal ganglion cell adaptation
        1.0 + eccentricity_deg / 90.0 * 4.0
    }

    /// Create with cochlea HSR config (no CF, uses default adaptation).
    pub fn cochlea_hsr(sample_rate: f64) -> Self {
        Self::new(RibbonSynapseConfig::cochlea_hsr(), sample_rate)
    }

    /// Create with cochlea HSR config with CF-dependent adaptation.
    pub fn cochlea_hsr_with_cf(sample_rate: f64, cf: f64) -> Self {
        Self::new_for_cochlea(RibbonSynapseConfig::cochlea_hsr(), sample_rate, cf)
    }

    /// Create with retina midget cell config.
    pub fn retina_midget(sample_rate: f64) -> Self {
        Self::new(RibbonSynapseConfig::retina_midget(), sample_rate)
    }

    /// Create with retina midget cell config with eccentricity.
    pub fn retina_midget_with_eccentricity(sample_rate: f64, eccentricity_deg: f64) -> Self {
        Self::new_for_retina(RibbonSynapseConfig::retina_midget(), sample_rate, eccentricity_deg)
    }

    /// Reset processor state.
    pub fn reset(&mut self) {
        self.power_law.reset();
        self.sout1_history = [0.0; 2];
        self.sout2_history = [0.0; 2];

        // Reset double-exponential state
        let spont = self.config.spontaneous_rate;
        let asp = spont * 2.75;
        let prest = self.pi_max / (3.0 * self.config.max_rate) * asp;
        self.ci = asp / prest;
        self.cl = self.ci * (prest + self.pl) / self.pl;
    }

    /// Process receptor potential signal through ribbon synapse.
    ///
    /// # Arguments
    /// * `receptor_potential` - Input signal (voltage from sensor)
    /// * `rng` - Random number generator for noise
    ///
    /// # Returns
    /// Instantaneous firing rate (spikes/s)
    pub fn process<R: Rng>(&mut self, receptor_potential: &[f64], rng: &mut R) -> Vec<f64> {
        let totalstim = receptor_potential.len();
        let delaypoint = 750; // Fixed delay for alignment

        // Generate noise if enabled
        let noise_len = ((totalstim + 2 * delaypoint) as f64 * self.tdres * self.internal_rate).ceil() as usize;
        let noise = if self.config.use_noise {
            generate_fgn(
                noise_len,
                self.internal_rate,
                self.config.power_law_exponent,
                self.config.noise_sigma,
                rng,
            )
        } else {
            vec![0.0; noise_len]
        };

        // Double-exponential adaptation
        let mut expon_out = vec![0.0; totalstim];

        for (indx, &vr) in receptor_potential.iter().enumerate() {
            let tmp = self.synstrength * vr;
            let tmp = if tmp < 400.0 {
                (1.0 + tmp.exp()).ln()
            } else {
                tmp
            };
            let ppi = self.synslope / self.synstrength * tmp;

            let ci_last = self.ci;
            self.ci += (self.tdres / self.vi) * (-ppi * self.ci + self.pl * (self.cl - self.ci));
            self.cl += (self.tdres / self.vl) * (-self.pl * (self.cl - ci_last) + self.pg * (self.cg - self.cl));

            if self.ci < 0.0 {
                let temp = 1.0 / self.pg + 1.0 / self.pl + 1.0 / ppi;
                self.ci = self.cg / (ppi * temp);
                self.cl = self.ci * (ppi + self.pl) / self.pl;
            }

            expon_out[indx] = self.ci * ppi;
        }

        // Add delay padding
        let power_law_in_len = totalstim + 3 * delaypoint;
        let mut power_law_in = vec![0.0; power_law_in_len];
        for k in 0..delaypoint {
            power_law_in[k] = expon_out[0];
        }
        for k in delaypoint..(totalstim + delaypoint) {
            power_law_in[k] = expon_out[k - delaypoint];
        }
        for k in (totalstim + delaypoint)..power_law_in_len {
            power_law_in[k] = power_law_in[k - 1];
        }

        // Downsample for power-law
        let samp_input = decimate(&power_law_in, self.resamp);

        // Power-law adaptation
        let synapse_len = ((totalstim + 2 * delaypoint) as f64 * self.tdres * self.internal_rate).floor() as usize;
        let mut syn_samp_out = vec![0.0; synapse_len];

        self.power_law.reset();
        self.sout1_history = [0.0; 2];
        self.sout2_history = [0.0; 2];

        for k in 0..synapse_len.min(samp_input.len()).min(noise.len()) {
            let samp = samp_input[k];
            let n = noise[k];

            let sout1 = (samp + n - self.alpha1 * self.power_law.m5[0]).max(0.0);
            let sout2 = (samp - self.alpha2 * self.power_law.n3[0]).max(0.0);

            let _ = self.power_law.process_fast(k, sout1, &self.sout1_history);
            let _ = self.power_law.process_slow(k, sout2, &self.sout2_history);

            self.sout1_history[1] = self.sout1_history[0];
            self.sout1_history[0] = sout1;
            self.sout2_history[1] = self.sout2_history[0];
            self.sout2_history[0] = sout2;

            syn_samp_out[k] = sout1 + sout2;
        }

        // Upsample
        let mut tmp_syn = vec![0.0; totalstim + 2 * delaypoint];
        for z in 0..(synapse_len - 1).min(syn_samp_out.len() - 1) {
            let incr = (syn_samp_out[z + 1] - syn_samp_out[z]) / self.resamp as f64;
            for b in 0..self.resamp {
                let idx = z * self.resamp + b;
                if idx < tmp_syn.len() {
                    tmp_syn[idx] = syn_samp_out[z] + b as f64 * incr;
                }
            }
        }

        // Extract with delay correction
        let mut output = vec![0.0; totalstim];
        for i in 0..totalstim {
            if i + delaypoint < tmp_syn.len() {
                output[i] = tmp_syn[i + delaypoint];
            }
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_ribbon_synapse_basic() {
        let mut rng = StdRng::seed_from_u64(42);
        let sample_rate = 100e3;
        let mut synapse = RibbonSynapse::cochlea_hsr(sample_rate);

        // Simple ramp input
        let n = 1000;
        let input: Vec<f64> = (0..n).map(|i| 0.5 * (i as f64 / n as f64)).collect();

        let output = synapse.process(&input, &mut rng);

        assert_eq!(output.len(), n);
        assert!(output.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_retina_config() {
        let mut rng = StdRng::seed_from_u64(42);
        let sample_rate = 10e3; // Typical for vision experiments
        let mut synapse = RibbonSynapse::retina_midget(sample_rate);

        let n = 500;
        let input: Vec<f64> = (0..n).map(|i| 0.3 * (1.0 + (i as f64 * 0.1).sin())).collect();

        let output = synapse.process(&input, &mut rng);

        assert_eq!(output.len(), n);
        assert!(output.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_fgn_generation() {
        let mut rng = StdRng::seed_from_u64(123);

        let n = 1000;
        let sample_rate = 10e3;
        let hurst = 0.9;
        let sigma = 30.0;

        let noise = generate_fgn(n, sample_rate, hurst, sigma, &mut rng);

        assert!(!noise.is_empty(), "fGn should produce output");
        assert_eq!(noise.len(), n);

        let mean = noise.iter().sum::<f64>() / noise.len() as f64;
        let variance: f64 = noise.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / noise.len() as f64;

        assert!(variance > 0.0, "fGn should have non-zero variance");
        assert!(noise.iter().all(|&x| x.is_finite()), "fGn should have finite values");
    }
}
