//! Adaptive exponential integrate-and-fire (AdEx) neuron.
//!
//! Brette & Gerstner (2005), "Adaptive exponential integrate-and-fire
//! model as an effective description of neuronal activity." J. Neurophysiol.
//!
//!   C dV/dt = -g_L (V - E_L) + g_L Δ_T exp((V - V_T) / Δ_T) - w + I
//!   τ_w dw/dt = a (V - E_L) - w
//!
//!   on V >= V_peak:   spike; V := V_reset; w += b; refractory for t_ref

use serde::{Deserialize, Serialize};

/// AdEx parameters. Defaults from Brette & Gerstner 2005 regular-spiking cortical fit.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdExParams {
    /// Membrane capacitance (pF).
    pub c: f64,
    /// Leak conductance (nS).
    pub g_l: f64,
    /// Leak reversal / resting potential (mV).
    pub e_l: f64,
    /// Exponential threshold (mV) — soft threshold where spike initiation accelerates.
    pub v_t: f64,
    /// Slope factor (mV).
    pub delta_t: f64,
    /// Peak voltage at which a spike is registered (mV).
    pub v_peak: f64,
    /// Reset voltage after spike (mV).
    pub v_reset: f64,
    /// Adaptation coupling to subthreshold voltage (nS).
    pub a: f64,
    /// Adaptation time constant (ms).
    pub tau_w: f64,
    /// Spike-triggered adaptation increment (pA).
    pub b: f64,
    /// Absolute refractory period (ms).
    pub t_ref: f64,
}

impl Default for AdExParams {
    fn default() -> Self {
        Self {
            c: 281.0,
            g_l: 30.0,
            e_l: -70.6,
            v_t: -50.4,
            delta_t: 2.0,
            v_peak: 20.0,
            v_reset: -70.6,
            a: 4.0,
            tau_w: 144.0,
            b: 80.5,
            t_ref: 2.0,
        }
    }
}

impl AdExParams {
    /// Regular-spiking cortical pyramidal (default).
    pub fn regular_spiking() -> Self {
        Self::default()
    }

    /// Fast-spiking interneuron (PV): low adaptation, low spike-triggered increment.
    pub fn fast_spiking() -> Self {
        Self {
            a: 2.0,
            tau_w: 30.0,
            b: 0.0,
            t_ref: 1.0,
            ..Self::default()
        }
    }

    /// Bursting cortical neuron: strong spike-triggered adaptation.
    pub fn bursting() -> Self {
        Self {
            a: 4.0,
            tau_w: 20.0,
            b: 200.0,
            v_reset: -50.0,
            ..Self::default()
        }
    }
}

/// AdEx neuron state.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdEx {
    pub params: AdExParams,
    /// Membrane voltage (mV).
    pub v: f64,
    /// Adaptation current (pA).
    pub w: f64,
    /// Time remaining in refractory (ms).
    refractory_remaining: f64,
}

impl AdEx {
    pub fn new(params: AdExParams) -> Self {
        let v = params.e_l;
        Self {
            params,
            v,
            w: 0.0,
            refractory_remaining: 0.0,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(AdExParams::default())
    }

    pub fn reset(&mut self) {
        self.v = self.params.e_l;
        self.w = 0.0;
        self.refractory_remaining = 0.0;
    }

    /// Advance the neuron by `dt` ms with injected current `i_input` (pA).
    /// Returns true if the neuron spiked during this step.
    pub fn step(&mut self, i_input: f64, dt_ms: f64) -> bool {
        let p = &self.params;

        if self.refractory_remaining > 0.0 {
            self.refractory_remaining -= dt_ms;
            // Adaptation still decays during refractory.
            self.w += dt_ms * (p.a * (self.v - p.e_l) - self.w) / p.tau_w;
            self.v = p.v_reset;
            return false;
        }

        // dV/dt = (-g_L*(V - E_L) + g_L*ΔT*exp((V - V_T)/ΔT) - w + I) / C
        let leak = -p.g_l * (self.v - p.e_l);
        let exp_arg = (self.v - p.v_t) / p.delta_t;
        // Clamp exponential to avoid overflow on the way to spike; v_peak handles emission.
        let spike_init = p.g_l * p.delta_t * exp_arg.min(20.0).exp();
        let dv = (leak + spike_init - self.w + i_input) / p.c;

        // dw/dt = (a*(V - E_L) - w) / τ_w
        let dw = (p.a * (self.v - p.e_l) - self.w) / p.tau_w;

        self.v += dt_ms * dv;
        self.w += dt_ms * dw;

        if self.v >= p.v_peak {
            self.v = p.v_reset;
            self.w += p.b;
            self.refractory_remaining = p.t_ref;
            return true;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn count_spikes(neuron: &mut AdEx, current: f64, duration_ms: f64, dt_ms: f64) -> usize {
        let n_steps = (duration_ms / dt_ms) as usize;
        (0..n_steps).filter(|_| neuron.step(current, dt_ms)).count()
    }

    #[test]
    fn silent_at_rest() {
        let mut n = AdEx::with_defaults();
        let spikes = count_spikes(&mut n, 0.0, 100.0, 0.1);
        assert_eq!(spikes, 0);
    }

    #[test]
    fn fires_above_rheobase() {
        // Rheobase for default params ≈ g_L * (V_T - E_L) ≈ 30 * 20.2 ≈ 606 pA
        // Use clearly suprathreshold drive.
        let mut n = AdEx::with_defaults();
        let spikes = count_spikes(&mut n, 800.0, 1000.0, 0.1);
        assert!(spikes > 0, "expected spikes under 800 pA, got {}", spikes);
    }

    #[test]
    fn f_i_curve_monotonic() {
        let make = || AdEx::with_defaults();
        let mut low = make();
        let mut high = make();
        let n_low = count_spikes(&mut low, 700.0, 500.0, 0.1);
        let n_high = count_spikes(&mut high, 1500.0, 500.0, 0.1);
        assert!(n_high > n_low, "f-I non-monotonic: low={}, high={}", n_low, n_high);
    }

    #[test]
    fn adaptation_reduces_rate() {
        // Spike count in first half vs second half of a long constant drive should decrease.
        let mut n = AdEx::with_defaults();
        let dt_ms = 0.1;
        let half_steps = 5000; // 500 ms
        let early: usize = (0..half_steps).filter(|_| n.step(1000.0, dt_ms)).count();
        let late: usize = (0..half_steps).filter(|_| n.step(1000.0, dt_ms)).count();
        assert!(late <= early, "adaptation should reduce rate: early={}, late={}", early, late);
        assert!(late < early, "expected strict decrease, got early={}, late={}", early, late);
    }

    #[test]
    fn refractory_lower_bound_on_isi() {
        let mut n = AdEx::with_defaults();
        let dt_ms = 0.05;
        let n_steps = 10000;
        let mut last_spike_step: Option<usize> = None;
        for i in 0..n_steps {
            if n.step(2000.0, dt_ms) {
                if let Some(last) = last_spike_step {
                    let isi_ms = (i - last) as f64 * dt_ms;
                    assert!(
                        isi_ms >= n.params.t_ref - 1e-6,
                        "ISI {} ms violates refractory {} ms",
                        isi_ms,
                        n.params.t_ref
                    );
                }
                last_spike_step = Some(i);
            }
        }
    }

    #[test]
    fn fast_spiking_no_adaptation_increment() {
        // Fast-spiking interneurons have b=0; spike count should be ~constant across windows.
        let mut n = AdEx::new(AdExParams::fast_spiking());
        let dt_ms = 0.1;
        let win = 2000;
        let a: usize = (0..win).filter(|_| n.step(800.0, dt_ms)).count();
        let b: usize = (0..win).filter(|_| n.step(800.0, dt_ms)).count();
        let diff = (a as i64 - b as i64).abs();
        assert!(diff <= 5, "fast-spiking should not adapt: a={}, b={}", a, b);
    }
}
