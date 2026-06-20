//! Apparent-motion experiment — Phase 1 v1 falsification test.
//!
//! Two microcircuit units (A and B) receive temporally alternating stimuli.
//! Each unit's prediction is augmented by a top-down signal derived from
//! the *delayed* trace of the other unit's PE+ activity. The delay matches
//! the inter-stimulus interval, so A's activity at time `t − Δ` predicts B's
//! input at time `t` — anticipation, not concurrence.
//!
//! Hebbian learning is bidirectional on `(PE+ − PE−)`: weights grow when the
//! partner's delayed trace coincides with PE+ firing here, shrink when it
//! coincides with PE− firing. This drives weights to the value at which
//! delayed top-down matches actual upcoming input.
//!
//! Falsification: with `coupling_enabled = false`, the cross-prediction
//! route is forced to zero. The same probe should NOT produce prediction-
//! driven PE− in B — confirming the illusion is prediction-driven.

use crate::microcircuit::canonical::{Microcircuit, MicrocircuitParams, StepOutput};
use std::collections::VecDeque;

/// Two-unit network running the apparent-motion experiment.
#[derive(Clone, Debug)]
pub struct ApparentMotion {
    pub a: Microcircuit,
    pub b: Microcircuit,
    /// Exponentially decaying trace of PE+ activity per unit.
    pub a_trace: f64,
    pub b_trace: f64,
    /// Ring buffers of past traces — used for delayed cross-prediction.
    /// The buffer length IS the cross-prediction delay in steps.
    a_trace_history: VecDeque<f64>,
    b_trace_history: VecDeque<f64>,
    /// Cross-prediction weights: `delayed_a_trace * w_ab` biases B's prediction.
    pub w_ab: f64,
    pub w_ba: f64,
    pub tau_trace: f64,
    pub learning_rate: f64,
    pub coupling_enabled: bool,
}

#[derive(Clone, Debug, Default)]
pub struct TwoUnitOutput {
    pub a: StepOutput,
    pub b: StepOutput,
}

impl ApparentMotion {
    /// Create with the given delay (ms) and dt (ms) for the cross-prediction lag.
    /// Delay should match the stimulus inter-onset interval for the test to converge.
    pub fn new_with_delay(delay_ms: f64, dt_ms: f64) -> Self {
        let delay_steps = (delay_ms / dt_ms).round() as usize;
        Self {
            a: Microcircuit::with_defaults(),
            b: Microcircuit::with_defaults(),
            a_trace: 0.0,
            b_trace: 0.0,
            a_trace_history: VecDeque::from(vec![0.0; delay_steps]),
            b_trace_history: VecDeque::from(vec![0.0; delay_steps]),
            w_ab: 0.0,
            w_ba: 0.0,
            tau_trace: 50.0,
            learning_rate: 1e-6,
            coupling_enabled: true,
        }
    }

    pub fn new() -> Self {
        Self::new_with_delay(100.0, 0.1)
    }

    pub fn with_unit_params(params: MicrocircuitParams, delay_ms: f64, dt_ms: f64) -> Self {
        let mut s = Self::new_with_delay(delay_ms, dt_ms);
        s.a = Microcircuit::new(params.clone());
        s.b = Microcircuit::new(params);
        s
    }

    pub fn disable_coupling(&mut self) {
        self.coupling_enabled = false;
    }

    pub fn enable_coupling(&mut self) {
        self.coupling_enabled = true;
    }

    pub fn reset(&mut self) {
        self.a.reset();
        self.b.reset();
        self.a_trace = 0.0;
        self.b_trace = 0.0;
        for x in self.a_trace_history.iter_mut() {
            *x = 0.0;
        }
        for x in self.b_trace_history.iter_mut() {
            *x = 0.0;
        }
    }

    pub fn step(&mut self, s_a: f64, s_b: f64, dt_ms: f64) -> TwoUnitOutput {
        // Decay live traces.
        let decay = (-dt_ms / self.tau_trace).exp();
        self.a_trace *= decay;
        self.b_trace *= decay;

        // Delayed traces — what the other unit looked like delay_steps ago.
        let delayed_a = *self.a_trace_history.front().unwrap_or(&0.0);
        let delayed_b = *self.b_trace_history.front().unwrap_or(&0.0);

        let (top_down_a, top_down_b) = if self.coupling_enabled {
            (self.w_ba * delayed_b, self.w_ab * delayed_a)
        } else {
            (0.0, 0.0)
        };

        let out_a = self.a.step_with_top_down(s_a, top_down_a, dt_ms);
        let out_b = self.b.step_with_top_down(s_b, top_down_b, dt_ms);

        // Update live traces with this step's PE+ spikes.
        self.a_trace += out_a.pe_plus_spikes as f64;
        self.b_trace += out_b.pe_plus_spikes as f64;

        // Bidirectional Hebbian on delayed pre × signed post (PE+ − PE−).
        // Weights settle at the value where delayed top-down matches the
        // actual upcoming input.
        if self.coupling_enabled {
            let signed_b = out_b.pe_plus_spikes as f64 - out_b.pe_minus_spikes as f64;
            let signed_a = out_a.pe_plus_spikes as f64 - out_a.pe_minus_spikes as f64;
            self.w_ab += self.learning_rate * delayed_a * signed_b;
            self.w_ba += self.learning_rate * delayed_b * signed_a;
            // Clip to non-negative; predictions are excitatory in this framing.
            self.w_ab = self.w_ab.max(0.0);
            self.w_ba = self.w_ba.max(0.0);
        }

        // Advance ring buffers: pop oldest, push newest.
        self.a_trace_history.pop_front();
        self.b_trace_history.pop_front();
        self.a_trace_history.push_back(self.a_trace);
        self.b_trace_history.push_back(self.b_trace);

        TwoUnitOutput { a: out_a, b: out_b }
    }
}

impl Default for ApparentMotion {
    fn default() -> Self {
        Self::new()
    }
}

/// Build an alternating apparent-motion stimulus: A on for `on_ms`, then
/// gap for `gap_ms`, then B on for `on_ms`, then gap for `gap_ms`, repeating.
pub fn alternating_stimulus(t_ms: f64, on_ms: f64, gap_ms: f64) -> (f64, f64) {
    let cycle = 2.0 * (on_ms + gap_ms);
    let phase = t_ms.rem_euclid(cycle);
    if phase < on_ms {
        (1.0, 0.0)
    } else if phase < on_ms + gap_ms {
        (0.0, 0.0)
    } else if phase < 2.0 * on_ms + gap_ms {
        (0.0, 1.0)
    } else {
        (0.0, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f64 = 0.1;
    const ON_MS: f64 = 40.0;
    const GAP_MS: f64 = 60.0;
    // A-onset to B-onset = ON_MS + GAP_MS = 100 ms — use that as the delay.
    const DELAY_MS: f64 = 100.0;

    fn run_alternating(net: &mut ApparentMotion, total_ms: f64) {
        let n_steps = (total_ms / DT) as usize;
        for k in 0..n_steps {
            let t = k as f64 * DT;
            let (s_a, s_b) = alternating_stimulus(t, ON_MS, GAP_MS);
            net.step(s_a, s_b, DT);
        }
    }

    #[test]
    fn cross_weights_grow_with_training() {
        let mut net = ApparentMotion::new_with_delay(DELAY_MS, DT);
        run_alternating(&mut net, 3000.0);
        assert!(net.w_ab > 0.0, "w_ab should grow, got {}", net.w_ab);
        assert!(net.w_ba > 0.0, "w_ba should grow, got {}", net.w_ba);
    }

    #[test]
    fn coupling_off_keeps_cross_weights_zero() {
        let mut net = ApparentMotion::new_with_delay(DELAY_MS, DT);
        net.disable_coupling();
        run_alternating(&mut net, 3000.0);
        assert_eq!(net.w_ab, 0.0);
        assert_eq!(net.w_ba, 0.0);
    }

    /// Falsification test: after learning, presenting only A while B is silent
    /// produces PE− spikes in B (prediction-driven anticipation). Same probe
    /// with coupling OFF produces no such spikes.
    #[test]
    fn probe_shows_prediction_driven_pe_minus_in_b_only_with_coupling() {
        // Train on the alternating stimulus.
        let mut net = ApparentMotion::new_with_delay(DELAY_MS, DT);
        run_alternating(&mut net, 4000.0);
        assert!(net.w_ab > 0.0, "training must establish w_ab, got {}", net.w_ab);

        // Probe with coupling ON: drive A only, count PE− in B over the
        // window when prediction would arrive.
        let probe_steps = ((DELAY_MS + 100.0) / DT) as usize;
        let mut pe_minus_b_on = 0;
        for _ in 0..probe_steps {
            let out = net.step(1.0, 0.0, DT);
            pe_minus_b_on += out.b.pe_minus_spikes;
        }

        // Reset state (keep weights), disable coupling, repeat the probe.
        net.reset();
        net.disable_coupling();
        let mut pe_minus_b_off = 0;
        for _ in 0..probe_steps {
            let out = net.step(1.0, 0.0, DT);
            pe_minus_b_off += out.b.pe_minus_spikes;
        }

        assert!(
            pe_minus_b_on > pe_minus_b_off,
            "prediction-driven PE− in B should exceed control: on={}, off={}",
            pe_minus_b_on,
            pe_minus_b_off
        );
    }
}
