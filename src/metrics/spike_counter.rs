//! Spike counter for the project's compass metric: prediction error
//! spikes per inference.
//!
//! Spikes are recorded against named sources ("pe_plus", "pe_minus", etc.).
//! Per-inference snapshots are taken explicitly via `snapshot()` so the
//! caller controls inference boundaries.

use std::collections::HashMap;

/// Snapshot of per-source spike counts at one moment.
#[derive(Clone, Debug, Default)]
pub struct Snapshot {
    pub counts: HashMap<String, usize>,
}

impl Snapshot {
    pub fn total(&self) -> usize {
        self.counts.values().sum()
    }

    pub fn count(&self, source: &str) -> usize {
        self.counts.get(source).copied().unwrap_or(0)
    }
}

#[derive(Clone, Debug, Default)]
pub struct SpikeCounter {
    counts: HashMap<String, usize>,
    history: Vec<Snapshot>,
}

impl SpikeCounter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one spike from `source`.
    pub fn record(&mut self, source: &str) {
        *self.counts.entry(source.to_string()).or_insert(0) += 1;
    }

    /// Record N spikes from `source` (useful for batched updates).
    pub fn record_n(&mut self, source: &str, n: usize) {
        *self.counts.entry(source.to_string()).or_insert(0) += n;
    }

    pub fn count(&self, source: &str) -> usize {
        self.counts.get(source).copied().unwrap_or(0)
    }

    pub fn total(&self) -> usize {
        self.counts.values().sum()
    }

    /// Store current counts as one inference snapshot and reset live counts.
    pub fn snapshot(&mut self) -> Snapshot {
        let snap = Snapshot { counts: self.counts.clone() };
        self.history.push(snap.clone());
        self.counts.clear();
        snap
    }

    /// Reset live counts without snapshotting.
    pub fn reset(&mut self) {
        self.counts.clear();
    }

    pub fn history(&self) -> &[Snapshot] {
        &self.history
    }

    /// Compass-metric helper: total prediction-error spikes from a snapshot,
    /// summing whichever PE sources the user has recorded. Convention:
    /// sources whose name starts with `"pe_"` are prediction errors.
    pub fn pe_total(snap: &Snapshot) -> usize {
        snap.counts
            .iter()
            .filter(|(k, _)| k.starts_with("pe_"))
            .map(|(_, v)| v)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_counter_has_zero_total() {
        let c = SpikeCounter::new();
        assert_eq!(c.total(), 0);
        assert_eq!(c.count("anything"), 0);
    }

    #[test]
    fn records_by_source() {
        let mut c = SpikeCounter::new();
        c.record("pe_plus");
        c.record("pe_plus");
        c.record("pe_minus");
        c.record("other");
        assert_eq!(c.count("pe_plus"), 2);
        assert_eq!(c.count("pe_minus"), 1);
        assert_eq!(c.count("other"), 1);
        assert_eq!(c.total(), 4);
    }

    #[test]
    fn snapshot_resets_live_counts_and_appends_history() {
        let mut c = SpikeCounter::new();
        c.record_n("pe_plus", 5);
        c.record_n("pe_minus", 3);
        let s1 = c.snapshot();
        assert_eq!(s1.total(), 8);
        assert_eq!(c.total(), 0, "live counts must reset after snapshot");

        c.record_n("pe_plus", 2);
        c.snapshot();
        assert_eq!(c.history().len(), 2);
        assert_eq!(c.history()[0].count("pe_plus"), 5);
        assert_eq!(c.history()[1].count("pe_plus"), 2);
    }

    #[test]
    fn pe_total_sums_pe_prefixed_sources() {
        let mut c = SpikeCounter::new();
        c.record_n("pe_plus", 7);
        c.record_n("pe_minus", 3);
        c.record_n("other_population", 99);
        let snap = c.snapshot();
        assert_eq!(SpikeCounter::pe_total(&snap), 10);
        assert_eq!(snap.total(), 109);
    }

    #[test]
    fn reset_clears_without_history() {
        let mut c = SpikeCounter::new();
        c.record("pe_plus");
        c.reset();
        assert_eq!(c.total(), 0);
        assert_eq!(c.history().len(), 0);
    }
}
