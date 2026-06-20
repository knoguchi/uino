//! Manifold geometry metrics for population activity.
//!
//! Per the research notes, three geometric quantities govern manifold
//! separability: dimension, radius, and correlation structure. This module
//! computes them directly from samples without forming an SVD.
//!
//!   centroid   = sample mean
//!   radius     = RMS distance from centroid = sqrt(tr(C))
//!   dimension  = participation ratio = tr(C)² / tr(C²)
//!
//! Separability between two classes is centroid-distance / mean radius —
//! the simplest signal-to-noise index. Higher = more separable.

/// Per-class manifold statistics.
#[derive(Clone, Debug)]
pub struct ClassStats {
    pub centroid: Vec<f64>,
    pub radius: f64,
    /// Participation ratio — effective dimensionality of the class manifold.
    pub dimension: f64,
}

/// Pairwise separability between two classes.
#[derive(Clone, Debug)]
pub struct Separability {
    pub centroid_distance: f64,
    pub mean_radius: f64,
    /// centroid_distance / mean_radius — signal-to-noise.
    pub index: f64,
}

/// Compute manifold statistics for one class of samples.
/// Returns `None` if samples is empty or shapes are inconsistent.
pub fn class_stats(samples: &[Vec<f64>]) -> Option<ClassStats> {
    if samples.is_empty() {
        return None;
    }
    let d = samples[0].len();
    if d == 0 || samples.iter().any(|s| s.len() != d) {
        return None;
    }
    let n = samples.len() as f64;

    let mut centroid = vec![0.0; d];
    for s in samples {
        for i in 0..d {
            centroid[i] += s[i];
        }
    }
    for c in &mut centroid {
        *c /= n;
    }

    // Centered samples and covariance C = (1/n) X^T X where X is centered.
    // We compute tr(C) and tr(C²) directly.
    // tr(C) = sum_i (1/n) sum_k X[k,i]²
    // tr(C²) = sum_{i,j} C[i,j]² where C[i,j] = (1/n) sum_k X[k,i] X[k,j]
    //
    // tr(C²) via samples without forming D×D matrix:
    //   tr(C²) = (1/n²) sum_{a,b} (sum_i X[a,i] X[b,i])² = (1/n²) sum_{a,b} dot(X_a, X_b)²
    //
    let mut centered: Vec<Vec<f64>> = Vec::with_capacity(samples.len());
    for s in samples {
        let mut row = vec![0.0; d];
        for i in 0..d {
            row[i] = s[i] - centroid[i];
        }
        centered.push(row);
    }

    let mut trace_c = 0.0;
    for row in &centered {
        for &x in row {
            trace_c += x * x;
        }
    }
    trace_c /= n;

    // tr(C²) via O(N²·D) inner products. Acceptable for moderate sample counts.
    let mut trace_c2 = 0.0;
    for a in 0..centered.len() {
        for b in 0..centered.len() {
            let mut dot = 0.0;
            for i in 0..d {
                dot += centered[a][i] * centered[b][i];
            }
            trace_c2 += dot * dot;
        }
    }
    trace_c2 /= (n * n) as f64;

    let radius = trace_c.sqrt();
    let dimension = if trace_c2 > 1e-15 {
        trace_c * trace_c / trace_c2
    } else {
        // Degenerate: all samples identical → dimension undefined; report 0.
        0.0
    };

    Some(ClassStats { centroid, radius, dimension })
}

/// Compute separability between two classes' samples.
pub fn separability(a: &[Vec<f64>], b: &[Vec<f64>]) -> Option<Separability> {
    let sa = class_stats(a)?;
    let sb = class_stats(b)?;
    if sa.centroid.len() != sb.centroid.len() {
        return None;
    }

    let mut dist2 = 0.0;
    for i in 0..sa.centroid.len() {
        let d = sa.centroid[i] - sb.centroid[i];
        dist2 += d * d;
    }
    let centroid_distance = dist2.sqrt();
    let mean_radius = 0.5 * (sa.radius + sb.radius);
    let index = if mean_radius > 1e-15 {
        centroid_distance / mean_radius
    } else {
        f64::INFINITY
    };

    Some(Separability { centroid_distance, mean_radius, index })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cluster(center: &[f64], spread: f64, n: usize, seed: u64) -> Vec<Vec<f64>> {
        // Deterministic pseudo-random cluster around center with `spread` half-width.
        let mut samples = Vec::with_capacity(n);
        let mut state = seed;
        let mut rand = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f64 / u64::MAX as f64) * 2.0 - 1.0
        };
        for _ in 0..n {
            let s: Vec<f64> = center.iter().map(|c| c + spread * rand()).collect();
            samples.push(s);
        }
        samples
    }

    #[test]
    fn centroid_of_zero_cluster_is_origin() {
        let samples = cluster(&[0.0, 0.0, 0.0], 0.0, 100, 1);
        let s = class_stats(&samples).unwrap();
        for &c in &s.centroid {
            assert!(c.abs() < 1e-12, "centroid {:?} should be origin", s.centroid);
        }
        assert!(s.radius < 1e-12, "zero-spread cluster radius should be 0");
    }

    #[test]
    fn radius_scales_with_spread() {
        let tight = cluster(&[0.0, 0.0], 0.1, 200, 7);
        let loose = cluster(&[0.0, 0.0], 1.0, 200, 7);
        let st = class_stats(&tight).unwrap();
        let sl = class_stats(&loose).unwrap();
        assert!(
            sl.radius > 5.0 * st.radius,
            "loose radius {} should be >>5x tight radius {}",
            sl.radius,
            st.radius
        );
    }

    #[test]
    fn dimension_isotropic_matches_data_dim() {
        // Isotropic D-dim cloud should have PR ≈ D.
        let samples = cluster(&[0.0, 0.0, 0.0, 0.0], 1.0, 500, 42);
        let s = class_stats(&samples).unwrap();
        assert!(s.dimension > 3.0, "isotropic 4D cluster PR should be ~4, got {}", s.dimension);
    }

    #[test]
    fn dimension_one_for_rank_one_data() {
        // All samples on the line (t, 2t, 3t) — rank 1.
        let mut samples = Vec::new();
        for i in 0..100 {
            let t = i as f64;
            samples.push(vec![t, 2.0 * t, 3.0 * t]);
        }
        let s = class_stats(&samples).unwrap();
        assert!(s.dimension < 1.2, "rank-1 data PR should be ~1, got {}", s.dimension);
    }

    #[test]
    fn well_separated_clusters_have_high_separability() {
        let a = cluster(&[0.0, 0.0], 0.1, 100, 1);
        let b = cluster(&[10.0, 10.0], 0.1, 100, 2);
        let sep = separability(&a, &b).unwrap();
        assert!(sep.index > 10.0, "well-separated: expected high index, got {}", sep.index);
    }

    #[test]
    fn overlapping_clusters_have_low_separability() {
        let a = cluster(&[0.0, 0.0], 1.0, 100, 1);
        let b = cluster(&[0.5, 0.5], 1.0, 100, 2);
        let sep = separability(&a, &b).unwrap();
        assert!(sep.index < 2.0, "overlapping: expected low index, got {}", sep.index);
    }

    #[test]
    fn empty_samples_return_none() {
        assert!(class_stats(&[]).is_none());
        assert!(separability(&[], &[]).is_none());
    }

    #[test]
    fn dimension_mismatch_returns_none() {
        let samples = vec![vec![0.0, 1.0], vec![1.0, 2.0, 3.0]];
        assert!(class_stats(&samples).is_none());
    }
}
