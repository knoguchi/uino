# Findings

Measured results from the architecture. Updated as experiments accumulate.

## MNIST classification (2026-06-20)

10-class digit classification through the full pipeline:
`image → retinula → RetinaBridge → MultiStage cortex → nearest-centroid readout`.

100 training images, 50 test images, 2 epochs, brief per-image presentation
(~50 ms retina + ~30 ms cortex). See `examples/mnist.rs`.

| Configuration | Accuracy | vs chance (10%) |
|---|---|---|
| Retinula features only (no cortex) | 30% | 3.0× |
| Deep cortex 16→8→4, top stage = 16 cells | 6% | 0.6× |
| Shallow cortex 16→8, top = 64 cells | 22% | 2.2× |
| Shallow + 3×3 input RFs (plastic, Hebbian) | 22% (18% → 22% over 2 epochs) | 2.2× |
| Shallow + input RFs + lateral inhibition (strength 0.05) | 8% | 0.8× |
| Shallow + input RFs + lateral inhibition (strength 0.3) | 12% | 1.2× |

### What the numbers mean

- **Front-end alone gets 30%**: the retinula+bridge has plenty of class-discriminative signal in its 256-dim mean-rate vector.
- **Deep cortex drops to 6%** (below chance): aggressive spatial pooling (16 → 8 → 4 with 4×4 final stage) collapses different digits into similar top-stage representations. The 16-cell top is too compressed for 10 classes.
- **Shallow cortex with input RFs gets 22%**: 64-cell top retains enough information. Learning curve (18% → 22% across 2 epochs) confirms the system is actually consolidating exposure into weights, not just relaying input.
- **Lateral inhibition hurts as currently implemented**: divisive normalization on raw cell input uniformly suppresses all cells early in training (when their weights are similar), reducing total firing and learning signal. Need to either operate on PE output, use winner-take-all, or activate only after symmetry has broken.

### Architectural notes

- The architecture as built now: spiking PE+/PE− populations per cell, plastic cross-stage weights (Hebbian-Ca), plastic input RFs (rate-based Hebbian + L1 decay), spatial pooling between stages. Save/load via serde.
- The compass metric (PE spikes per inference) drops as the cortex's predictions catch up to its inputs — verified in synthetic tasks but not the bottleneck for MNIST.
- Multi-timescale eta gradient (lower stages fast, upper slow) verified to produce noise-robust class discrimination on synthetic input — orthogonal mechanism from MNIST results.

### What's clearly needed for higher accuracy

1. **Lateral inhibition that actually breaks symmetry.** The current implementation hurts. Either move it post-PE (winner-take-all on firing) or make it sparse-only-activates-when-cells-differ.
2. **More training.** 100 images × 2 epochs is a thin diet. Real biology sees the same patterns thousands of times across infancy.
3. **Larger / different top representation.** 64 cells worked. More cells, or non-pooled stages, could carry more.
4. **Better readout than nearest-centroid.** Linear classifier on top signatures would extract more.
5. **Sparse activation.** Real V1 has ~1-5% of cells active at once; ours has many more. Sparsity drives feature specialization.

### Context for these numbers

In published biologically-grounded spiking neural networks (no backprop, local learning rules), MNIST accuracy in the 50–80% range is the common ceiling. Industry CNN performance (99%+) uses backpropagation and many architectural tricks our system intentionally avoids. 22% with this architecture is in the right ballpark for "spiking + local learning + no backprop + small training set" but well below what proper feature competition would achieve.
