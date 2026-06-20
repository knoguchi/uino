# Cortex Research Notes

Mechanistic findings from the literature that inform the predictive-coding
cortex implementation. Each finding lists its sources. Findings are tagged
**HIGH** or **MEDIUM** confidence based on adversarial verification of 25
extracted claims across 28 primary sources (2015-2025).

## Confirmed primitives

### 1. Canonical microcircuit = hierarchical predictive coding (HIGH)

- **L2/3 superficial pyramidals** encode prediction errors, broadcast feedforward in **gamma**
- **L5/6 deep pyramidals** encode predictions/expectations, sent feedback in **beta**
- Inference proceeds as gradient descent on precision-weighted prediction error (free-energy minimization)

Direct empirical support is stronger for the L2/3-PE half than for the L5/6-predictions half. Most evidence is visual cortex; cross-modal generality is inferred from reviews.

Sources:
- Bastos et al. (2012) *Neuron* — Canonical microcircuits for predictive coding. [PMC3777738](https://pmc.ncbi.nlm.nih.gov/articles/PMC3777738/)
- Bastos et al. (2020) *PNAS* — Predictive routing across visual cortex. [10.1073/pnas.2014868117](https://www.pnas.org/doi/10.1073/pnas.2014868117)
- Michalareas et al. (2016) *Neuron* — Gamma-feedforward / beta-feedback spectral asymmetry. [PMC4871751](https://pmc.ncbi.nlm.nih.gov/articles/PMC4871751/)

### 2. PE+ and PE− as separate non-negative populations (HIGH)

Prediction errors are represented by **two** populations, not one signed population:

- **PE+** = max(stimulus − prediction, 0), wired with bottom-up E / top-down I
- **PE−** = max(prediction − stimulus, 0), wired oppositely

Reason: L2/3 baseline firing rates are too low (Niell & Stryker 2008) to encode signed errors as deviations from a high baseline. Empirically supported by Keller-lab mouse V1 PE+/PE− neurons, PLOS Bio auditory omission nPE neurons, Cerebral Cortex 2025 co-occurring PE+/PE− signals.

Sources:
- [eLife 95127](https://elifesciences.org/articles/95127) — formalizes UPE+ / UPE−
- [Frontiers in Computational Neuroscience 2024](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2024.1338280/full) — SNN-PC implementation

### 3. Spiking implementation blueprint (HIGH)

A concrete biologically-grounded spiking predictive-coding network uses:

- **AdEx neurons** (Brette & Gerstner 2005)
- **AMPA synapse current** with rise time ~5 ms
- **NMDA synapse current** with decay time ~50 ms (voltage-dependent Mg²⁺ block)
- **Local Hebbian plasticity** approximating NMDA-Ca²⁺-dependent LTP/LTD via filtered spike traces as Ca²⁺ proxies
- **L1 (Laplacian-prior) weight decay** — encourages sparse weights, no backpropagation
- **Plasticity only on inter-areal weights** (W_l,l+1); intra-areal recurrent weights fixed

N'dri/Triesch/Ororbia et al. (2024) survey organizes spiking PC implementations into three classes: explicit error neurons, PE encoded in membrane potentials, implicit PE encoding.

Sources:
- N'dri/Triesch/Ororbia et al. (2024) — Spiking predictive coding survey. [arXiv:2409.05386](https://arxiv.org/pdf/2409.05386)
- Dora & Pennartz et al. (2024) — concrete SNN-PC demonstrator. [Frontiers Comp Neurosci](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2024.1338280/full)
- Brette & Gerstner (2005) — AdEx model (original).

### 4. Manifold geometry as the measurement target (HIGH)

Cortical population activity occupies low-dimensional manifolds. Linear separability of task-variable manifolds is governed by three measurable geometric quantities:

- **dimension**
- **radius**
- **correlation structure**

This gives a **modality-general, quantitative target** for evaluating whether a learning hierarchy is doing useful work. Watch how object/category manifolds separate along these three numbers.

Sources:
- Chung & Cohen (2018) *Phys Rev X* — Classification and geometry of general perceptual manifolds. [arXiv:1710.06487](https://arxiv.org/abs/1710.06487)
- Cohen et al. (2020) — manifold capacity extensions.
- Stringer et al. (2019) *Nature* — high-dimensional geometry of V1.
- Manifold review 2023. [PMC10695674](https://pmc.ncbi.nlm.nih.gov/articles/PMC10695674/)
- Task-variable manifold review 2024. [PMC11058347](https://pmc.ncbi.nlm.nih.gov/articles/PMC11058347/)

### 5. Attractor motifs are reused across functions (HIGH)

Same circuit pattern, different content:

- **Discrete WTA attractors**: per-choice excitatory groups + shared inhibition → action selection, perceptual decisions
- **Ring/bump attractors**: strong local E + uniform I → head direction (HD), continuous working memory, PFC spatial WM

Sources:
- Wang (2002), Wong & Wang (2006) — biophysical decision model
- Wimmer et al. (2014) *Nat Neurosci* — monkey PFC bump dynamics. [nn.3645](https://www.nature.com/articles/nn.3645)
- Compte & Wang — foundational WM modeling
- Kim et al. (2017), Seelig & Jayaraman — Drosophila ring attractor
- Khona & Fiete (2022) — continuous attractor motif review. [arXiv:2112.03978](https://arxiv.org/abs/2112.03978)

### 6. PFC working memory: two complementary mechanisms (HIGH)

- **Orthogonal-subspace coding**: stimulus identity sits in a stable mnemonic subspace (often a ring for continuous variables) while time-varying computation evolves in an orthogonal subspace. Ongoing dynamics don't corrupt the mnemonic content.
- **Activity-silent traces**: between trials, stimulus info becomes undecodable from firing rates but remains in spike-synchrony patterns. Reactivates from synchrony when new input arrives.

Sources:
- Murray et al. (2017) *PNAS* — Stable population coding + heterogeneous dynamics. [10.1073/pnas.1619449114](https://www.pnas.org/doi/10.1073/pnas.1619449114)
- Spaak et al. (2017). [PMC5511881](https://pmc.ncbi.nlm.nih.gov/articles/PMC5511881/)
- Barbosa, Stein et al. (2020) *Nat Neurosci* — Activity-silent WM (monkey PFC + human EEG/TMS replication). [s41593-020-0644-4](https://www.nature.com/articles/s41593-020-0644-4)

## Medium-confidence findings

### 7. Fast feedforward "gist" pathway (MEDIUM)

A sparse random projection (~5% connectivity, e.g. 784→16 units) supplies coarse high-level priors to each cortical area, modeling the ~150 ms ventral-stream feedforward sweep (Thorpe et al. 1996). In SNN-PC models this improved classification (p<0.001) without harming reconstruction.

Single-paper proposal but biologically grounded.

Source: [Frontiers in Computational Neuroscience 2024](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2024.1338280/full)

### 8. E+PV+SOM canonical microcircuit motif (MEDIUM)

Building block of the microcircuit: excitatory pyramidals + parvalbumin (PV) + somatostatin (SOM) interneurons with defined roles. SOM-mediated gain modulation can switch the circuit between sampling regimes.

Sources:
- Yamauchi et al. (2025) *Neuroscience Research*. [10.1016/j.neures.2025...](https://www.sciencedirect.com/science/article/pii/S0168010225001853)
- Hertag & Sprekeler (2022) *PNAS* — PE+/PE− microcircuit formation
- Anatomical scaffold: Dura-Bernal 2023, Billeh 2020, Thomson 2007, Tremblay 2016

## Refuted — do NOT build

Adversarial verification killed the following popular claims (≥2/3 refute votes). Treat as still-open hypotheses, not load-bearing:

- **SST = mean, PV = variance** as a precision-weighting scheme (0-3 refuted)
- **Uncertainty-modulated PEs** with inverse-variance scaling Δμ̂ ∝ (1/σ²)(s−μ̂) (0-3 refuted)
- **Single canonical microcircuit doing both perception AND motor control** via inference/control duality (0-3 refuted; 1-2 for sensory+motor unified circuit)
- **Specific attractor-motif-to-function map** (fixed points = memory, line attractors = integration, limit cycles = sequences) — too neat to survive (0-3 refuted)
- **E/I balance + structured feedforward weights alone** producing canonical RF properties without plasticity (1-2 refuted)
- **Context-switching via geometric transformations** (rotation + translation of decision boundaries) as a primitive (1-2 refuted)
- **Distinct interneuron types implement distinct computational roles** (SST = subtractive, PV = divisive in a clean split) (0-3 refuted)

## Open questions (not yet vetted)

1. How does the L5/6-as-predictions half map empirically in non-visual cortex? Direct evidence is thinner than L2/3-as-PE.
2. Cortex-hippocampus consolidation and engram mechanisms at a spiking-implementable level were under-covered in this pass. Needs a follow-up.
3. Basal ganglia / dopamine action selection — the spiking interface between cortical WTA decision circuits and BG gating — was under-covered. Needs a follow-up.
4. How should precision/uncertainty weighting be implemented in a spiking circuit, given that the popular SST/PV proposals were refuted? Open.

## Statistics

- 5 search angles, 28 primary sources fetched
- 134 candidate claims extracted
- 25 verified adversarially (3-vote panel each)
- 17 confirmed, 8 killed
- Date: 2026-06-20
