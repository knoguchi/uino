//! MNIST classification demo through the uino pipeline:
//!
//!   image → retinula → RetinaBridge → MultiStage (with plasticity) → readout
//!
//! The "memorize" mechanism is Hebbian-Ca on cross-stage feedforward weights
//! plus per-class centroid accumulation. The system starts at chance and
//! improves with repeated exposure — like a baby learning what each digit
//! looks like after seeing many examples.
//!
//! ## Setup
//!
//! Download MNIST IDX files from https://yann.lecun.com/exdb/mnist/ (or any
//! mirror) and put them in `mnist/` next to the project root:
//!
//!   mnist/train-images-idx3-ubyte
//!   mnist/train-labels-idx1-ubyte
//!   mnist/t10k-images-idx3-ubyte
//!   mnist/t10k-labels-idx1-ubyte
//!
//! Then run:
//!
//!     cargo run --release --example mnist
//!
//! Note: this is a demonstration, not a benchmark. The architecture has no
//! orientation detectors, so MNIST accuracy will be far below what a CNN
//! achieves. The point is to show the learning curve climbing as the cortex
//! consolidates exposure.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use retinula::Retina;
use uino::bridge::{Channel, RetinaBridge};
use uino::microcircuit::MultiStage;

// Quick-glance training: brief per-image presentation, small training set,
// single pass. The point is to verify the system learns from short exposure,
// not to drive accuracy with sheer exposure time.
const CELLS: usize = 16;
const N_CLASSES: usize = 10;
const N_TRAIN: usize = 100; // 10 per class
const N_TEST: usize = 50;
const EPOCHS: usize = 2;
const PRESENT_STEPS: usize = 200; // ~20 ms cortex settle per image
const MEASURE_STEPS: usize = 100; // ~10 ms cortex measure per image
const DT_MS: f64 = 0.1;
const RETINA_DURATION_S: f64 = 0.05; // 50 ms retinal "glance"

fn read_idx(path: &Path) -> Result<(Vec<u32>, Vec<u8>), String> {
    let mut f = File::open(path).map_err(|e| format!("open {}: {}", path.display(), e))?;
    let mut header = [0u8; 4];
    f.read_exact(&mut header).map_err(|e| e.to_string())?;
    if header[0] != 0 || header[1] != 0 {
        return Err(format!("bad magic in {}", path.display()));
    }
    let n_dims = header[3] as usize;
    let mut dims = Vec::with_capacity(n_dims);
    for _ in 0..n_dims {
        let mut buf = [0u8; 4];
        f.read_exact(&mut buf).map_err(|e| e.to_string())?;
        dims.push(u32::from_be_bytes(buf));
    }
    let mut data = Vec::new();
    f.read_to_end(&mut data).map_err(|e| e.to_string())?;
    Ok((dims, data))
}

fn load_mnist() -> Result<(Vec<Vec<u8>>, Vec<u8>, Vec<Vec<u8>>, Vec<u8>), String> {
    let base = Path::new("mnist");
    let (tr_dims, tr_data) = read_idx(&base.join("train-images-idx3-ubyte"))?;
    let (_, tr_labels) = read_idx(&base.join("train-labels-idx1-ubyte"))?;
    let (te_dims, te_data) = read_idx(&base.join("t10k-images-idx3-ubyte"))?;
    let (_, te_labels) = read_idx(&base.join("t10k-labels-idx1-ubyte"))?;
    let stride = (tr_dims[1] * tr_dims[2]) as usize;
    let train_imgs: Vec<Vec<u8>> = tr_data.chunks(stride).map(|c| c.to_vec()).collect();
    let test_imgs: Vec<Vec<u8>> = te_data.chunks((te_dims[1] * te_dims[2]) as usize).map(|c| c.to_vec()).collect();
    Ok((train_imgs, tr_labels, test_imgs, te_labels))
}

fn make_retina() -> Retina {
    Retina::for_image(28, 28)
        .resolution(CELLS, CELLS)
        .on_cells_only()
        .no_eccentricity()
        .seed(42)
}

/// Run the full pipeline (image bytes → cortex top-stage signature).
fn signature_for(image: &[u8], retina: &mut Retina, cortex: &mut MultiStage) -> Vec<f64> {
    let img: Vec<f64> = image.iter().map(|&b| b as f64 / 255.0).collect();
    let output = retina.simulate(&img, RETINA_DURATION_S);
    let bridge = RetinaBridge::from_retina_output(&output, CELLS, CELLS, DT_MS, 5.0, Channel::On);
    let s = bridge.mean_rates(1.0 / 30.0);
    let top = cortex.stages.len() - 1;
    let n_up = cortex.stages[top].units.len();
    // Settle.
    for _ in 0..PRESENT_STEPS {
        cortex.step(&s, DT_MS);
    }
    // Measure.
    let mut sig = vec![0.0; n_up];
    for _ in 0..MEASURE_STEPS {
        let out = cortex.step(&s, DT_MS);
        for i in 0..n_up {
            sig[i] += out.pe_plus[top][i] as f64 - out.pe_minus[top][i] as f64;
        }
    }
    sig
}

fn nearest_class(sig: &[f64], centroids: &[Vec<f64>]) -> usize {
    let mut best = 0usize;
    let mut best_d = f64::INFINITY;
    for (c, centroid) in centroids.iter().enumerate() {
        let mut d = 0.0;
        for i in 0..sig.len() {
            let x = sig[i] - centroid[i];
            d += x * x;
        }
        if d < best_d {
            best_d = d;
            best = c;
        }
    }
    best
}

fn evaluate(
    cortex: &mut MultiStage,
    retina: &mut Retina,
    centroids: &[Vec<f64>],
    test_imgs: &[Vec<u8>],
    test_labels: &[u8],
) -> f64 {
    // Freeze the cortex during testing — held-out images should not update weights.
    cortex.disable_plasticity();
    let mut correct = 0usize;
    for i in 0..test_imgs.len() {
        let sig = signature_for(&test_imgs[i], retina, cortex);
        let pred = nearest_class(&sig, centroids);
        if pred as u8 == test_labels[i] {
            correct += 1;
        }
    }
    cortex.enable_plasticity();
    correct as f64 / test_imgs.len() as f64
}

/// Skip-cortex baseline: classify directly on retinula+bridge mean rates.
/// Tells us whether the visual front-end produces class-separable features
/// at all, independent of cortex training.
fn baseline_accuracy(retina: &mut Retina, train_imgs: &[Vec<u8>], train_labels: &[u8], test_imgs: &[Vec<u8>], test_labels: &[u8]) -> f64 {
    let feature_for = |retina: &mut Retina, img: &[u8]| -> Vec<f64> {
        let img_f64: Vec<f64> = img.iter().map(|&b| b as f64 / 255.0).collect();
        let output = retina.simulate(&img_f64, RETINA_DURATION_S);
        let bridge = RetinaBridge::from_retina_output(&output, CELLS, CELLS, DT_MS, 5.0, Channel::On);
        bridge.mean_rates(1.0 / 30.0)
    };
    let n_dim = CELLS * CELLS;
    let mut centroids = vec![vec![0.0; n_dim]; N_CLASSES];
    let mut counts = vec![0usize; N_CLASSES];
    for i in 0..train_imgs.len() {
        let f = feature_for(retina, &train_imgs[i]);
        let c = train_labels[i] as usize;
        for j in 0..n_dim {
            centroids[c][j] += f[j];
        }
        counts[c] += 1;
    }
    for c in 0..N_CLASSES {
        if counts[c] > 0 {
            for j in 0..n_dim {
                centroids[c][j] /= counts[c] as f64;
            }
        }
    }
    let mut correct = 0;
    for i in 0..test_imgs.len() {
        let f = feature_for(retina, &test_imgs[i]);
        if nearest_class(&f, &centroids) as u8 == test_labels[i] {
            correct += 1;
        }
    }
    correct as f64 / test_imgs.len() as f64
}

fn main() -> Result<(), String> {
    let (train_imgs, train_labels, test_imgs, test_labels) = load_mnist().map_err(|e| {
        format!(
            "{}\n\nDownload MNIST IDX files from https://yann.lecun.com/exdb/mnist/ and place them under ./mnist/",
            e
        )
    })?;
    let train_imgs = &train_imgs[..N_TRAIN.min(train_imgs.len())];
    let train_labels = &train_labels[..N_TRAIN.min(train_labels.len())];
    let test_imgs = &test_imgs[..N_TEST.min(test_imgs.len())];
    let test_labels = &test_labels[..N_TEST.min(test_labels.len())];

    let mut retina = make_retina();

    // Sanity check: how good is the retinula front-end alone?
    let baseline = baseline_accuracy(&mut retina, train_imgs, train_labels, test_imgs, test_labels);
    println!("Baseline (retinula features, nearest-centroid, no cortex): {:.1}%", 100.0 * baseline);
    println!();

    let mut retina = make_retina(); // reset
    let mut cortex = MultiStage::with_defaults(&[(CELLS, CELLS), (CELLS / 2, CELLS / 2), (CELLS / 4, CELLS / 4)], &[2, 2]);
    cortex.enable_plasticity();
    let n_top = cortex.stages[cortex.stages.len() - 1].units.len();

    // Each epoch: walk through training images once, accumulating per-class
    // centroids from the cortex's top-stage signatures. The cortex's plastic
    // weights consolidate the more often each pattern is presented. After
    // the epoch, classify the held-out test set with nearest-centroid.
    println!("Architecture: retinula({c}×{c}) → cortex {c}→{m}→{t}", c = CELLS, m = CELLS / 2, t = CELLS / 4);
    println!("{} training images, {} test images, {} classes", train_imgs.len(), test_imgs.len(), N_CLASSES);
    println!("Chance accuracy = {:.1}%\n", 100.0 / N_CLASSES as f64);

    let mut centroids: Vec<Vec<f64>> = vec![vec![0.0; n_top]; N_CLASSES];
    let mut counts: Vec<usize> = vec![0; N_CLASSES];

    for epoch in 1..=EPOCHS {
        // Train: present each image, accumulate centroid.
        for i in 0..train_imgs.len() {
            let sig = signature_for(&train_imgs[i], &mut retina, &mut cortex);
            let label = train_labels[i] as usize;
            for j in 0..n_top {
                centroids[label][j] += sig[j];
            }
            counts[label] += 1;
        }
        // Average centroids (running mean).
        let mut snapshot = vec![vec![0.0; n_top]; N_CLASSES];
        for c in 0..N_CLASSES {
            if counts[c] > 0 {
                for j in 0..n_top {
                    snapshot[c][j] = centroids[c][j] / counts[c] as f64;
                }
            }
        }
        // Evaluate.
        let acc = evaluate(&mut cortex, &mut retina, &snapshot, test_imgs, test_labels);
        println!("epoch {}/{}: test accuracy = {:.1}%", epoch, EPOCHS, 100.0 * acc);
    }

    // Persist the trained cortex.
    cortex.save_to_file("mnist_cortex.bin").map_err(|e| e.to_string())?;
    println!("\nSaved trained cortex to mnist_cortex.bin");
    Ok(())
}
