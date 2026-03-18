use ndarray::{Array1, Array2, Axis, s};

// --- STUBS for Spectral Features ---
pub fn chroma_stft() { unimplemented!() }
pub fn chroma_cqt() { unimplemented!() }
pub fn chroma_cens() { unimplemented!() }
pub fn chroma_vqt() { unimplemented!() }
pub fn melspectrogram() { unimplemented!() }
pub fn mfcc() { unimplemented!() }
pub fn rms() { unimplemented!() }
pub fn spectral_centroid() { unimplemented!() }
pub fn spectral_bandwidth() { unimplemented!() }
pub fn spectral_contrast() { unimplemented!() }
pub fn spectral_flatness() { unimplemented!() }
pub fn spectral_rolloff() { unimplemented!() }
pub fn zero_crossing_rate() { unimplemented!() }

// --- TONNETZ IMPLEMENTATION ---
/// Compute the tonal centroid features (tonnetz)
/// 
/// Takes a pre-computed constant-Q chromagram `chroma` [shape=(12, t)] 
pub fn tonnetz(chroma: &Array2<f64>) -> Array2<f64> {
    // dim_map = linspace(0, 12, 12, endpoint=False)
    let mut dim_map = Array1::<f64>::zeros(12);
    for i in 0..12 {
        dim_map[i] = i as f64;
    }
    
    let scale = Array1::from_vec(vec![7.0/6.0, 7.0/6.0, 3.0/2.0, 3.0/2.0, 2.0/3.0, 2.0/3.0]);
    
    // V = outer(scale, dim_map) -> shape (6, 12)
    let mut v = Array2::<f64>::zeros((6, 12));
    for i in 0..6 {
        for j in 0..12 {
            v[[i, j]] = scale[i] * dim_map[j];
        }
    }
    
    // Even rows compute sin() in python version via V[::2] -= 0.5 then cos(pi * V)
    for i in (0..6).step_by(2) {
        for j in 0..12 {
            v[[i, j]] -= 0.5;
        }
    }
    
    let r = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0, 0.5, 0.5]);
    let mut phi = Array2::<f64>::zeros((6, 12));
    for i in 0..6 {
        for j in 0..12 {
            phi[[i, j]] = r[i] * (std::f64::consts::PI * v[[i, j]]).cos();
        }
    }
    
    // Compute normalization factor for each frame
    let mut chroma_norm = chroma.clone();
    for mut col in chroma_norm.axis_iter_mut(Axis(1)) {
        let sum: f64 = col.sum().abs();
        if sum > 1e-10 {
            col.mapv_inplace(|x| x / sum);
        }
    }
    
    // Do the transform to tonnetz: (6, 12) x (12, t) -> (6, t)
    phi.dot(&chroma_norm)
}


// --- POLY_FEATURES IMPLEMENTATION ---

/// Get coefficients of fitting an nth-order polynomial to the columns of a spectrogram.
pub fn poly_features(s: &Array2<f64>, freq: &Array1<f64>, order: usize) -> Array2<f64> {
    let n_freqs = s.nrows();
    let n_frames = s.ncols();
    
    let mut coeffs = Array2::<f64>::zeros((order + 1, n_frames));
    
    // Construct Vandermonde matrix A for least squares: shape (n_freqs, order + 1)
    let mut a = Array2::<f64>::zeros((n_freqs, order + 1));
    for i in 0..n_freqs {
        let x = freq[i];
        for j in 0..=order {
            a[[i, j]] = x.powi((order - j) as i32);
        }
    }
    
    // Solve A w = y frame by frame
    // Normal equations: (A^T A) w = A^T y
    let a_t = a.t();
    let a_t_a = a_t.dot(&a);
    
    let inv_ata = invert_matrix(&a_t_a).expect("Polynomial fit failed: Matrix is singular");
    
    // Pseudoinverse: pinv = (A^T A)^-1 A^T
    let pinv = inv_ata.dot(&a_t);
    
    for t in 0..n_frames {
        let y = s.column(t);
        let w = pinv.dot(&y);
        for j in 0..=order {
            coeffs[[j, t]] = w[j];
        }
    }
    
    coeffs
}

/// Helper function to invert a square n x n matrix using Gaussian elimination
fn invert_matrix(m: &Array2<f64>) -> Result<Array2<f64>, String> {
    let n = m.nrows();
    if n != m.ncols() { return Err("Matrix must be square".to_string()); }
    
    if n == 1 {
        let mut inv = Array2::zeros((1, 1));
        inv[[0, 0]] = 1.0 / m[[0, 0]];
        return Ok(inv);
    }
    
    let mut a = m.clone();
    let mut inv = Array2::eye(n);
    
    for i in 0..n {
        let mut pivot = i;
        for j in i+1..n {
            if a[[j, i]].abs() > a[[pivot, i]].abs() {
                pivot = j;
            }
        }
        if a[[pivot, i]].abs() < 1e-12 { return Err("Singular matrix".to_string()); }
        
        // Swap rows i and pivot
        if pivot != i {
            for j in 0..n {
                let tmp = a[[i, j]]; a[[i, j]] = a[[pivot, j]]; a[[pivot, j]] = tmp;
                let tmp = inv[[i, j]]; inv[[i, j]] = inv[[pivot, j]]; inv[[pivot, j]] = tmp;
            }
        }
        
        let diag = a[[i, i]];
        for j in 0..n {
            a[[i, j]] /= diag;
            inv[[i, j]] /= diag;
        }
        
        for j in 0..n {
            if i != j {
                let factor = a[[j, i]];
                for k in 0..n {
                    a[[j, k]] -= factor * a[[i, k]];
                    inv[[j, k]] -= factor * inv[[i, k]];
                }
            }
        }
    }
    Ok(inv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    #[test]
    fn test_tonnetz_shapes() {
        // Ported test from test_features.py:test_tonnetz_audio -> asserts shapes bounds
        let frames: [usize; 3] = [4, 10, 50];
        
        for &t in &frames {
            let chroma = Array2::<f64>::ones((12, t));
            let res = tonnetz(&chroma);
            
            assert_eq!(res.nrows(), 6);
            assert_eq!(res.ncols(), t);
            
            // Should not output NaNs on valid data
            for val in res.iter() {
                assert!(!val.is_nan());
            }
        }
    }
    
    #[test]
    fn test_poly_features_synthetic() {
        // Ported test from test_features.py:test_poly_features_synthetic
        let orders: [usize; 3] = [1, 2, 3];
        let n_freqs = 10;
        let mut freq = Array1::<f64>::zeros(n_freqs);
        for i in 0..n_freqs {
            freq[i] = i as f64;
        }
        
        for &order in &orders {
            let mut true_coeffs = vec![0.0; order + 1];
            for j in 0..=order {
                true_coeffs[j] = (j + 1) as f64; // arbitrary monotonic coefficients
            }
            
            let mut s = Array2::<f64>::zeros((n_freqs, 2));
            for t in 0..2 {
                for i in 0..n_freqs {
                    let mut val = 0.0;
                    let x = freq[i];
                    for j in 0..=order {
                        val += true_coeffs[j] * x.powi(j as i32);
                    }
                    s[[i, t]] = val;
                }
            }
            
            let recovered_coeffs = poly_features(&s, &freq, order);
            assert_eq!(recovered_coeffs.nrows(), order + 1);
            
            // Check parameter extraction. Note that `poly_features` returns coefficients from highest order to lowest.
            for t in 0..2 {
                for j in 0..=order {
                    let expected = true_coeffs[order - j];
                    assert!((recovered_coeffs[[j, t]] - expected).abs() < 1e-5,
                            "Failed order {}: expected {}, got {}", order, expected, recovered_coeffs[[j, t]]);
                }
            }
        }
    }
}
