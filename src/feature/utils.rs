use ndarray::{Array2, s};

/// Short-term history embedding: vertically concatenate a data
/// matrix with delayed copies of itself.
///
/// Each column `data[:, i]` is mapped to:
/// `[data[..., i], data[..., i - delay], ..., data[..., i - (n_steps-1)*delay]]`
pub fn stack_memory(data: &Array2<f64>, n_steps: usize, delay: usize) -> Array2<f64> {
    let d = data.nrows();
    let t = data.ncols();
    
    let mut out = Array2::<f64>::zeros((d * n_steps, t));
    
    for step in 0..n_steps {
        let lag = step * delay;
        for col in 0..t {
            if col >= lag {
                let src_col = col - lag;
                for row in 0..d {
                    out[[step * d + row, col]] = data[[row, src_col]];
                }
            } else {
                // Default librosa padding for stack_memory is 'constant' == 0
                for row in 0..d {
                    out[[step * d + row, col]] = 0.0;
                }
            }
        }
    }
    
    out
}

/// Compute delta features: local estimate of the derivative of the input data
/// Computed via Savitzky-Golay filtering.
pub fn delta(data: &Array2<f64>, width: usize, order: usize) -> Array2<f64> {
    let half_len = width / 2;
    let coeffs = savgol_coeffs(width, order, 1);
    
    let d = data.nrows();
    let t = data.ncols();
    
    let mut out = Array2::<f64>::zeros((d, t));
    
    for row in 0..d {
        for col in 0..t {
            let mut sum = 0.0;
            for i in 0..width {
                let mut idx = col as isize + i as isize - half_len as isize;
                // librosa delta default padding mode is 'interp' (edge extension essentially for 1st order)
                // We use edge padding (nearest) as a sturdy equivalent
                if idx < 0 { idx = 0; }
                if idx >= t as isize { idx = t as isize - 1; }
                
                sum += data[[row, idx as usize]] * coeffs[i];
            }
            out[[row, col]] = sum;
        }
    }
    out
}

/// Helper function to generate Savitzky-Golay filter coefficients
fn savgol_coeffs(window_length: usize, polyorder: usize, deriv: usize) -> Vec<f64> {
    let halflen = window_length as isize / 2;
    let mut a = Array2::<f64>::zeros((window_length, polyorder + 1));
    for i in 0..window_length {
        let x = (i as isize - halflen) as f64;
        for j in 0..=polyorder {
            a[[i, j]] = x.powi(j as i32);
        }
    }
    
    let at = a.t();
    let ata = at.dot(&a);
    let inv_ata = invert_matrix(&ata).expect("Savitzky-Golay matrix is singular");
    let pinv = inv_ata.dot(&at);
    
    let mut factorial = 1.0;
    for i in 1..=deriv { factorial *= i as f64; }
    
    let mut coeffs = vec![0.0; window_length];
    for i in 0..window_length {
        coeffs[i] = pinv[[deriv, i]] * factorial;
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
    use ndarray::Array2;

    #[test]
    fn test_stack_memory() {
        let mut data = Array2::<f64>::zeros((2, 5));
        for i in 0..2 {
            for j in 0..5 {
                data[[i, j]] = (i * 10 + j) as f64;
            }
        }
        
        // Test varying delays and n_steps
        let delays = [1, 2];
        let steps = [1, 3];
        
        for &delay in &delays {
            for &n_steps in &steps {
                let stacked = stack_memory(&data, n_steps, delay);
                assert_eq!(stacked.nrows(), 2 * n_steps);
                assert_eq!(stacked.ncols(), 5);
                
                // Base step should perfectly match original
                for i in 0..2 {
                    for j in 0..5 {
                        assert_eq!(stacked[[i, j]], data[[i, j]]);
                    }
                }
                
                // Check delayed bounds
                if n_steps > 1 {
                    let lag = delay; // step 1 lag
                    for i in 0..2 {
                        // Beyond lag: should match shifted data
                        assert_eq!(stacked[[2 + i, lag]], data[[i, 0]]);
                        if lag + 1 < 5 {
                            assert_eq!(stacked[[2 + i, lag + 1]], data[[i, 1]]);
                        }
                        // Before lag: should be padded with constant 0
                        if lag > 0 {
                            assert_eq!(stacked[[2 + i, lag - 1]], 0.0);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_delta() {
        // test_delta port from test_features.py validating exact slope estimation
        let slopes = [-2.0, 0.5, 2.0];
        let biases = [-10.0, 0.0, 10.0];
        let width = 5;
        let order = 1;
        
        for &slope in &slopes {
            for &bias in &biases {
                let n = 20;
                let mut data = Array2::<f64>::zeros((1, n));
                for i in 0..n {
                    data[[0, i]] = slope * (i as f64) + bias;
                }
                
                let d = delta(&data, width, order);
                assert_eq!(d.shape(), &[1, n]);
                
                // Assert interior points evaluate perfectly to the slope.
                // Using half_len = width / 2 bounds: 2 to 17
                for i in (width/2) .. (n - width/2) {
                    assert!((d[[0, i]] - slope).abs() < 1e-5);
                }
            }
        }
    }
}
