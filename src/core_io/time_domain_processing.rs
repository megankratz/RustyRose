
// dependencies to support scipy,rfft and numpy functionality
// https://librosa.org/doc/latest/core.html#time-domain-processing
use rustfft::FftPlanner;
use num_complex::Complex;
use ndarray::ArrayD;
use ndarray::Axis;
use ndarray::IxDyn;

// ------------------------------------------------------------------------
// AUTOCORRELATE -> uses FFT-based to compute autocorrelation
// parameters: y: input array, max_size: maximum size of the output, axis: axis to compute autocorrelation
// returns: autocorrelation of the input along the specified axis, with size limited by max_size
// ------------------------------------------------------------------------
pub fn autocorrelate(y: &ArrayD<f32>, max_size: Option<usize>, axis: usize) -> Result<ArrayD<f32>, String> {
    if axis >= y.ndim() { return Err("axis out of bounds for array with ".to_string());}
    if y.iter().any(|v| !v.is_finite()) { return Err("non-finite values".to_string());}

    let n = y.shape()[axis];
    let max_size = max_size.unwrap_or(n).min(n);
    let y_swapped = swapaxis_from_front(y, axis);
    let in_shape = y_swapped.shape().to_vec();
    let signal_len = in_shape[0];

    let lane_count: usize = if in_shape.len() == 1 {
        1
    } else {
        in_shape[1..].iter().product()
    };

    let mut out_shape = in_shape.clone();
    out_shape[0] = max_size;
    let mut out = ArrayD::<f32>::zeros(IxDyn(&out_shape));

    let y_2d = y_swapped
        .view()
        .into_shape((signal_len, lane_count))
        .map_err(|e| format!("reshape error: {}", e))?;
    let mut out_2d = out
        .view_mut()
        .into_shape((max_size, lane_count))
        .map_err(|e| format!("reshape error: {}", e))?;
    
    for lane in 0..lane_count {
        let signal: Vec<f32> = y_2d.column(lane).iter().copied().collect();
        let autocorr = autocorrelate_oned(&signal, max_size);
        for i in 0..max_size {
            out_2d[(i, lane)] = autocorr[i];
        }
    }

    Ok(swapaxis_from_front(&out, axis))
}

// autocorrelate_oned - helper function to compute autocorrelation of a 1D signal
fn autocorrelate_oned(y: &[f32], max_size: usize) -> Vec<f32> {
    let n = y.len();
    let max_size = max_size.min(n);

    let n_fft = next_fast_len(2 * n - 1);
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);
    let ifft = planner.plan_fft_inverse(n_fft);

    let mut buffer = vec![Complex{re:0.0, im:0.0}; n_fft];
    for i in 0..n {
        buffer[i].re = y[i];
    }
    fft.process(&mut buffer);
    for c in buffer.iter_mut() {
        let mag2 = c.re * c.re + c.im * c.im;
        c.re = mag2;
        c.im = 0.0;
    }

    ifft.process(&mut buffer);
    let scale = 1.0 / n_fft as f32;
    let mut res = vec![0.0_f32; max_size];
    for i in 0..max_size {
        res[i] = buffer[i].re * scale;
    }
    res
}

// next_fast_len - helper function based on librosa to compute the next power of two for autocorrelate
pub fn next_fast_len(n: usize) -> usize {
    n.next_power_of_two()
}

// -------------------------------------------------------------------------
// LPC -> computes linear predictive coding coefficients for a given input signal
// parameters: y: input array, order: order of the LPC, axis: axis to compute LPC
// returns: LPC coefficients of the input along the specified axis, with size order + 1
// -------------------------------------------------------------------------
pub fn lpc(y: &ArrayD<f32>, order: usize, axis: usize) -> Result<ArrayD<f32>, String> {
    // input validations
    if order == 0 { return Err("order must be an integer > 0".to_string());}
    if axis >= y.ndim() { return Err("axis out of bounds for array with ".to_string());}
    if y.shape()[axis] <= order { return Err("input length must be greater than order".to_string());}
    if y.iter().any(|v| !v.is_finite()) { return Err("non-finite values".to_string());}


    let y_swapped = swapaxis_from_front(y, axis);
    let in_shape = y_swapped.shape().to_vec();
    let n = in_shape[0];
    if n <= order { return Err("input signal is too short".to_string());}

    let mut out_shape = in_shape.clone();
    out_shape[0] = order + 1;
    let mut out = ArrayD::<f32>::zeros(IxDyn(&out_shape));


    let lane_count: usize = if in_shape.len() == 1 {
        1
    } else {
        in_shape[1..].iter().product()
    };

    let y_2d = y_swapped
        .view()
        .into_shape((n, lane_count))
        .map_err(|e| format!("reshape error: {}", e))?;

    let mut out_2d = out
        .view_mut()
        .into_shape((order + 1, lane_count))
        .map_err(|e| format!("reshape error: {}", e))?;

    for lane in 0..lane_count {
        let signal: Vec<f32> = y_2d.column(lane).iter().copied().collect();
        let coeffs = lpc_oned(&signal, order)?;
        for i in 0..=order {
            out_2d[(i, lane)] = coeffs[i];
        }
    }
    Ok(swapaxis_from_front(&out, axis))
}

// lpc_oned - helper function to compute LPC coefficients for a 1D signal
fn lpc_oned(y: &[f32], order: usize) -> Result<Vec<f32>, String> {
    if order == 0 || order >= y.len() || y.iter().any(|&v| !v.is_finite()) { return Err("input error".to_string());}

    let mut ar_coeffs = vec![0.0_f32; order + 1];
    let mut ar_coeffs_prev = vec![0.0_f32; order + 1];
    ar_coeffs[0] = 1.0_f32;
    ar_coeffs_prev[0] = 1.0_f32;


    let mut fwd_pred_error = y[1..].to_vec();
    let mut bwd_pred_error = y[..y.len() - 1].to_vec();
    let mut den = 0.0_f32;
    for i in 0..fwd_pred_error.len() {
        den += fwd_pred_error[i] * fwd_pred_error[i]
            + bwd_pred_error[i] * bwd_pred_error[i];
    }

    let epsilon = f32::EPSILON;
    for i in 0..order {
        let mut reflect_coeff = 0.0_f32;
        for k in 0..fwd_pred_error.len() {
            reflect_coeff += bwd_pred_error[k] * fwd_pred_error[k];
        }
        reflect_coeff *= -2.0_f32;
        reflect_coeff /= den + epsilon;

        std::mem::swap(&mut ar_coeffs, &mut ar_coeffs_prev);
        for j in 1..=i + 1 {
            ar_coeffs[j] = ar_coeffs_prev[j] + reflect_coeff * ar_coeffs_prev[i + 1 - j];
        }
        let old_fwd = fwd_pred_error.clone();
        for k in 0..fwd_pred_error.len() {
            fwd_pred_error[k] = fwd_pred_error[k] + reflect_coeff * bwd_pred_error[k];
            bwd_pred_error[k] = bwd_pred_error[k] + reflect_coeff * old_fwd[k];
        }
        let q = 1.0_f32 - reflect_coeff * reflect_coeff;
        den = q * den
            - bwd_pred_error[bwd_pred_error.len() - 1] * bwd_pred_error[bwd_pred_error.len() - 1]
            - fwd_pred_error[0] * fwd_pred_error[0];

        if !den.is_finite() {
            return Err("numerical error".to_string());
        }
        if fwd_pred_error.len() > 1 {
            fwd_pred_error = fwd_pred_error[1..].to_vec();
            bwd_pred_error = bwd_pred_error[..bwd_pred_error.len() - 1].to_vec();
        }
    }
    Ok(ar_coeffs)
}

// --------------------------------------------------------------------------
// ZERO_CROSSING -> computes zero-crossing points in the input signal along the axis with options for thresholding and zero handling
// parameters: y: input array, threshold: min mag, ref_magnitude: optional ref mag for threshold scaling
// pad: whether to pad the first element of the output with true, zero_pos: whether to treat zeros as positive for crossing detection, 
// axis: axis to compute zero-crossing
// returns: boolean array indicating zero-crossing points along the specified axis, with the same shape as the input
// ----------------------------------------------------------
pub fn zero_crossing(y: &ArrayD<f32>, threshold: f32, ref_magnitude: Option<f32>, pad: bool, zero_pos: bool, axis: usize) -> ArrayD<bool> {
    let threshold = match ref_magnitude {
        Some(mag) => threshold * mag,
        None => threshold,
    };

    let yi = swapaxis_toend(y, axis);
    let mut zi = ArrayD::<bool>::from_elem(yi.raw_dim(), false);
    zc_wrapper(&yi, threshold, zero_pos, &mut zi);

    let last_axis = zi.ndim() - 1;
    for mut lane in zi.lanes_mut(Axis(last_axis)) {
        if !lane.is_empty() {
            lane[0] = pad;
        }
    }
    swapaxis_fromend(&zi, axis)
}

// swapaxis_from_front - helper function to swap the specified axis to the front for processing and then swap back after processing
fn swapaxis_from_front<T: Clone>(arr: &ArrayD<T>, axis: usize) -> ArrayD<T> {
    let ndim = arr.ndim();
    let mut axes: Vec<usize> = (0..ndim).collect();
    axes.swap(0, axis);
    arr.view().permuted_axes(axes).to_owned()
}

// swapaxis_toend - helper function to swap the specified axis to the end for processing and then swap back after processing
pub fn swapaxis_toend(arr: &ArrayD<f32>, axis: usize) -> ArrayD<f32> {
    let ndim = arr.ndim();
    let mut axes: Vec<usize> = (0..ndim).collect();
    axes.swap(axis, ndim - 1);
    arr.view().permuted_axes(axes).to_owned()
}

// swapaxis_fromend - helper function to swap the specified axis from the end back to its original position after processing
pub fn swapaxis_fromend(arr: &ArrayD<bool>, original_axis: usize) -> ArrayD<bool> {
    let ndim = arr.ndim();
    let mut axes: Vec<usize> = (0..ndim).collect();
    axes.swap(original_axis, ndim - 1);
    arr.view().permuted_axes(axes).to_owned()
}

// sign_value - helper function to determine the sign of a value with thresholding and zero_pos option for zero handling
pub fn sign_value(x: f32, threshold: f32, zero_pos: bool) -> i8 {
    let x = if -threshold <= x && x <= threshold { 0.0 } else { x };

    if zero_pos {
        if x < 0.0 { -1 } else { 1 }
    } else {
        if x < 0.0 {
            -1
        } else if x > 0.0 {
            1
        } else {
            0
        }
    }
}

// zc_stencil - helper function to compute zero-crossing points for a 1D slice
// using the sign_value function to determine crossings based on the specified threshold and zero handling
fn zc_stencil(x: &[f32], threshold: f32, zero_pos: bool, y: &mut [bool]) {
    if x.is_empty() {
        return;
    }
    y[0] = false;

    for i in 1..x.len() {
        let x0 = x[i];
        let x1 = x[i - 1];

        let s0 = sign_value(x0, threshold, zero_pos);
        let s1 = sign_value(x1, threshold, zero_pos);

        y[i] = s0 != s1;
    }
}

// zc_wrapper - helper function to handle multi-d input for zero-crossing to each lane
fn zc_wrapper(x: &ArrayD<f32>,threshold: f32,zero_pos: bool,y: &mut ArrayD<bool>,) {
    let last_axis = x.ndim() - 1;

    for (input_lane, mut output_lane) in x
        .lanes(Axis(last_axis))
        .into_iter()
        .zip(y.lanes_mut(Axis(last_axis)).into_iter())
    {
        let input_vec: Vec<f32> = input_lane.iter().copied().collect();
        let output_slice = output_lane
            .as_slice_mut()
            .expect("Output lane should be contiguous");

        zc_stencil(&input_vec, threshold, zero_pos, output_slice);
    }
}

// ------------------------------------------------------------------------
// MU_COMPRESS -> applies mu-law compression to the input signal with options for quantization
// parameters: x: input array, mu: compression parameter, quantize: whether to quantize the output to integer values
// returns: mu-law compressed version of the input signal, either as continuous values or quantized integers based on the quantize parameter
// -------------------------------------------------------------------------
pub fn mu_compress(x: &ArrayD<f32>, mu: f32, quantize: bool) -> ArrayD<f32> {
    if mu <= 0.0 {
        panic!("mu-law compression parameter mu={} must be strictly positive.", mu);
    }
    if (x.iter().any(|&v| v < -1.0) || x.iter().any(|&v| v > 1.0)) {
        panic!("mu-law input x must be in the range [-1, +1].");
    }

    let x_comp = x.mapv(|v| v.signum() * (1.0 + mu * v.abs()).ln() / (1.0 + mu).ln());

    if quantize {
        let bins = (1.0 + mu) as usize;
        let edges: Vec<f32> = (0..bins)
            .map(|i| -1.0 + 2.0 * i as f32 / (bins as f32 - 1.0))
            .collect();
        let center = ((mu + 1.0) as i32) / 2;
        let y = x_comp.mapv(|v| {
            let idx = edges.partition_point(|&edge| edge < v);
            (idx as i32 - center) as f32
        });
        return y;
    }
    return x_comp;

}

// ------------------------------------------------------------------------
// MU_EXPAND -> applies inverse mu-law expansion to the input signal with options for handling quantized inputs
// parameters: x: input array, mu: expansion parameter, quantize: whether the input is quantized and needs to be scaled back
// returns: expanded version of the input signal using the inverse mu-law formula, with handling for quantized inputs if quantize is true
// --------------------------------------------------------------------------
pub fn mu_expand(x: &ArrayD<f32>, mu: f32, quantize: bool) -> ArrayD<f32> {
    if mu <= 0.0 {
        panic!("Inverse mu-law compression parameter mu={} must be strictly positive.", mu);
    }

    let x = if quantize {
        x.mapv(|v| v * 2.0 / (1.0 + mu))
    } else {
        x.clone()
    };

    if (x.iter().any(|&v| v < -1.0) || x.iter().any(|&v| v > 1.0)) {
        panic!("Inverse mu-law input x must be in the range [-1, +1].");
    }

    x.mapv(|v| v.signum() / mu * ((1.0 + mu).powf(v.abs()) - 1.0))

}

// ------------------------------------------------------------------------
// TESTS - BASED ON LIBROSA EXAMPLES
// ------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, ArrayD, IxDyn};
    const TOL: f32 = 1e-4;

    fn assert_close(a: f32, b: f32, tol: f32) {
        assert!(
            (a - b).abs() <= tol,
            "expected {b}, got {a}, diff={}",
            (a - b).abs()
        );
    }

    fn assert_slice_close(actual: &[f32], expected: &[f32], tol: f32) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "length mismatch: actual={}, expected={}",
            actual.len(),
            expected.len()
        );

        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= tol,
                "index {}: expected {}, got {}, diff={}",
                i,
                e,
                a,
                (a - e).abs()
            );
        }
    }

    fn linspace(start: f32, end: f32, num: usize) -> Vec<f32> {
        if num == 1 {
            return vec![start];
        }
        let step = (end - start) / (num as f32 - 1.0);
        (0..num).map(|i| start + i as f32 * step).collect()
    }

    fn to_dyn(v: Vec<f32>) -> ArrayD<f32> {
        ArrayD::from_shape_vec(IxDyn(&[v.len()]), v).unwrap()
    }

    fn flatten_bool(a: &ArrayD<bool>) -> Vec<bool> {
        a.iter().copied().collect()
    }

    fn flatten_f32(a: &ArrayD<f32>) -> Vec<f32> {
        a.iter().copied().collect()
    }

    #[test]
    fn test_lpc_random_signal_stability() {
        let y = (0..100).map(|i| (i as f32).sin()).collect::<Vec<_>>();
        let y = to_dyn(y);
        let coeffs = lpc(&y, 10, 0).unwrap();
        for &v in coeffs.iter() {
            assert!(v.is_finite());
        }
    }
    // ZERO CROSSING TESTS

    #[test]
    fn test_zero_crossing_librosa_example_exact_pattern() {
        // librosa example - modified for f32 using explit value
        let y = to_dyn(vec![
            0.0000000,  0.9694000,  0.4759000, -0.7357000,
           -0.8372000,  0.3247000,  0.9966000,  0.1646000,
           -0.9158000, -0.6142000,  0.6142000,  0.9158000,
           -0.1646000, -0.9966000, -0.3247000,  0.8372000,
            0.7357000, -0.4759000, -0.9694000, -9.797e-16_f32,
       ]);
        let z = zero_crossing(&y, 1e-10, None, true, true, 0);
        let actual = flatten_bool(&z);
        let expected = vec![
            true, false, false, true, false, true, false, false, true, false,
            true, false, true, false, false, true, false, true, false, true,
        ];
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_zero_crossing_librosa_example_indices() {
        let xs = linspace(0.0, 8.0 * std::f32::consts::PI, 20);
        let y: Vec<f32> = xs.iter().map(|v| v.sin()).collect();
        let y = to_dyn(y);
        let z = zero_crossing(&y, 1e-10, None, true, true, 0);
        let actual = flatten_bool(&z);
        let indices: Vec<usize> = actual
            .iter()
            .enumerate()
            .filter_map(|(i, &b)| if b { Some(i) } else { None })
            .collect();
        let expected = vec![0, 3, 5, 8, 10, 12, 15, 17];
        assert_eq!(indices, expected);
    }

    #[test]
    fn test_zero_crossing_zero_pos_false_distinguishes_zero() {
        let y = to_dyn(vec![-1.0, 0.0, 1.0]);
        let z_true = zero_crossing(&y, 0.0, None, true, true, 0);
        let z_false = zero_crossing(&y, 0.0, None, true, false, 0);
        let actual_true = flatten_bool(&z_true);
        let actual_false = flatten_bool(&z_false);
        assert_eq!(actual_true, vec![true, true, false]);
        assert_eq!(actual_false, vec![true, true, true]);
    }

    #[test]
    fn test_zero_crossing_threshold_suppresses_small_values() {
        let y = to_dyn(vec![-1e-6, 1e-6, 0.5, -0.5]);

        let z_no_thresh = zero_crossing(&y, 0.0, None, true, true, 0);
        let z_thresh = zero_crossing(&y, 1e-5, None, true, true, 0);
        let actual_no_thresh = flatten_bool(&z_no_thresh);
        let actual_thresh = flatten_bool(&z_thresh);
        assert_eq!(actual_no_thresh, vec![true, true, false, true]);
        assert_eq!(actual_thresh, vec![true, false, false, true]);
    }

    #[test]
    fn test_zero_crossing_2d_shape_preserved() {
        let y = array![
            [ 1.0_f32, -1.0,  1.0, -1.0],
            [-1.0_f32, -1.0,  1.0,  1.0]
        ].into_dyn();
        let z = zero_crossing(&y, 0.0, None, true, true, 1);
        assert_eq!(z.shape(), &[2, 4]);
    }

    // MU-COMPRESS TESTS

    #[test]
    fn test_mu_compress_no_quantize_librosa_example() {
        let x = to_dyn(linspace(-1.0, 1.0, 16));
        let y = mu_compress(&x, 255.0, false);
        let actual = flatten_f32(&y);
        let expected = vec![
            -1.00000000, -0.97430198, -0.94432361, -0.90834832,
            -0.86336132, -0.80328309, -0.71255496, -0.52124063,
             0.52124063,  0.71255496,  0.80328309,  0.86336132,
             0.90834832,  0.94432361,  0.97430198,  1.00000000,
        ];
        assert_slice_close(&actual, &expected, 1e-4);
    }

    #[test]
    fn test_mu_compress_quantized_librosa_example() {
        let x = to_dyn(linspace(-1.0, 1.0, 16));
        let y = mu_compress(&x, 255.0, true);
        let actual = flatten_f32(&y);
        let expected = vec![
            -128.0, -124.0, -120.0, -116.0, -110.0, -102.0, -91.0, -66.0,
             66.0,   91.0,  102.0,  110.0,  116.0,  120.0,  124.0,  127.0,
        ];
        assert_slice_close(&actual, &expected, 1e-4);
    }

    #[test]
    fn test_mu_compress_quantized_mu15_librosa_example() {
        let x = to_dyn(linspace(-1.0, 1.0, 16));
        let y = mu_compress(&x, 15.0, true);
        let actual = flatten_f32(&y);
        let expected = vec![
            -8.0, -7.0, -7.0, -6.0, -6.0, -5.0, -4.0, -2.0,
             2.0,  4.0,  5.0,  6.0,  6.0,  7.0,  7.0,  7.0,
        ];
        assert_slice_close(&actual, &expected, 1e-4);
    }

    #[test]
    fn test_mu_compress_preserves_endpoints_without_quantization() {
        let x = to_dyn(vec![-1.0, 0.0, 1.0]);
        let y = mu_compress(&x, 255.0, false);
        let actual = flatten_f32(&y);
        assert_close(actual[0], -1.0, TOL);
        assert_close(actual[1],  0.0, TOL);
        assert_close(actual[2],  1.0, TOL);
    }

    #[test]
    #[should_panic]
    fn test_mu_compress_panics_on_invalid_mu() {
        let x = to_dyn(vec![0.0, 0.5, -0.5]);
        let _ = mu_compress(&x, 0.0, false);
    }

    #[test]
    #[should_panic]
    fn test_mu_compress_panics_on_out_of_range_input() {
        let x = to_dyn(vec![-1.2, 0.0, 0.5]);
        let _ = mu_compress(&x, 255.0, false);
    }
    // MU-EXPAND TESTS

    #[test]
    fn test_mu_expand_roundtrip_no_quantize_librosa_example() {
        let x = to_dyn(linspace(-1.0, 1.0, 16));
        let y = mu_compress(&x, 255.0, false);
        let z = mu_expand(&y, 255.0, false);
        let actual = flatten_f32(&z);
        let expected = linspace(-1.0, 1.0, 16);
        assert_slice_close(&actual, &expected, 1e-4);
    }

    #[test]
    fn test_mu_expand_quantized_librosa_example() {
        let x = to_dyn(linspace(-1.0, 1.0, 16));
        let y = mu_compress(&x, 255.0, true);
        let z = mu_expand(&y, 255.0, true);
        let actual = flatten_f32(&z);
        let expected = vec![
            -1.00000000, -0.84027248, -0.70595818, -0.59301377,
            -0.45637850, -0.32155973, -0.19817918, -0.06450245,
             0.06450245,  0.19817918,  0.32155973,  0.45637850,
             0.59301377,  0.70595818,  0.84027248,  0.95743702,
        ];
        assert_slice_close(&actual, &expected, 1e-4);
    }

    #[test]
    fn test_mu_expand_preserves_zero_without_quantization() {
        let x = to_dyn(vec![0.0]);
        let z = mu_expand(&x, 255.0, false);
        let actual = flatten_f32(&z);
        assert_close(actual[0], 0.0, TOL);
    }

    #[test]
    #[should_panic]
    fn test_mu_expand_panics_on_invalid_mu() {
        let x = to_dyn(vec![0.0, 0.5, -0.5]);
        let _ = mu_expand(&x, -1.0, false);
    }

    #[test]
    #[should_panic]
    fn test_mu_expand_panics_on_out_of_range_input() {
        let x = to_dyn(vec![2.0, 0.0, -0.5]);
        let _ = mu_expand(&x, 255.0, false);
    }

    // LPC TESTS
    #[test]
    fn test_lpc_1d_returns_order_plus_one_coeffs() {
        let y = to_dyn(vec![0.1, 0.3, -0.2, 0.5, -0.1, 0.2, 0.4, -0.3]);
        let coeffs = lpc(&y, 2, 0).unwrap();
        assert_eq!(coeffs.shape(), &[3]);
        let actual = flatten_f32(&coeffs);
        assert_eq!(actual.len(), 3);
        assert_close(actual[0], 1.0, TOL);
        for &v in &actual {
            assert!(v.is_finite(), "LPC coefficient should be finite, got {}", v);
        }
    }

    #[test]
    fn test_lpc_invalid_order_zero_returns_err() {
        let y = to_dyn(vec![0.1, 0.2, 0.3, 0.4]);
        let result = lpc(&y, 0, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_lpc_order_too_large_returns_err() {
        let y = to_dyn(vec![0.1, 0.2, 0.3]);
        let result = lpc(&y, 3, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_lpc_nonfinite_input_returns_err() {
        let y = to_dyn(vec![0.1, f32::NAN, 0.3, 0.4]);
        let result = lpc(&y, 2, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_lpc_axis_0_shape_2d() {
        let y = array![
            [0.1_f32, 0.2],
            [0.2_f32, 0.3],
            [0.3_f32, 0.4],
            [0.4_f32, 0.5],
            [0.5_f32, 0.6],].into_dyn();
        let coeffs = lpc(&y, 2, 0).unwrap();
        assert_eq!(coeffs.shape(), &[3, 2]);
        for &v in coeffs.iter() {
            assert!(v.is_finite(), "LPC coefficient should be finite, got {}", v);
        }
    }

    #[test]
    fn test_lpc_axis_1_shape_2d() {
        let y = array![
            [0.1_f32, 0.2, 0.3, 0.4, 0.5],
            [0.5_f32, 0.4, 0.3, 0.2, 0.1],].into_dyn();
        let coeffs = lpc(&y, 2, 1).unwrap();
        assert_eq!(coeffs.shape(), &[2, 3]);
        for &v in coeffs.iter() {
            assert!(v.is_finite(), "LPC coefficient should be finite, got {}", v);
        }
    }

    #[test]
    fn test_lpc_first_coeff_is_one_for_each_lane() {
        let y = array![
            [0.1_f32, 0.2, 0.3, 0.4, 0.5],
            [0.5_f32, 0.4, 0.3, 0.2, 0.1],
        ]
        .into_dyn();
        let coeffs = lpc(&y, 2, 1).unwrap();
        for row in coeffs.outer_iter() {
            assert_close(row[0], 1.0, TOL);
        }
    }
    // AUTOCORRELATE TESTS

    #[test]
    fn test_autocorrelate_1d_known_values() {
        let y = to_dyn(vec![1.0_f32, 2.0, 3.0]);
        let ac = autocorrelate(&y, None, 0).unwrap();
        let actual = flatten_f32(&ac);
        let expected = vec![14.0_f32, 8.0, 3.0];
        assert_slice_close(&actual, &expected, TOL);
    }

    #[test]
    fn test_autocorrelate_1d_known_values_bounded() {
        let y = to_dyn(vec![1.0_f32, 2.0, 3.0]);
        let ac = autocorrelate(&y, Some(2), 0).unwrap();
        let actual = flatten_f32(&ac);
        let expected = vec![14.0_f32, 8.0];
        assert_slice_close(&actual, &expected, TOL);
    }
    #[test]
    fn test_autocorrelate_default_max_size_equals_axis_length() {
        let y = to_dyn(vec![0.5_f32, -0.25, 0.75, -1.0, 0.125]);
        let ac = autocorrelate(&y, None, 0).unwrap();
        assert_eq!(ac.shape(), &[5]);
    }

    #[test]
    fn test_autocorrelate_max_size_is_clamped_to_axis_length() {
        let y = to_dyn(vec![0.5_f32, -0.25, 0.75]);

        let ac = autocorrelate(&y, Some(100), 0).unwrap();

        assert_eq!(ac.shape(), &[3]);
    }

    #[test]
    fn test_autocorrelate_lag_zero_is_signal_energy() {
        let y_vec = vec![0.2_f32, -0.4, 0.6, -0.8];
        let y = to_dyn(y_vec.clone());

        let ac = autocorrelate(&y, None, 0).unwrap();
        let actual = flatten_f32(&ac);

        let expected_energy: f32 = y_vec.iter().map(|v| v * v).sum();
        assert_close(actual[0], expected_energy, TOL);
    }

    #[test]
    fn test_autocorrelate_singleton_signal() {
        let y = to_dyn(vec![2.5_f32]);

        let ac = autocorrelate(&y, None, 0).unwrap();
        let actual = flatten_f32(&ac);

        assert_eq!(actual.len(), 1);
        assert_close(actual[0], 6.25, TOL);
    }

    #[test]
    fn test_autocorrelate_zero_signal() {
        let y = to_dyn(vec![0.0_f32, 0.0, 0.0, 0.0]);

        let ac = autocorrelate(&y, None, 0).unwrap();
        let actual = flatten_f32(&ac);

        let expected = vec![0.0_f32, 0.0, 0.0, 0.0];
        assert_slice_close(&actual, &expected, TOL);
    }

    #[test]
    fn test_autocorrelate_axis_1_shape_2d() {
        let y = array![
            [1.0_f32, 2.0, 3.0],
            [4.0_f32, 5.0, 6.0]
        ]
        .into_dyn();
        let ac = autocorrelate(&y, Some(2), 1).unwrap();
        assert_eq!(ac.shape(), &[2, 2]);
    }

    #[test]
    fn test_autocorrelate_axis_1_values_2d() {
        let y = array![
            [1.0_f32, 2.0, 3.0],
            [4.0_f32, 5.0, 6.0]
        ]
        .into_dyn();

        let ac = autocorrelate(&y, Some(3), 1).unwrap();
        let expected = array![
            [14.0_f32, 8.0, 3.0],
            [77.0_f32, 50.0, 24.0]
        ]
        .into_dyn();

        assert_eq!(ac.shape(), expected.shape());
        for (a, e) in ac.iter().zip(expected.iter()) {
            assert_close(*a, *e, TOL);
        }
    }

    #[test]
    fn test_autocorrelate_axis_0_shape_2d() {
        let y = array![
            [1.0_f32, 10.0],
            [2.0_f32, 20.0],
            [3.0_f32, 30.0]
        ]
        .into_dyn();

        let ac = autocorrelate(&y, Some(2), 0).unwrap();

        assert_eq!(ac.shape(), &[2, 2]);
    }

    #[test]
    fn test_autocorrelate_axis_0_values_2d() {
        let y = array![
            [1.0_f32, 10.0],
            [2.0_f32, 20.0],
            [3.0_f32, 30.0]
        ]
        .into_dyn();

        let ac = autocorrelate(&y, Some(3), 0).unwrap();
        let expected = array![
            [14.0_f32, 1400.0],
            [8.0_f32, 800.0],
            [3.0_f32, 300.0]
        ]
        .into_dyn();

        assert_eq!(ac.shape(), expected.shape());

        for (a, e) in ac.iter().zip(expected.iter()) {
            assert_close(*a, *e, TOL);
        }
    }

    #[test]
    fn test_autocorrelate_nonnegative_for_positive_signal() {
        let y = to_dyn(vec![1.0_f32, 2.0, 3.0, 4.0]);
        let ac = autocorrelate(&y, None, 0).unwrap();
        let actual = flatten_f32(&ac);
        for v in actual {
            assert!(v >= 0.0, "expected nonnegative autocorrelation, got {}", v);
        }
    }

    #[test]
    fn test_autocorrelate_first_value_largest_for_positive_signal() {
        let y = to_dyn(vec![1.0_f32, 2.0, 3.0, 4.0]);
        let ac = autocorrelate(&y, None, 0).unwrap();
        let actual = flatten_f32(&ac);
        let first = actual[0];
        for &v in &actual[1..] {
            assert!(
                first >= v,
                "expected lag-0 autocorrelation to be largest, but {} < {}",
                first,
                v
            );
        }
    }
    #[test]
    fn test_autocorrelate_invalid_axis_returns_err() {
        let y = to_dyn(vec![1.0_f32, 2.0, 3.0]);

        let result = autocorrelate(&y, None, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_autocorrelate_nonfinite_input_returns_err_nan() {
        let y = to_dyn(vec![1.0_f32, f32::NAN, 3.0]);

        let result = autocorrelate(&y, None, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_autocorrelate_nonfinite_input_returns_err_inf() {
        let y = to_dyn(vec![1.0_f32, f32::INFINITY, 3.0]);

        let result = autocorrelate(&y, None, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_autocorrelate_on_onset_like_vector_with_frame_limit() {
        let odf = to_dyn(vec![
            0.0_f32, 0.2, 0.8, 0.1, 0.0, 0.5, 1.0, 0.3, 0.0, 0.4
        ]);

        let ac = autocorrelate(&odf, Some(4), 0).unwrap();
        let actual = flatten_f32(&ac);
        assert_eq!(actual.len(), 4);
        let expected_energy: f32 = vec![0.0_f32, 0.2, 0.8, 0.1, 0.0, 0.5, 1.0, 0.3, 0.0, 0.4]
            .iter()
            .map(|v| v * v)
            .sum();

        assert_close(actual[0], expected_energy, TOL);
    }

    #[test]
    fn test_autocorr_added_1d_known_values() {
        let y = to_dyn(vec![1.0_f32, 2.0, 3.0]);
    
        let ac = autocorrelate(&y, None, 0).unwrap();
        let actual = flatten_f32(&ac);
    
        let expected = vec![14.0_f32, 8.0, 3.0];
    
        assert_slice_close(&actual, &expected, 1e-4);
    }
    
    #[test]
    fn test_autocorr_added_bounded_length() {
        let y = to_dyn(vec![1.0_f32, 2.0, 3.0]);
        let ac = autocorrelate(&y, Some(2), 0).unwrap();
        let actual = flatten_f32(&ac);
        let expected = vec![14.0_f32, 8.0];
        assert_slice_close(&actual, &expected, 1e-4);
    }
    
    #[test]
    fn test_autocorr_added_lag_zero_energy() {
        let y_vec = vec![0.2_f32, -0.4, 0.6, -0.8];
        let y = to_dyn(y_vec.clone());
        let ac = autocorrelate(&y, None, 0).unwrap();
        let actual = flatten_f32(&ac);
        let expected: f32 = y_vec.iter().map(|v| v * v).sum();
        assert_close(actual[0], expected, 1e-4);
    }
    
    #[test]
    fn test_autocorr_added_axis_1_values() {
        let y = ndarray::array![
            [1.0_f32, 2.0, 3.0],
            [4.0_f32, 5.0, 6.0]
        ]
        .into_dyn();
        let ac = autocorrelate(&y, Some(3), 1).unwrap();
        let expected = ndarray::array![
            [14.0_f32, 8.0, 3.0],
            [77.0_f32, 50.0, 24.0]
        ]
        .into_dyn();
        for (a, e) in ac.iter().zip(expected.iter()) {
            assert_close(*a, *e, 1e-4);
        }
    }
    
    #[test]
    fn test_autocorr_added_axis_0_values() {
        let y = ndarray::array![
            [1.0_f32, 10.0],
            [2.0_f32, 20.0],
            [3.0_f32, 30.0]
        ]
        .into_dyn();
        let ac = autocorrelate(&y, Some(3), 0).unwrap();
        let expected = ndarray::array![
            [14.0_f32, 1400.0],
            [8.0_f32, 800.0],
            [3.0_f32, 300.0]
        ]
        .into_dyn();
        for (a, e) in ac.iter().zip(expected.iter()) {
            assert_close(*a, *e, 1e-4);
        }
    }
    
    #[test]
    fn test_autocorr_added_max_size_clamped() {
        let y = to_dyn(vec![0.5_f32, -0.25, 0.75]);
    
        let ac = autocorrelate(&y, Some(100), 0).unwrap();
    
        assert_eq!(ac.shape(), &[3]);
    }
    
    #[test]
    fn test_autocorr_added_zero_signal() {
        let y = to_dyn(vec![0.0_f32, 0.0, 0.0]);
        let ac = autocorrelate(&y, None, 0).unwrap();
        let actual = flatten_f32(&ac);
        let expected = vec![0.0_f32, 0.0, 0.0];
        assert_slice_close(&actual, &expected, 1e-4);
    }
    
    #[test]
    fn test_autocorr_added_invalid_axis() {
        let y = to_dyn(vec![1.0_f32, 2.0, 3.0]);
        let result = autocorrelate(&y, None, 2);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_autocorr_added_nan_input() {
        let y = to_dyn(vec![1.0_f32, f32::NAN, 3.0]);
        let result = autocorrelate(&y, None, 0);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_autocorr_added_inf_input() {
        let y = to_dyn(vec![1.0_f32, f32::INFINITY, 3.0]);
        let result = autocorrelate(&y, None, 0);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_autocorr_added_onset_style_vector() {
        let y = to_dyn(vec![
            0.0_f32, 0.2, 0.8, 0.1, 0.0, 0.5, 1.0, 0.3, 0.0, 0.4
        ]);
        let ac = autocorrelate(&y, Some(4), 0).unwrap();
        let actual = flatten_f32(&ac);
        assert_eq!(actual.len(), 4);
        let expected_energy: f32 = vec![
            0.0_f32, 0.2, 0.8, 0.1, 0.0, 0.5, 1.0, 0.3, 0.0, 0.4
        ]
        .iter()
        .map(|v| v * v)
        .sum();
        assert_close(actual[0], expected_energy, 1e-4);
    }

}

