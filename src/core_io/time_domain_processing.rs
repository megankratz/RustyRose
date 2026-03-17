
// dependencies to support scipy and rfft functionality
use rustfft::FftPlanner;
use num_complex::Complex;
use ndarray::ArrayD;

// ------------------------------------------------------------------------------------
// Work in progress - lpc and testing needs to be completed still for these functions
// ------------------------------------------------------------------------------------

//autocorrelate
/*
def autocorrelate(
    y: np.ndarray, *, max_size: Optional[int] = None, axis: int = -1
) -> np.ndarray:
    if max_size is None:
        max_size = y.shape[axis]

    max_size = int(min(max_size, y.shape[axis]))

    fft = get_fftlib()

    real = not np.iscomplexobj(y)

    # Pad out the signal to support full-length auto-correlation
    n_pad = scipy.fft.next_fast_len(2 * y.shape[axis] - 1, real=real)

    if real:
        # Compute the power spectrum along the chosen axis
        powspec = util.abs2(fft.rfft(y, n=n_pad, axis=axis))

        # Convert back to time domain
        autocorr = fft.irfft(powspec, n=n_pad, axis=axis)
    else:
        # Compute the power spectrum along the chosen axis
        powspec = util.abs2(fft.fft(y, n=n_pad, axis=axis))

        # Convert back to time domain
        autocorr = fft.ifft(powspec, n=n_pad, axis=axis)

    # Slice down to max_size
    subslice = [slice(None)] * autocorr.ndim
    subslice[axis] = slice(max_size)

    autocorr_slice: np.ndarray = autocorr[tuple(subslice)]

    return autocorr_slice
*/
pub enum Signal<'a> {
    Real(&'a [f32]),
    Complex(&'a [Complex<f32>]),
}

pub enum AutoCorrResult {
    Real(Vec<f32>),
    Complex(Vec<Complex<f32>>),
}

pub fn autocorrelate(signal: Signal, max_size: Option<usize>) -> AutoCorrResult {
    let signal_len = match &signal {
        Signal::Real(y) => y.len(),
        Signal::Complex(y) => y.len(),};

    let mut max_size = max_size.unwrap_or(signal_len);

    max_size = max_size.min(signal_len);

    let mut planner = FftPlanner::<f32>::new();

    let real = matches!(signal, Signal::Real(_));

    let n_pad = next_fast_len(2 * signal_len - 1);

    if real {
        let fft = planner.plan_fft_forward(n_pad);
        let ifft = planner.plan_fft_inverse(n_pad);

        let mut powspec = vec![Complex::new(0.0, 0.0); n_pad];

        if let Signal::Real(y) = signal {
            for (i, &value) in y.iter().enumerate() {
                powspec[i] = Complex::new(value, 0.0);
            }
        }

        fft.process(&mut powspec);

        for value in powspec.iter_mut() {
            *value = Complex::new(value.norm_sqr(), 0.0);
        }

        let mut autocorr = powspec;
        ifft.process(&mut autocorr);

        let scale = n_pad as f32;
        let autocorr_slice: Vec<f32> = autocorr[..max_size]
            .iter()
            .map(|z| z.re / scale)
            .collect();

        AutoCorrResult::Real(autocorr_slice)
    } else {
        let fft = planner.plan_fft_forward(n_pad);
        let ifft = planner.plan_fft_inverse(n_pad);
        let mut powspec = vec![Complex::new(0.0, 0.0); n_pad];

        if let Signal::Complex(y) = signal {
            for (i, &value) in y.iter().enumerate() {
                powspec[i] = value;
            }
        }

        fft.process(&mut powspec);

        for value in powspec.iter_mut() {
            *value = Complex::new(value.norm_sqr(), 0.0);
        }

        let mut autocorr = powspec;
        ifft.process(&mut autocorr);

        let scale = n_pad as f32;

        let autocorr_slice: Vec<Complex<f32>> = autocorr[..max_size]
            .iter()
            .map(|z| *z / scale)
            .collect();

        AutoCorrResult::Complex(autocorr_slice)
    }
}

// librosa uses next_fast_len(2*N - 1) -> helper for autocorrelate
pub fn next_fast_len(n: usize) -> usize {
    n.next_power_of_two()
}

// lpc
/*
[docs]def lpc(y: np.ndarray, *, order: int, axis: int = -1) -> np.ndarray:
    if not util.is_positive_int(order):
        raise ParameterError(f"order={order} must be an integer > 0")

    util.valid_audio(y)

    # Move the lpc axis around front, because numba is silly
    y = y.swapaxes(axis, 0)

    dtype = y.dtype

    shape = list(y.shape)
    shape[0] = order + 1

    ar_coeffs = np.zeros(tuple(shape), dtype=dtype)
    ar_coeffs[0] = 1

    ar_coeffs_prev = ar_coeffs.copy()

    shape[0] = 1
    reflect_coeff = np.zeros(shape, dtype=dtype)
    den = reflect_coeff.copy()

    epsilon = util.tiny(den)

    # Call the helper, and swap the results back to the target axis position
    return np.swapaxes(
        __lpc(y, order, ar_coeffs, ar_coeffs_prev, reflect_coeff, den, epsilon), 0, axis
    )
*/
pub fn lpc(y: &ArrayD<f32>, order: usize, axis: usize) -> ArrayD<f32> {
    todo!();
}
//zero_crossing
/*
@cache(level=20)
def zero_crossings(
    y: np.ndarray,
    *,
    threshold: float = 1e-10,
    ref_magnitude: Optional[Union[float, Callable]] = None,
    pad: bool = True,
    zero_pos: bool = True,
    axis: int = -1,
) -> np.ndarray:
    if callable(ref_magnitude):
        threshold = threshold * ref_magnitude(np.abs(y))

    elif ref_magnitude is not None:
        threshold = threshold * ref_magnitude

    yi = y.swapaxes(-1, axis)
    z = np.empty_like(y, dtype=bool)
    zi = z.swapaxes(-1, axis)

    _zc_wrapper(yi, threshold, zero_pos, zi)

    zi[..., 0] = pad

    return z
*/

// mu_compress
/*
def mu_compress(
    x: Union[np.ndarray, _FloatLike_co], *, mu: float = 255, quantize: bool = True
) -> np.ndarray:
    if mu <= 0:
        raise ParameterError(
            f"mu-law compression parameter mu={mu} must be strictly positive."
        )

    if np.any(x < -1) or np.any(x > 1):
        raise ParameterError(f"mu-law input x={x} must be in the range [-1, +1].")

    x_comp: np.ndarray = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)

    if quantize:
        y: np.ndarray = (
            np.digitize(
                x_comp, np.linspace(-1, 1, num=int(1 + mu), endpoint=True), right=True
            )
            - int(mu + 1) // 2
        )
        return y

    return x_comp
*/
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
        let edges: Vec<f32> = (0..=bins).map(|i| -1.0 + 2.0 * i as f32 / bins as f32).collect();
        let y = x_comp.mapv(|v| {
            let idx = edges.iter().position(|&edge| v <= edge).unwrap_or(bins);
            idx as f32 - (bins as f32 / 2.0)
        });
        return y;
    }
    return x_comp;

}


// mu_expand
/*
[docs]def mu_expand(
    x: Union[np.ndarray, _FloatLike_co], *, mu: float = 255.0, quantize: bool = True
) -> np.ndarray:
    if mu <= 0:
        raise ParameterError(
            f"Inverse mu-law compression parameter mu={mu} must be strictly positive."
        )

    if quantize:
        x = x * 2.0 / (1 + mu)

    if np.any(x < -1) or np.any(x > 1):
        raise ParameterError(
            f"Inverse mu-law input x={x} must be in the range [-1, +1]."
        )

    return np.sign(x) / mu * (np.power(1 + mu, np.abs(x)) - 1)
 */
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


// TESTS

