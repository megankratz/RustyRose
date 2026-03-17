// temp file i have added while trying to write my other functions, feel free to 
// add or change anything I have written here 

/*
def valid_audio(y: np.ndarray) -> bool:
    """Determine whether a variable contains valid audio data.

    The following conditions must be satisfied:

    - ``type(y)`` is ``np.ndarray``
    - ``y.dtype`` is floating-point
    - ``y.ndim != 0`` (must have at least one dimension)
    - ``np.isfinite(y).all()`` samples must be all finite values
    ---
    if not isinstance(y, np.ndarray):
        raise ParameterError("Audio data must be of type numpy.ndarray")

    if not np.issubdtype(y.dtype, np.floating):
        raise ParameterError("Audio data must be floating-point")

    if y.ndim == 0:
        raise ParameterError(
            f"Audio data must be at least one-dimensional, given y.shape={y.shape}"
        )

    if not np.isfinite(y).all():
        raise ParameterError("Audio buffer is not finite everywhere")

    return True
*/

pub fn valid_audio(y: &[f32]) -> bool {
    if y.is_empty() {
        panic!("Audio data must be at least one-dimensional, given length=0");
    }

    if !y.iter().all(|&v| v.is_finite()) {
        panic!("Audio buffer is not finite everywhere");
    }
    true
}