use core::f64;

use ndarray::{Array1, Array2, ArrayD, Axis, s, array, concatenate, arr1};
use crate::{core_io::*, feature::rms};

/// Trim leading and trailing silence from input samples
/// 
/// ### Arguments
/// y : Input samples
/// 
/// top_db : Threshold in decibels, below which is considered silence
/// 
/// ref_ : Reference amplitude. If None, uses max function to get reference
/// 
/// frame_length : size of each frame. Defaults to 2048
/// 
/// hop_length: distance between frames. Defaults to 512
/// 
/// ### Returns
/// yt : Input with leading and trailing silence removed
/// 
/// idx : start and end indices of trimmed samples
pub fn trim(
    y : &Array1<f64>,
    top_db: Option<f64>,
    ref_ : f64,
    frame_length : Option<i32>,
    hop_length : Option<i32>
) -> (Array1<f64>, Array1<i32>)
{
    let non_silent : Array1<bool> = signal_to_frame_nonsilent(y, frame_length, hop_length, top_db.unwrap_or(60.), Some(ref_));

    let nonzero : Array1<usize> = Array1::from(non_silent.iter()
        .enumerate()
        .filter_map(|(i, &v)| if v {Some(i) } else {None})
        .collect::<Vec<_>>());

    let start: i32 = if nonzero.len() > 0 {
        frames_to_samples(nonzero[0] as i32, hop_length , None)
    } else {
        0
    };

    let end : i32 = match nonzero.len() {
        x if x == non_silent.len()  => x as i32 - 1,
        x if x <= 0                 => 0,
        _ => (*nonzero.last().unwrap() as i32 + 1).min(y.len() as i32 - 1)
    };

    (y.slice(s![start..end]).to_owned(), Array1::from(array![start, end]))
}

/// Split input into intervals separated by silent frames
/// 
/// ### Arguments
/// y : Input samples
/// 
/// top_db : Threshold in decibels, below which is considered silence
/// 
/// ref_ : Reference amplitude. If None, uses max function to get reference
/// 
/// frame_length : size of each frame. Defaults to 2048
/// 
/// hop_length: distance between frames. Defaults to 512
/// 
/// ### Returns
/// indices : Array of size (m, 2). Each row contains the start and 
/// stop index of a new interval
pub fn split(
    y : &Array1<f64>,
    top_db: Option<f64>,
    ref_ : f64,
    frame_length : Option<i32>,
    hop_length : Option<i32>
) -> Array2<usize> {

    let non_silent : Array1<bool> = signal_to_frame_nonsilent(y, frame_length, hop_length, top_db.unwrap_or(60.), Some(ref_));

    // Get all indices where frames change between silent and non-silent
    let mut edges : Array1<usize> = non_silent
        .iter()
        .zip(non_silent.iter().skip(1))
        .enumerate()
        .filter_map(|(i, (&a, &b))| {
            if a != b {
                Some(i + 1) // add 1 since we're not counting first frame
            } else {
                None
            }
        }).collect();

    // If the first frame had high energy, count it
    if non_silent[0]{
        edges = concatenate![Axis(0), arr1(&[0]), edges.view()];
    }

    // Same for the last frame
    if *non_silent.last().unwrap_or(&false) {
        edges = concatenate![Axis(0), edges.view(), arr1(&[non_silent.len()])];
    }

    // Convert from frames to samples
    let samples : Array1<i32> = edges
        .iter()
        .map(|&x |{ 
            let i = frames_to_samples(x as i32, hop_length, None);
            i.min(y.len() as i32) //  ensure each sample is in the input array
        })
        .collect();

    let len = samples.len();
    samples 
        .into_shape((len / 2, 2))
        .unwrap()
        .mapv(|x| x as usize)
}


/// Rearrange input samples using the given intervals
/// 
/// ### Arguments
/// y : Input samples
/// 
/// intervals : Start and stop points of each interval, in the order they'll be used
/// 
/// align_zeros : CURRENTLY UNUSED
/// 
/// ### Returns
/// y_remix : y remixed in the order specified by intervals
pub fn remix(
    y : &Array1<f64>,
    intervals: &Array2<usize>,
    align_zeros : Option<bool>
) -> Array1<f64>{

    let mut out : Vec<f64> = Vec::new();

    // Not currently implemented
    let zeros : Vec<usize> = if align_zeros.unwrap_or(true) {
        let y_d: ArrayD<f32> = y.mapv(|x| x as f32).into_dyn(); // convert y to right format
        let crossings : Array1<bool>  = zero_crossing(&y_d, 1e-10, None, true, true, 0)
            .into_dimensionality::<ndarray::Ix1>().unwrap();
        
        let mut indices : Vec<usize> = crossings
            .indexed_iter()
            .filter_map(|(i, &v)| if v {Some(i)} else {None})
            .collect();
        indices.push(y.len());
        indices
    } else {
        Vec::new()
    };


    for row in intervals.rows() {
        let start : usize = row[0];
        let end : usize = row[1];
        if align_zeros.unwrap_or(true) {
            // TODO: Implement match_events
        }        
        out.extend(y
            .slice(s![start..end])
            .iter());
    }

    Array1::from(out)
}

/// Pre-emphasize an audio signal with a first order differencing filter
/// 
/// ### Arguments
/// y : input signal
/// 
/// coef : pre-emphasis coefficient. Usually between 0-1
/// 
/// zi_in : Initial filter state. If not provided, defaults to 2*y[0] - y[1]
/// 
/// ### Returns
/// y_out : pre-emphasized signal
/// 
/// return_zf : Final filter state
pub fn preemphasis(
    y : &Array1<f64>,
    coef : Option<f64>,
    zi_in : Option<f64>
) -> (Array1<f64>, f64){

    let mut zi = match zi_in {
        Some(v) => v,
        None => 2.0 * y[0] - y[1]
    };

    let mut out : Array1<f64> = Array1::<f64>::zeros(y.len());
    for (i, &x) in y.iter().enumerate() {
        out[i] = x - coef.unwrap_or(0.97) * zi;
        zi = x;
    }

    let zf : f64 = zi;
    (out, zf)
}

/// De-emphasize an audio signal, inverse of preemphasis function
/// 
/// ### Arguments
/// y : input signal
/// 
/// coef : pre-emphasis coefficient. Usually between 0-1
/// 
/// zi_in : Initial filter state. If not provided, defaults to 2*y[0] - y[1]
/// 
/// ### Returns
/// y_out : de-emphasized signal
/// 
/// return_zf : Final filter state
pub fn deemphasis(
    y : &Array1<f64>,
    coef : Option<f64>,
    zi_in : Option<f64>
) -> (Array1<f64>, f64) {

    let mut zi = match zi_in {
        Some(v) => v,
        None => (2.0 - coef.unwrap_or(0.97)) * y[0] - y[1] / (3.0 - coef.unwrap_or(0.97))
    };

    let mut x : Array1<f64> = Array1::<f64>::zeros(y.len());
    for i in 0..y.len() {
        let val = y[i] + coef.unwrap_or(0.97) * zi;
        x[i] = val;
        zi = val;
    }

    let zf = zi;
    (x, zf)
}


fn signal_to_frame_nonsilent(
    y : &Array1<f64>,
    frame_length : Option<i32>,
    hop_length : Option<i32>,
    top_db : f64,
    reference : Option<f64>
) -> Array1<bool>
{
    let mse : Array1<f64> = rms(y, frame_length, hop_length, true);
    let amin = 1e-10;

    let ref_ : f64 = match reference {
        Some(v) => v,
        None => mse.iter().cloned().fold(f64::NEG_INFINITY, f64::max) // Computes max
    }
        .max(amin);


    let slice : &[f64] = mse.as_slice().unwrap();
    let db = amplitude_to_db(slice, ref_, amin, Some(top_db));
    Array1::<f64>::from(db).mapv(|x| x > -top_db)
}

#[cfg(test)]
mod tests {
    use std::array;

    use crate::effects::*;
    use ndarray::{Array1, Array2, array};

    // Trim
    #[test]
    fn test_trim_simple() {
        let y : Array1<f64> = array![0.0, 0.1, 0.5, 0.9, 0.0];
        let (yt, i) = trim(&y, Some(40.), 1., Some(2), Some(1));

        assert!(i[0] < i[1], "Ending trim has earlier index than start");
        assert!(yt[0] != 0.0, "Leading silence after trim");
        assert!(yt[yt.len() - 1] != 0.0, "Trailing silence after trim");
    }

    #[test]
    fn test_trim_no_silence() {
        let y : Array1<f64> = array![0.2, 0.3, 0.4, 0.5];
        let (yt, i) = trim(&y, Some(40.), 0.1, Some(2), Some(1));

        assert_eq!(i[0], 0, "Start trim does not end at 0");
        assert_eq!(i[1], y.len() as i32, " end trim does not start at end of input");

        assert_eq!(y[0], yt[0], "first index does not match");
        assert_eq!(y[1], yt[1], "second index does not match");
        assert_eq!(y[2], yt[2], "third index does not match");
        assert_eq!(y[3], yt[3], "fourth index does not match");
    }

    #[test]
    fn test_trim_all_silence() {
        let y: Array1<f64> = array![0.0,0.0,0.0,0.0];
        let (yt, i) = trim(&y, Some(40.), 1.0, Some(2), Some(1));

        assert_eq!(0, yt.len(), "Did not return empty array");
        assert_eq!(0, i[0], "Start trim not at 0");
        assert_eq!(0, i[1] as usize, "End trim not at 0");
    }

    // Split
    #[test]
    fn test_split_basic() {
        let y : Array1<f64> = array![0.0, 0.0, 0.5, 0.6, 0.0, 0.0, 0.1, 0.2];
        let intervals : Array2<usize> = split(&y, Some(40.), 1.0, Some(2), Some(1));

        assert_eq!(2, intervals[[0,0]]);
        assert_eq!(5, intervals[[0,1]]);
        assert_eq!(6, intervals[[1,0]]);
        assert_eq!(8, intervals[[1,1]]);
    }

    #[test]
    fn test_split_no_silence() {
        let y : Array1<f64> = array![0.1,0.2,0.3,0.4,0.5];
        let intervals : Array2<usize> = split(&y, Some(40.), 1.0, Some(2), Some(1));

        assert_eq!(0, intervals[[0,0]]);
        assert_eq!(y.len(), intervals[[0,1]]);
    }

    #[test]
    fn test_split_all_silence() {
        let y : Array1<f64> = array![0.0,0.0,0.0,0.0,0.0];
        let intervals : Array2<usize> = split(&y, Some(40.), 1.0, Some(2), Some(1));

        assert_eq!(0, intervals.len());
    }

    // Remix
    #[test]
    fn test_remix_single_interval() {
        let y : Array1<f64> = array![0.1, 0.2, 0.3, 0.4];
        let intervals : Array2<usize> = array![[1,3]];

        let output = remix(&y, &intervals, None);

        assert_eq!(0.2, output[0]);
        assert_eq!(0.3, output[1]);
    }

    #[test]
    fn test_remix_multiple_intervals() {
        let y : Array1<f64> = array![0.1, 0.2, 0.3, 0.4, 0.5];
        let intervals: Array2<usize> = array![[0, 2], [3, 5]];

        let output = remix(&y, &intervals, None);

        assert_eq!(0.1, output[0]);
        assert_eq!(0.2, output[1]);
        assert_eq!(0.4, output[2]);
        assert_eq!(0.5, output[3]);
    }

    #[test]
    fn test_remix_reorder_intervals() {
        let y : Array1<f64> = array![0.1, 0.2, 0.3, 0.4, 0.5];
        let intervals: Array2<usize> = array![[3,5], [0,2]];

        let output = remix(&y, &intervals, None);

        assert_eq!(0.4, output[0]);
        assert_eq!(0.5, output[1]);
        assert_eq!(0.1, output[2]);
        assert_eq!(0.2, output[3]);
    }

    // Preemphasis
    #[test]
    fn test_preemphasis_basic() {
        let y = array![1.0, 2.0, 3.0];
        let coef = Some(0.5);

        let (out, zf) = preemphasis(&y, coef, Some(0.0));

        assert!((out[0] - 1.0).abs() < 1e-9);
        assert!((out[1] - 1.5).abs() < 1e-9);
        assert!((out[2] - 2.0).abs() < 1e-9);
        assert!((zf - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_preemphasis_identity() {
        let y = array![0.0, 1.0, -2.0];

        let (x, _) = preemphasis(&y, Some(0.0), Some(0.0));

        assert!((x[0] - y[0]).abs() < 1e-9);
        assert!((x[1] - y[1]).abs() < 1e-9);
        assert!((x[2] - y[2]).abs() < 1e-9);
    }

    #[test]
    fn test_preemphasis_default_zi() {
        let y = array![2.0, 1.0];
        let coef = Some(0.5);

        let (out, _) = preemphasis(&y, coef, None);

        let zi = 2.0 * y[0] - y[1]; // default
        let expected0 = y[0] - 0.5 * zi;

        assert!((out[0] - expected0).abs() < 1e-9);
    }

    // Deemphasis
    #[test]
    fn test_deemphasis_identity() {
        let y = array![0.0,1.0,-2.0];
        let (x , _) = deemphasis(&y, Some(0.0), None);

        assert!((x[0] - y[0]).abs() < 1e-9);
        assert!((x[1] - y[1]).abs() < 1e-9);
        assert!((x[2] - y[2]).abs() < 1e-9);
    }

    #[test]
    fn test_deemphasis_basic() {
        let y = array![1.0, 2.0, 3.0];
        let coef = Some(0.5);
        let zi_in = Some(0.0);

        let (x, zf) = deemphasis(&y, coef, zi_in);

        assert!((x[0] - 1.0).abs() < 1e-9);
        assert!((x[1] - 2.5).abs() < 1e-9);
        assert!((x[2] - 4.25).abs() < 1e-9);
        assert!((zf - 4.25).abs() < 1e-9);
    }

    #[test]
    fn test_default_coef() {
        let y = array![1.0, 1.0];
        let (x1, _) = deemphasis(&y, None, Some(0.0));
        let (x2, _) = deemphasis(&y, Some(0.97), Some(0.0));

        assert!((x1[0] - x2[0]).abs() < 1e-9);
        assert!((x1[1] - x2[1]).abs() < 1e-9);
    }
    #[test]
    fn test_deemph_default_zi() {
        let y = array![2.0, 1.0];
        let coef = Some(0.5);

        let (x, _) = deemphasis(&y, coef, None);

        let expected_zi = (2.0 - 0.5) * 2.0 - 1.0 / (3.0 - 0.5);
        let expected_x0 = 2.0 + 0.5 * expected_zi;

        assert!((x[0] - expected_x0).abs() < 1e-9);
    }
}