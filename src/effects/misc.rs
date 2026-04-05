use ndarray::{Array1, Array2, ArrayD, Axis, s, array, concatenate, arr1};
use crate::{core_io::*, feature::rms};

pub fn trim(
    y : &Array1<f64>,
    top_db: Option<f64>,
    ref_ : f64,
    frame_length : Option<i64>,
    hop_length : Option<i32>
) -> (Array1<f64>, Array1<i32>)
{
    let non_silent : Array1<bool> = signal_to_frame_nonsilent(y, frame_length, hop_length, top_db.unwrap_or(60.), ref_);

    let nonzero : Array1<usize> = Array1::from(non_silent.iter()
        .enumerate()
        .filter_map(|(i, &v)| if v {Some(i) } else {None})
        .collect::<Vec<_>>());

    let start: i32 = if nonzero.len() > 0 {
        frames_to_samples(nonzero[0] as i32, hop_length , None)
    } else {
        0
    };
    let end: i32 = if nonzero.len() > 0 {
        (*nonzero.last().unwrap() as i32 + 1).min(y.len() as i32 -1)
    } else {
        0
    };

    (y.slice(s![start..end]).to_owned(), Array1::from(array![start, end]))
}


pub fn split(
    y : &Array1<f64>,
    top_db: Option<f64>,
    ref_ : f64,
    frame_length : Option<i64>,
    hop_length : Option<i32>
) -> Array2<i32> {

    let non_silent : Array1<bool> = signal_to_frame_nonsilent(y, frame_length, hop_length, top_db.unwrap_or(60.), ref_);

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
}


pub fn remix(
    y : Array1<f64>,
    intervals: Array2<usize>,
    align_zeros : Option<bool>
) -> Array1<f64>{

    let mut out : Vec<f64> = Vec::new();

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



fn signal_to_frame_nonsilent(
    y : &Array1<f64>,
    frame_length : Option<i64>,
    hop_length : Option<i32>,
    top_db : f64,
    ref_ : f64
) -> Array1<bool>
{
    let mse : Array2<f64> = Array2::<f64>::zeros((frame_length.unwrap_or(2048) as usize,1000));
    // let mse = rms(y = y, frame_length = frame_length, hop_length = hop_length);

    let slice = mse.as_slice().expect("Array must be contiguous");
    let result = amplitude_to_db(slice, ref_, 1e-5, Some(top_db));

    let db : Array2<f64> = Array2::from_shape_vec(mse.raw_dim(), result).unwrap();

    let db_reduced: Array1<f64> = db.map_axis(Axis(0), 
    |col| {col.mean().unwrap()});

    db_reduced.mapv(|x| x > -top_db)
}