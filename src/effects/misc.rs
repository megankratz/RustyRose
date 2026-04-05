use ndarray::{Array1, Array2, Axis, s, array};
use crate::{core_io::*, feature::rms, utilities::frame};

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