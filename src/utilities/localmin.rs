pub fn localmin(frames: &Vec<Vec<f32>>) -> Vec<Vec<bool>> {

    let n_frames : usize = frames.len();
    let n_tau : usize= frames[0].len();

    let mut out :Vec<Vec<bool>> = vec![vec![false; n_tau]; n_frames];

    for f in 0..n_frames {

        for t in 1..n_tau-1 {

            let prev : f32 = frames[f][t-1];
            let cur : f32  = frames[f][t];
            let next : f32 = frames[f][t+1];

            if cur < prev && cur <= next {
                out[f][t] = true;
            }
        }
    }

    out
}