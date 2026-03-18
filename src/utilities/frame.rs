pub fn frame(y : Vec<f32>, frame_length : i32, hop_length : i32) -> Vec<Vec<f32>>{
    
    if y.len() < frame_length as usize {
        panic!("Input is too short (n={}) for frame_length={}", y.len(), frame_length);
    }
    if hop_length < 1 {
        panic!("Invalid hop_length: {}", hop_length);
    }

    let mut output : Vec<Vec<f32>> = vec![];

    let num_frames : i32 = (y.len() as i32 - frame_length) / hop_length + 1;

    for i in 0..frame_length {
        let indices : Vec<i32> = (0..num_frames)
        .map(|x| i + (x*hop_length))
        .take_while(|&x| x < y.len() as i32)
        .collect();

        output.push(indices.iter().map(|&i| y[i as usize]).collect());

    }

    output
    
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame() {
        let y : Vec<f32> = (0..10).map(|x| x as f32 / 10.0).collect();

        let frame : Vec<Vec<f32>> = frame(y, 5, 2);
        let expected : Vec<Vec<f32>> = vec![vec![0.0,0.2,0.4],
                                        vec![0.1,0.3,0.5],
                                        vec![0.2,0.4,0.6],
                                        vec![0.3,0.5,0.7],
                                        vec![0.4,0.6,0.8]];
        assert_eq!(expected, frame);
    }

    #[test]
    fn test_frame_lower_bound_length() {
        let y : Vec<f32> = (0..10).map(|x| x as f32 / 10.0).collect();

        let frame : Vec<Vec<f32>> = frame(y, 6, 4);
        let expected : Vec<Vec<f32>> = vec![vec![0.0,0.4],
                                        vec![0.1,0.5],
                                        vec![0.2,0.6],
                                        vec![0.3,0.7],
                                        vec![0.4,0.8],
                                        vec![0.5,0.9]];
        assert_eq!(expected, frame);
    }

    #[test]
    fn test_frame_upper_bound_length() {
        let y : Vec<f32> = (0..10).map(|x| x as f32 / 10.0).collect();

        let frame : Vec<Vec<f32>> = frame(y, 7, 4);
        let expected : Vec<Vec<f32>> = vec![vec![0.0],
                                        vec![0.1],
                                        vec![0.2],
                                        vec![0.3],
                                        vec![0.4],
                                        vec![0.5],
                                        vec![0.6]];
        assert_eq!(expected, frame);
    }
}