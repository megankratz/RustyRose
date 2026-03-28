use rustfft::{FftPlanner, num_complex::Complex};

pub fn autocorrelate(
    y_frames: &Vec<Vec<f32>>,
    max_size: usize,
) -> Vec<Vec<f32>> {

    let frame_length : usize = y_frames.len();
    let n_frames : usize = y_frames[0].len();

    let mut planner = FftPlanner::<f32>::new();

    let fft_size : usize = (2 * frame_length - 1).next_power_of_two();

    let fft = planner.plan_fft_forward(fft_size);
    let ifft = planner.plan_fft_inverse(fft_size);

    let mut result : Vec<Vec<f32>> = vec![vec![0.0f32; n_frames]; max_size];

    for frame in 0..n_frames {

        // Copy signal
        let mut buffer = vec![Complex{ re:0.0, im:0.0 }; fft_size];

        for i in 0..frame_length {
            buffer[i].re = y_frames[i][frame];
        }

        // Forward FFT
        fft.process(&mut buffer);

        // Power spectrum
        for v in &mut buffer {
            *v = *v * v.conj();
        }

        // Inverse FFT
        ifft.process(&mut buffer);

        for lag in 0..max_size {
            result[lag][frame] = buffer[lag].re / fft_size as f32;
        }
    }

    result
}

#[cfg(test)]
mod tests 
{
    use crate::audio::*;
    #[test]
    fn test_autocorr_delta() {
        let y = vec![
            vec![1.0],
            vec![0.0],
            vec![0.0],
            vec![0.0],
            vec![0.0],
        ];

        let ac = autocorrelate(&y, 5);

        assert!((ac[0][0] - 1.0).abs() < 1e-5);

        for k in 1..5 {
            assert!(ac[k][0].abs() < 1e-5);
        }
    }

    #[test]
    fn test_autocorr_constant() {
        let y = vec![
            vec![1.0],
            vec![1.0],
            vec![1.0],
            vec![1.0],
        ];

        let ac = autocorrelate(&y, 4);

        let expected = [4.0, 3.0, 2.0, 1.0];

        for i in 0..4 {
            assert!((ac[i][0] - expected[i]).abs() < 1e-3);
        }
    }

    #[test]
    fn test_multiple_frames() {

        let y = vec![
            vec![1.0, 2.0],
            vec![0.0, 3.0],
            vec![0.0, 4.0],
        ];

        let ac = autocorrelate(&y, 3);

        assert_eq!(ac.len(), 3);
        assert_eq!(ac[0].len(), 2);
    }
}