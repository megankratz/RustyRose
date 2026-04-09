use rusty_rose::core_io::*;

fn main()
{
    let freq: f32 = 440.0;
    let sr: i32 = 22050;

    let signal: Vec<f32> = (0..22050 * 2)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr as f32).sin())
        .collect();

    let output = yin(&signal, 400.0, 500.0, Some(sr), None, None, None, None);

    // Use median to stabilize pure sine
    let median_f0 = median_f32(&output);

    println!("Median f0 = {}", median_f0);

    // assert!((440.0 - median_f0).abs() < 10.0);
}

    /// Compute the median of a vector of f32
fn median_f32(values: &[f32]) -> f32 {
    let mut vals = values.to_vec();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = vals.len();
    if n % 2 == 0 {
        (vals[n / 2 - 1] + vals[n / 2]) / 2.0
    } else {
        vals[n / 2]
    }
}