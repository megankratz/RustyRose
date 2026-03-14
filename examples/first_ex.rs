use rusty_rose::core_io::fft_frequencies;

fn main() {
    let freqs = fft_frequencies(22050.0, 2048);
    println!("FFT bin 0: {} Hz, bin 1: {} Hz", freqs[0], freqs[1]);
}