/// Reference value for `power_to_db` — either a fixed scalar or a
/// function computed over the entire input (e.g. max or median).
pub enum Ref<'a> {
    Scalar(f64),
    Fn(&'a dyn Fn(&[f64]) -> f64),
}

/// Convert a power spectrogram (amplitude²) to decibel (dB) units.
pub fn power_to_db(s: &[f64], ref_: Ref<'_>, amin: f64, top_db: Option<f64>) -> Vec<f64> {
    assert!(amin > 0.0, "amin must be strictly positive");
    if let Some(t) = top_db {
        assert!(t >= 0.0, "top_db must be non-negative");
    }
    assert!(!s.is_empty(), "input must not be empty");

    let ref_val = match ref_ {
        Ref::Scalar(r) => r.abs(),
        Ref::Fn(f) => f(s).abs(),
    };
    let ref_clamped = ref_val.max(amin);

    let mut s_db: Vec<f64> = s
        .iter()
        .map(|&x| {
            let clamped = x.abs().max(amin);
            10.0 * (clamped / ref_clamped).log10()
        })
        .collect();

    if let Some(t) = top_db {
        let peak = s_db.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let floor = peak - t;
        for v in s_db.iter_mut() {
            if *v < floor {
                *v = floor;
            }
        }
    }

    s_db
}

/// Convert an amplitude spectrogram to a dB-scaled spectrogram.
pub fn amplitude_to_db(s: &[f64], ref_: f64, amin: f64, top_db: Option<f64>) -> Vec<f64> {
    assert!(amin > 0.0, "amin must be strictly positive");
    if let Some(t) = top_db {
        assert!(t >= 0.0, "top_db must be non-negative");
    }
    assert!(!s.is_empty(), "input must not be empty");

    let amin_power = amin * amin;
    let ref_power = ref_.abs() * ref_.abs();
    let ref_clamped = ref_power.max(amin_power);

    let mut s_db: Vec<f64> = s
        .iter()
        .map(|&x| {
            let magnitude = x * x;
            let clamped = magnitude.max(amin_power);
            10.0 * (clamped / ref_clamped).log10()
        })
        .collect();

    if let Some(t) = top_db {
        let peak = s_db.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let floor = peak - t;
        for v in s_db.iter_mut() {
            if *v < floor {
                *v = floor;
            }
        }
    }

    s_db
}


#[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_amplitude_to_db_unity() {
            // amplitude of 1.0 relative to ref=1.0 should be 0 dB
            let result = amplitude_to_db(&[1.0], 1.0, 1e-5, None);
            assert!((result[0] - 0.0).abs() < 1e-6);
        }

        #[test]
        fn test_amplitude_to_db_known_values() {
            // 0.1 amplitude → -20 dB, 0.01 → -40 dB
            let s = vec![0.1, 0.01];
            let result = amplitude_to_db(&s, 1.0, 1e-5, None);
            assert!((result[0] - (-20.0)).abs() < 1e-4);
            assert!((result[1] - (-40.0)).abs() < 1e-4);
        }

        #[test]
        fn test_amplitude_to_db_top_db_clamps() {
            // very small amplitude should be clamped to -80 dB
            let result = amplitude_to_db(&[1.0, 1e-10], 1.0, 1e-5, Some(80.0));
            assert_eq!(result[1], -80.0);
        }

        #[test]
        fn test_power_to_db_unity() {
            // power of 1.0 relative to ref=1.0 should be 0 dB
            let result = power_to_db(&[1.0], Ref::Scalar(1.0), 1e-10, None);
            assert!((result[0] - 0.0).abs() < 1e-6);
        }

        #[test]
        fn test_power_to_db_known_values() {
            // 0.01 power → -20 dB, 0.0001 → -40 dB
            let s = vec![0.01, 0.0001];
            let result = power_to_db(&s, Ref::Scalar(1.0), 1e-10, None);
            assert!((result[0] - (-20.0)).abs() < 1e-4);
            assert!((result[1] - (-40.0)).abs() < 1e-4);
        }

        #[test]
        fn test_power_to_db_ref_fn_max() {
            // with ref=max, the peak should always be 0 dB
            let s = vec![0.01, 0.1, 1.0];
            let result = power_to_db(&s, Ref::Fn(&|s| s.iter().cloned().fold(f64::NEG_INFINITY, f64::max)), 1e-10, None);
            assert!((result[2] - 0.0).abs() < 1e-6);
        }

        #[test]
        #[should_panic(expected = "amin must be strictly positive")]
        fn test_invalid_amin_panics() {
            amplitude_to_db(&[1.0], 1.0, 0.0, None);
        }

        #[test]
        #[should_panic(expected = "input must not be empty")]
        fn test_empty_input_panics() {
            amplitude_to_db(&[], 1.0, 1e-5, None);
        }
    }