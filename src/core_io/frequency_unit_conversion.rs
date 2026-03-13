//use rusty_rose::core_io;
use regex::Regex;
//use std::{collections::HashMap, string};

// note_to_hz 
pub fn note_to_hz(note: &str) -> f32 {
    return midi_to_hz(note_to_midi(note, true).unwrap());
}

// note_to_midi and midi_to_hz are SKETCHY fyi
// note_to_midi
// [docs]def note_to_midi(
//     
//     if not isinstance(note, str):
//         return np.array([note_to_midi(n, round_midi=round_midi) for n in note])
//     pitch_map: Dict[str, int] = {
//         "C": 0,
//         "D": 2,
//         "E": 4,
//         "F": 5,
//         "G": 7,
//         "A": 9,
//         "B": 11,
//     }
//     acc_map: Dict[str, int] = {
//         "#": 1,
//         "": 0,
//         "b": -1,
//         "!": -1,
//         "♯": 1,
//         "𝄪": 2,
//         "♭": -1,
//         "𝄫": -2,
//         "♮": 0,
//     }
//     match = notation.NOTE_RE.match(note)
//     if not match:
//         raise ParameterError(f"Improper note format: {note:s}")
//     pitch = match.group("note").upper()
//     offset = np.sum([acc_map[o] for o in match.group("accidental")])
//     octave = match.group("octave")
//     cents = match.group("cents")
//     if not octave:
//         octave = 0
//     else:
//         octave = int(octave)
//     if not cents:
//         cents = 0
//     else:
//         cents = int(cents) * 1e-2
//     note_value: float = 12 * (octave + 1) + pitch_map[pitch] + offset + cents
//     if round_midi:
//         return int(np.round(note_value))
//     else:
//         return note_value

pub fn note_to_midi(note: &str, round_midi: bool) -> Result<f32, String> {
    let pitch_map = |c: char| -> Option<i32> {
        match c {
            'C' => Some(0),
            'D' => Some(2),
            'E' => Some(4),
            'F' => Some(5),
            'G' => Some(7),
            'A' => Some(9),
            'B' => Some(11),
            _ => None,
        }
    };

    let accidental_value = |c: char| -> Option<i32> {
        match c {
            '#' => Some(1),
            'b' => Some(-1),
            '!' => Some(-1),
            '♯' => Some(1),
            '♭' => Some(-1),
            '♮' => Some(0),
            '𝄪' => Some(2),
            '𝄫' => Some(-2),
            _ => None,
        }
    };

    // note letter, accidental(s), octave, optional cents
    let re = Regex::new(r"(?i)^([A-G])([#b!♯♭♮𝄪𝄫]*)(-?\d+)?([+-]\d+)?$")
        .map_err(|e| format!("regex error: {}", e))?;

    let caps = re
        .captures(note)
        .ok_or_else(|| format!("improper note format: {}", note))?;

    let pitch_char = caps
        .get(1)
        .ok_or_else(|| "missing note letter".to_string())?
        .as_str()
        .chars()
        .next()
        .unwrap()
        .to_ascii_uppercase();

    let pitch = pitch_map(pitch_char)
        .ok_or_else(|| format!("invalid pitch letter: {}", pitch_char))?;

    let accidental_str = caps.get(2).map_or("", |m| m.as_str());
    let mut offset = 0;

    for ch in accidental_str.chars() {
        offset += accidental_value(ch)
            .ok_or_else(|| format!("Invalid accidental: {}", ch))?;
    }

    let octave: i32 = match caps.get(3) {
        Some(m) => m.as_str().parse().map_err(|_| "Invalid octave".to_string())?,
        None => 0,
    };

    let cents: f32 = match caps.get(4) {
        Some(m) => {
            let cents_int: i32 = m.as_str().parse().map_err(|_| "Invalid cents".to_string())?;
            cents_int as f32 * 1e-2
        }
        None => 0.0,
    };

    let note_value = 12.0 * (octave as f32 + 1.0) + pitch as f32 + offset as f32 + cents;

    if round_midi {
        return Ok(note_value.round() as f32);
    } else {
        return Ok(note_value as f32);
    }

}

// midi_to_note
/*
[docs]@vectorize(excluded=["octave", "cents", "key", "unicode"])
def midi_to_note(
   
    if cents and not octave:
        raise ParameterError("Cannot encode cents without octave information.")

    note_map = notation.key_to_notes(key=key, unicode=unicode)

    # mypy does not understand vectorization, suppress type checks
    note_num = int(np.round(midi))  # type: ignore
    note_cents = int(100 * np.around(midi - note_num, 2))  # type: ignore

    note = note_map[note_num % 12]

    if octave:
        note = "{:s}{:0d}".format(note, int(note_num / 12) - 1)
    if cents:
        note = f"{note:s}{note_cents:+02d}"

    return note
*/

pub fn midi_to_note(midi: f32, octave: bool, cents: bool, key: &str, unicode: bool) -> String {
    if cents && !octave {
        panic!("Cannot encode cents without octave information.");
    }
    // currently using using sharps as the full implimentatino is not there yet !!!!
    // if key_to_note is implimentented, we can replace this with that function 
    let note_names = [
        "C", "C#", "D", "D#", "E", "F",
        "F#", "G", "G#", "A", "A#", "B"
    ];

    let note_num = midi.round() as i32;
    let note_cents = ((midi - note_num as f32) * 100.0).round() as i32;

    let pitch_class= note_num.rem_euclid(12) as usize;
    let mut note = note_names[pitch_class].to_string();

    if octave {
        let octave_num = note_num / 12 - 1;
        note.push_str(&octave_num.to_string());
    }
    if cents {
        note.push_str(&format!("{:+}", note_cents));
    }

    return note as String;
}

// midi_to_hz
pub fn midi_to_hz(notes: f32) -> f32 {
    return 440.0 * (2.0_f32.powf((notes - 69.0) / 12.0));
}

// hz_to_midi
pub fn hz_to_midi(frequencies: f32) -> f32 {
    let midi: f32 = 12.0 * (frequencies.log2() - 440.0_f32.log2()) + 69.0;
    return midi;
}

// hz_to_note
pub fn hz_to_note(frequencies: f32, octave: bool, cents: bool, key: &str, unicode: bool) -> String {
    return midi_to_note(hz_to_midi(frequencies), octave, cents, key, unicode);
}

// hz_to_mel
pub fn hz_to_mel(frequencies: f32, htk: bool) -> f32 {
    if htk {
        return 2595.0 * (1.0 + frequencies / 700.0).log10();
    }

    let f_min = 0.0;
    let f_sp = 200.0 / 3.0;

    let mut mels = (frequencies - f_min) / f_sp;

    let min_log_hz = 1000.0;
    let min_log_mel = (min_log_hz - f_min) / f_sp;
    let logstep = (6.4_f32).ln() / 27.0;

    if frequencies >= min_log_hz {
        mels = min_log_mel + (frequencies / min_log_hz).ln() / logstep;
    }

    return mels;
}

// mel_to_hz
pub fn mel_to_hz(mels: f32, htk: bool) -> f32 {
    if htk {
        return 700.0 * (10.0_f32.powf(mels / 2595.0) - 1.0);
    }

    let f_min = 0.0;
    let f_sp = 200.0 / 3.0;
    let mut freqs = f_min + f_sp * mels;

    let min_log_hz = 1000.0;
    let min_log_mel = (min_log_hz - f_min) / f_sp;
    let logstep = (6.4_f32).ln() / 27.0;

    if mels >= min_log_mel {
        freqs = min_log_hz * (logstep * (mels - min_log_mel)).exp();
    }

    return freqs;
}

// hz_to_octs
pub fn hz_to_octs(frequencies: f32, tuning: f32, bins_per_octs: i32) -> f32 {
    let affz = 440.0 * 2.0_f32.powf(tuning / bins_per_octs as f32);
    let octs: f32 = (frequencies / (affz / 16.0)).log2();
    return octs;
}

// octs_to_hz
pub fn octs_to_hz(octs: f32, tuning: f32, bins_per_octave: i32) -> f32 {
    let affz = 440.0 * 2.0_f32.powf(tuning / bins_per_octave as f32);
    return (affz / 16.0) * (2.0_f32.powf(octs));
}

// a4_to_tuning
pub fn a4_to_tuning(a4: f32, bins_per_octave: i32) -> f32 {
    let tuning: f32 = bins_per_octave as f32 * (a4.log2() - 440.0_f32.log2());
    return tuning;
}

// tuning_to_a4
pub fn tuning_to_a4(tuning: f32, bins_per_octave: i32) -> f32 {
    return 440.0 * 2.0_f32.powf(tuning / bins_per_octave as f32);
}

// TESTS
#[cfg(test)]
mod tests {
    use super::*;

    // note_to_hz
    #[test]
    fn test_note_to_hz_c4() {
        assert_eq!(note_to_hz("C4"), 261.62555);
    }
    #[test]
    fn test_note_to_hz_a4() {
        assert_eq!(note_to_hz("A4"), 440.0);
    }

    // note_to_hz
    #[test]
    fn test_note_to_midi_natural_note() {
        assert_eq!(note_to_midi("C", true), Ok(12.0));
        assert_eq!(note_to_midi("f4", true), Ok(65.0));
    }

    #[test]
    fn test_note_to_midi_accidentals() {
        assert_eq!(note_to_midi("C#3", true), Ok(49.0));
        assert_eq!(note_to_midi("Bb-1", true), Ok(10.0));
        assert_eq!(note_to_midi("A!8", true ), Ok(116.0));
    }

    #[test]
    fn test_note_to_midi_enharmonic_equivalents() {
        assert_eq!(note_to_midi("G𝄪6", true), Ok(93.0));
        assert_eq!(note_to_midi("B𝄫6", true), Ok(93.0));
        assert_eq!(note_to_midi("C♭𝄫5", true), Ok(69.0));
    }

    //midi_to_note
    #[test]
    fn test_midi_to_note_c4() {
        assert_eq!(midi_to_note(60.0, true, false, "C:maj", true), "C4");
    }
    #[test]
    fn test_midi_to_note_a4() {
        assert_eq!(midi_to_note(69.0, true, false, "C:maj", true), "A4");
    }
    // midi_to_hz
    #[test]
    fn test_midi_to_hz_a4() {
        assert_eq!(midi_to_hz(69.0), 440.0);
    }
    #[test]
    fn test_midi_to_hz_c4() {
        assert_eq!(midi_to_hz(60.0), 261.62555);
    }

}