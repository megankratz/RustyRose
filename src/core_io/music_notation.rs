use std::collections::HashMap;

/// Thaat mappings (basic scale degrees)
pub fn thaat_to_degrees(thaat: &str) -> Result<Vec<usize>, String> {
    let thaat_lower = thaat.to_lowercase();
    match thaat_lower.as_str() {
        "bilaval" => Ok(vec![0, 2, 4, 5, 7, 9, 11]),
        "khamaj"  => Ok(vec![0, 2, 4, 5, 7, 9, 10]),
        "kafi"    => Ok(vec![0, 2, 3, 5, 7, 9, 10]),
        "asavari" => Ok(vec![0, 2, 3, 5, 7, 8, 10]),
        "bhairavi"=> Ok(vec![0, 1, 3, 5, 7, 8, 10]),
        "kalyan"  => Ok(vec![0, 2, 4, 6, 7, 9, 11]),
        "marva"   => Ok(vec![0, 1, 4, 6, 7, 9, 11]),
        "poorvi"  => Ok(vec![0, 1, 4, 6, 7, 8, 11]),
        "todi"    => Ok(vec![0, 1, 3, 6, 7, 8, 11]),
        "bhairav" => Ok(vec![0, 1, 4, 5, 7, 8, 11]),
        _ => Err(format!("Unknown thaat: {}", thaat)),
    }
}

pub fn list_thaat() -> Vec<String> {
    vec![
        "bilaval", "khamaj", "kafi", "asavari", "bhairavi",
        "kalyan", "marva", "poorvi", "todi", "bhairav"
    ].into_iter().map(String::from).collect()
}

pub fn list_mela() -> HashMap<String, usize> {
    let names = vec![
        "kanakangi", "ratnangi", "ganamurthi", "vanaspathi", "manavathi", "tanarupi", "senavathi",
        "hanumathodi", "dhenuka", "natakapriya", "kokilapriya", "rupavathi", "gayakapriya",
        "vakulabharanam", "mayamalavagaula", "chakravakom", "suryakantham", "hatakambari",
        "jhankaradhwani", "natabhairavi", "keeravani", "kharaharapriya", "gaurimanohari",
        "varunapriya", "mararanjini", "charukesi", "sarasangi", "harikambhoji",
        "dheerasankarabharanam", "naganandini", "yagapriya", "ragavardhini", "gangeyabhushani",
        "vagadheeswari", "sulini", "chalanatta", "salagam", "jalarnavam", "jhalavarali",
        "navaneetham", "pavani", "raghupriya", "gavambodhi", "bhavapriya", "subhapanthuvarali",
        "shadvidhamargini", "suvarnangi", "divyamani", "dhavalambari", "namanarayani",
        "kamavardhini", "ramapriya", "gamanasrama", "viswambhari", "syamalangi",
        "shanmukhapriya", "simhendramadhyamam", "hemavathi", "dharmavathi", "neethimathi",
        "kanthamani", "rishabhapriya", "latangi", "vachaspathi", "mechakalyani", "chitrambari",
        "sucharitra", "jyotisvarupini", "dhatuvardhini", "nasikabhushani", "kosalam",
        "rasikapriya",
    ];
    let mut map = HashMap::new();
    for (i, &n) in names.iter().enumerate() {
        map.insert(n.to_lowercase(), i + 1);
    }
    map
}

pub fn mela_to_degrees(mela: &str) -> Result<Vec<usize>, String> {
    if let Ok(idx) = mela.parse::<usize>() {
        return mela_to_degrees_idx(idx);
    }
    let map = list_mela();
    if let Some(&idx) = map.get(&mela.to_lowercase()) {
        mela_to_degrees_idx(idx)
    } else {
        Err(format!("Unknown mela: {}", mela))
    }
}

pub fn mela_to_degrees_idx(mela: usize) -> Result<Vec<usize>, String> {
    if mela == 0 || mela > 72 {
        return Err(format!("mela={} must be in range [1, 72]", mela));
    }
    let index = mela - 1;
    let mut degrees = vec![0];
    let lower = index % 36;
    if lower < 6 { degrees.extend_from_slice(&[1, 2]); }
    else if lower < 12 { degrees.extend_from_slice(&[1, 3]); }
    else if lower < 18 { degrees.extend_from_slice(&[1, 4]); }
    else if lower < 24 { degrees.extend_from_slice(&[2, 3]); }
    else if lower < 30 { degrees.extend_from_slice(&[2, 4]); }
    else { degrees.extend_from_slice(&[3, 4]); }

    if index < 36 { degrees.push(5); }
    else { degrees.push(6); }

    degrees.push(7);
    let upper = index % 6;
    if upper == 0 { degrees.extend_from_slice(&[8, 9]); }
    else if upper == 1 { degrees.extend_from_slice(&[8, 10]); }
    else if upper == 2 { degrees.extend_from_slice(&[8, 11]); }
    else if upper == 3 { degrees.extend_from_slice(&[9, 10]); }
    else if upper == 4 { degrees.extend_from_slice(&[9, 11]); }
    else { degrees.extend_from_slice(&[10, 11]); }

    Ok(degrees)
}

pub fn mela_to_svara(mela: &str, abbr: bool, unicode: bool) -> Result<Vec<String>, String> {
    if let Ok(idx) = mela.parse::<usize>() {
        return mela_to_svara_idx(idx, abbr, unicode);
    }
    let map = list_mela();
    if let Some(&idx) = map.get(&mela.to_lowercase()) {
        mela_to_svara_idx(idx, abbr, unicode)
    } else {
        Err(format!("Unknown mela: {}", mela))
    }
}

fn mela_to_svara_idx(mela: usize, abbr: bool, unicode: bool) -> Result<Vec<String>, String> {
    if mela == 0 || mela > 72 {
        return Err(format!("mela={} must be in range [1, 72]", mela));
    }
    let index = mela - 1;
    let mut svara_map = vec![
        "Sa".to_string(), "Ri\u{2081}".to_string(), "".to_string(), "".to_string(),
        "Ga\u{2083}".to_string(), "Ma\u{2081}".to_string(), "Ma\u{2082}".to_string(),
        "Pa".to_string(), "Dha\u{2081}".to_string(), "".to_string(), "".to_string(),
        "Ni\u{2083}".to_string()
    ];
    let lower = index % 36;
    if lower < 6 { svara_map[2] = "Ga\u{2081}".to_string(); }
    else { svara_map[2] = "Ri\u{2082}".to_string(); }
    if lower < 30 { svara_map[3] = "Ga\u{2082}".to_string(); }
    else { svara_map[3] = "Ri\u{2083}".to_string(); }
    let upper = index % 6;
    if upper == 0 { svara_map[9] = "Ni\u{2081}".to_string(); }
    else { svara_map[9] = "Dha\u{2082}".to_string(); }
    if upper == 5 { svara_map[10] = "Dha\u{2083}".to_string(); }
    else { svara_map[10] = "Ni\u{2082}".to_string(); }
    
    if abbr {
        svara_map = svara_map.into_iter().map(|s| s.replace('a', "").replace('h', "").replace('i', "")).collect();
    }
    if !unicode {
        svara_map = svara_map.into_iter().map(|s| s.replace('\u{2081}', "1").replace('\u{2082}', "2").replace('\u{2083}', "3")).collect();
    }
    Ok(svara_map)
}

fn map_accidental_offset(c: char) -> i32 {
    match c {
        '#' | '♯' => 1,
        'b' | '!' | '♭' => -1,
        '𝄪' => 2,
        '𝄫' => -2,
        '♮' | 'n' => 0,
        _ => 0,
    }
}

fn note_to_pitch_class(note: char) -> i32 {
    match note.to_ascii_uppercase() {
        'C' => 0, 'D' => 2, 'E' => 4, 'F' => 5, 'G' => 7, 'A' => 9, 'B' => 11, _ => 0
    }
}

pub fn fifths_to_note(unison: &str, fifths: i32, unicode: bool) -> Result<String, String> {
    let cofmap = ['F', 'C', 'G', 'D', 'A', 'E', 'B'];
    let first_char = unison.chars().next().unwrap_or('C').to_ascii_uppercase();
    let acc_str: String = unison.chars().skip(1).collect();
    
    let mut offset = 0;
    for c in acc_str.chars() { offset += map_accidental_offset(c); }
    
    let circle_idx = cofmap.iter().position(|&x| x == first_char).unwrap_or(1) as i32;
    let raw_output = cofmap[((circle_idx + fifths).rem_euclid(7)) as usize];
    let acc_index = offset + (circle_idx + fifths).div_euclid(7);
    
    let abs_acc = acc_index.abs();
    let sign = acc_index.signum();
    let sym_double = if unicode { "𝄪" } else { "##" };
    let sym_single = if unicode { "♯" } else { "#" };
    let sf_double = if unicode { "𝄫" } else { "bb" };
    let sf_single = if unicode { "♭" } else { "b" };
    
    let mut out_acc = String::new();
    if sign > 0 {
        for _ in 0..(abs_acc / 2) { out_acc.push_str(sym_double); }
        for _ in 0..(abs_acc % 2) { out_acc.push_str(sym_single); }
    } else if sign < 0 {
        for _ in 0..(abs_acc / 2) { out_acc.push_str(sf_double); }
        for _ in 0..(abs_acc % 2) { out_acc.push_str(sf_single); }
    }
    
    Ok(format!("{}{}", raw_output, out_acc))
}

pub fn pythagorean_intervals(bins_per_octave: usize, sort: bool) -> Vec<f64> {
    let mut log_ratios: Vec<f64> = (0..bins_per_octave).map(|i| {
        let f = (i as f64) * 3.0_f64.log2();
        let mut frac = f.fract();
        if frac < 0.0 { frac += 1.0; }
        frac
    }).collect();
    if sort {
        log_ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    }
    log_ratios.into_iter().map(|v| 2.0_f64.powf(v)).collect()
}

fn parse_key_signature(key: &str) -> Result<(char, i32, String), String> {
    let parts: Vec<&str> = key.split(':').collect();
    if parts.len() != 2 { return Err(format!("Improper key format: {}", key)); }
    let tonic_str = parts[0];
    let scale = parts[1].to_lowercase();
    
    let note = tonic_str.chars().next().unwrap().to_ascii_uppercase();
    let acc_str: String = tonic_str.chars().skip(1).collect();
    let mut offset = 0;
    for c in acc_str.chars() { offset += map_accidental_offset(c); }
    Ok((note, offset, scale))
}

pub fn key_to_degrees(key: &str) -> Result<Vec<usize>, String> {
    let (note, offset, scale) = parse_key_signature(key)?;
    let scale_pre = scale[..std::cmp::min(3, scale.len())].to_string();
    
    let base_degrees = if scale_pre == "maj" || scale_pre == "ion" {
        vec![0, 2, 4, 5, 7, 9, 11]
    } else if scale_pre == "min" || scale_pre == "aeo" {
        vec![0, 2, 3, 5, 7, 8, 10]
    } else if scale_pre == "dor" {
        vec![0, 2, 3, 5, 7, 9, 10]
    } else if scale_pre == "phr" {
        vec![0, 1, 3, 5, 7, 8, 10]
    } else if scale_pre == "lyd" {
        vec![0, 2, 4, 6, 7, 9, 11]
    } else if scale_pre == "mix" {
        vec![0, 2, 4, 5, 7, 9, 10]
    } else if scale_pre == "loc" {
        vec![0, 1, 3, 5, 6, 8, 10]
    } else {
        return Err(format!("Unknown scale/mode: {}", scale));
    };
    
    let root = (note_to_pitch_class(note) + offset).rem_euclid(12) as usize;
    Ok(base_degrees.into_iter().map(|d| (d + root) % 12).collect())
}

pub fn key_to_notes(key: &str, unicode: bool, natural: bool) -> Result<Vec<String>, String> {
    let (note, offset, scale) = parse_key_signature(key)?;
    let root = (note_to_pitch_class(note) + offset).rem_euclid(12);
    
    let mut notes_sharp = vec!["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"]
        .into_iter().map(String::from).collect::<Vec<_>>();
    let mut notes_flat = vec!["C", "D♭", "D", "E♭", "E", "F", "G♭", "G", "A♭", "A", "B♭", "B"]
        .into_iter().map(String::from).collect::<Vec<_>>();
        
    let use_sharps = if offset > 0 { true } else if offset < 0 { false } else {
        if root < 6 { true } else { false }
    };
    
    let mut notes = if use_sharps {
        notes_sharp
    } else {
        notes_flat
    };
    
    if !unicode {
        notes = notes.into_iter()
            .map(|s| s.replace('♯', "#").replace('♭', "b").replace('𝄪', "##").replace('𝄫', "bb"))
            .collect();
    }
    
    if natural {
        let degrees = key_to_degrees(key).unwrap_or_default();
        for i in 0..12 {
            if !degrees.contains(&i) && notes[i].len() == 1 {
                let nat = if unicode { "♮" } else { "n" };
                notes[i].push_str(nat);
            }
        }
    }
    
    Ok(notes)
}

fn harmonic_distance(logs: &[f64], a: &[i32], b: &[i32]) -> f64 {
    let mut dist = 0.0;
    for i in 0..a.len() {
        let a_num = a[i].max(0);
        let a_den = (a_num - a[i]).abs();
        let b_num = b[i].max(0);
        let b_den = (b_num - b[i]).abs();
        let gcd = a_num.min(b_num) - a_den.max(b_den);
        dist += logs[i] * (a[i] + b[i] - 2 * gcd) as f64;
    }
    dist
}

pub fn plimit_intervals(primes: &[u32], bins_per_octave: usize, sort: bool) -> Vec<f64> {
    let logs: Vec<f64> = primes.iter().map(|&p| (p as f64).log2()).collect();
    let mut seeds = Vec::new();
    for i in 0..primes.len() {
        let mut s1 = vec![0; primes.len()];
        s1[i] = 1;
        seeds.push(s1.clone());
        let mut s2 = vec![0; primes.len()];
        s2[i] = -1;
        seeds.push(s2);
    }
    
    let mut frontier = seeds.clone();
    let mut intervals = vec![vec![0; primes.len()]];
    
    while intervals.len() < bins_per_octave {
        let mut score = std::f64::INFINITY;
        let mut best_f = 0;
        
        for (f, point) in frontier.iter().enumerate() {
            let mut hd = 0.0;
            for s in &intervals {
                hd += harmonic_distance(&logs, point, s);
            }
            if hd < score {
                score = hd;
                best_f = f;
            }
        }
        
        let new_point = frontier.remove(best_f);
        intervals.push(new_point.clone());
        
        for seed in &seeds {
            let mut cand = vec![0; primes.len()];
            for i in 0..primes.len() {
                cand[i] = new_point[i] + seed[i];
            }
            if !intervals.contains(&cand) && !frontier.contains(&cand) {
                frontier.push(cand);
            }
        }
    }
    
    let mut log_ratios: Vec<f64> = intervals.into_iter().map(|iv| {
        let mut sum = 0.0;
        for i in 0..primes.len() {
            sum += (iv[i] as f64) * logs[i];
        }
        let mut frac = sum.fract();
        if frac < 0.0 { frac += 1.0; }
        frac
    }).collect();
    
    if sort {
        log_ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    }
    
    log_ratios.into_iter().map(|v| 2.0_f64.powf(v)).collect()
}

pub fn interval_frequencies(
    n_bins: usize, fmin: f64, intervals_str: &str, bins_per_octave: usize, tuning: f64, sort: bool
) -> Vec<f64> {
    let ratios = match intervals_str {
        "equal" => (0..bins_per_octave)
            .map(|i| 2.0_f64.powf((tuning + i as f64) / (bins_per_octave as f64)))
            .collect(),
        "pythagorean" => pythagorean_intervals(bins_per_octave, sort),
        "ji3" => plimit_intervals(&[3], bins_per_octave, sort),
        "ji5" => plimit_intervals(&[3, 5], bins_per_octave, sort),
        "ji7" => plimit_intervals(&[3, 5, 7], bins_per_octave, sort),
        _ => return vec![],
    };
    
    let mut freqs = Vec::new();
    let bpo = ratios.len();
    if bpo == 0 { return vec![]; }
    
    let n_oct = (n_bins as f64 / bpo as f64).ceil() as usize;
    for o in 0..n_oct {
        for &r in &ratios {
            freqs.push(r * 2.0_f64.powi(o as i32));
        }
    }
    freqs.truncate(n_bins);
    if sort { freqs.sort_by(|a, b| a.partial_cmp(b).unwrap()); }
    
    freqs.into_iter().map(|r| r * fmin).collect()
}

pub fn interval_to_fjs(interval: f64, unison: &str, tolerance: f64, unicode: bool) -> Result<String, String> {
    Err("interval_to_fjs requires librosa msgpack dictionary. Not fully implemented in Rust.".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_thaat() {
        assert_eq!(list_thaat().len(), 10);
    }
    
    #[test]
    fn test_mela() {
        let degs = mela_to_degrees("kanakangi").unwrap();
        assert_eq!(degs, vec![0, 1, 2, 5, 7, 8, 9]);
    }
    
    #[test]
    fn test_pythagorean() {
        let iv = pythagorean_intervals(7, false);
        assert_eq!(iv.len(), 7);
    }
}
