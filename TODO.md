## TO DO 
---
### Project Structure 

- `src/lib.rs`
  - Main library entry point, re-export public modules/functions
- `src/core_io/`
  - Core audio/DSP
  - `mod.rs` organizes submodules
  - other files 
- `examples/`
  - Example programs for testing features manually

- keep each task in its own file
- expose modules through `mod.rs`
- re-export stable public functions through `lib.rs` ?

### How to Run

1. Build
```bash
cargo build
```
2. Run tests
```bash
cargo test
```
3. Run formatting 
```bash
cargo fmt
```
4. Run clippy
```bash
cargo clippy
```

Example:
```bash
cargo run --example first_ex
```

### Workflow 

1. Add module file
2. register with mod.rs
3. export public functions in lib.rs if needed?
4. add tests
5. run:
```bash
cargo fmt
cargo clippy
cargo test
```
---

### CHECK POINT 1
#### General 
- [ ] Finalize base folder structure
- [ ] Ensure project builds correctly with 'cargo build' ...
- [ ] Decide on any naming conventions etc
- [ ] add starter tests for existing modules
#### Holly 
- [ ] Frequency Unit Conversion
- [ ] Frequency range generation
#### Megan 
- [ ] Magnitude-scaling (power_to_db, amplitude_to_db)
- [ ] Feature manipulation (delta features, stacking, normalization utilities)
#### Jonathan  
- [ ] Audio Loading,
- [ ] Signal Generation
#### Alex
- [ ] Time Unit Converstion (Core IO and DSP)
- [ ] Miscellaneous (Core IO and DSP)

---

### CHECK POINT 2
#### General 
- [ ] 
- [ ]
- [ ]
#### Holly 
- [ ] Time-Domain Processing (auto-correlate, lpc) 
- [ ] Time-Domain Processing (zero-crossings, mu-compress, mu-expand)
#### Megan 
- [ ] MFCCs + Mel-spectrogram
- [ ] Various chroma features (at least chroma_stft)
#### Jonathan  
- [ ] Music Notation
- [ ] Spectral Features: Poly-features, and tonnetz
#### Alex
- [ ] Pitch and Tuning (Core IO and DSP)
- [ ] Miscellaneous effects (Effects)

---

### CHECK POINT 3
#### General 
- [ ]
#### Holly 
- [ ] Spectral-Representations (fft, ifft, reassinged spectrum)
#### Megan 
- [ ] plp
#### Jonathan  
- [ ]  Rhythm Features
#### Alex
- [ ] Time stretch and pitch shift (Effects)

---

... More when we get here 