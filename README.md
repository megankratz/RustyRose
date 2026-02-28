# RustyRose - An Audio and Music Processing Library in Rust Based on Librosa 

The main goal of this project will be a re-implementation of the librosa library in Python in Rust. 

The goal will be to provide full coverage, rigorous testing, and experimental benchmarking of the performance. In addition, various other types of audio and signal processing will be implemented and different ways of interfacing the system and integrating it with other packages and software. This mega-project is more focused on implementation and programming languages. Compatibility through mir_eval and jams, MIREX task implementation 

### TIMELINE

***Note**: This is a preliminary design specification to be filled out further when specific tasks and direction are assigned to the group. We are awaiting further instructions as to how this project will be implemented and what will be expected of the members*

|Project Setup |February  10, 2026
|---|---|
|Set up repository and project structure| All Team
| Write Design Specification | All Team
| Plan features and expectations | All Team

|Basic Tasks | Completion Date
|---|---|
| Set up cargo structure | Names
| Basic tasks (IO/DSP) | Names
| Impliment Tests - Ensure usability | Names

|Expected Tasks | Completion Date
|---|---|
| Expected Tasks | Names
| Misc basic / helper functions | Names
| Testing | Names

|Advanced Tasks | Completion Date
|---|---|
| Advanced Tasks | Names
| Collect Python Benchmarks | Names
| Testing | Names

|Analysis | Completion Date
|---|---|
|Any functionality| Names
|Collect Data| Names
|Perform analysis | Names

|Documentation and Presentation| Completion Date
|---|---|
| Finalize code and testing | Names
| Final report | Names
| Presentation | Names


### TOOLS
This project primarily uses the Rust language and its dependency manager Cargo. We may utilize existing Python tools from the librosa ecosystem where appropriate.

* Other Related Tools: mir_eval, jams, MIREX

### DATASETS

* [GTZAN Genre Collection](https://www.kaggle.com/datasets/carlthome/gtzan-genre-collection)
* [LibriSpeech ASR Corpus](https://www.openslr.org/12)
* [MedleyDB](https://medleydb.weebly.com/)

### ASSOCIATED LITERATURE
Supporting documents, manuals and resources:

* [Librosa Audio and Music Processing in Python](https://librosa.org/)
* [The Rust Programming Language](https://doc.rust-lang.org/stable/book/title-page.html)

### RELATED WORK
This project has librosa as the primary reference for implementation of this project. Librosa is a widely adopted Python based MIR toolkit. Other related systems are listed here:
* [Essentia](https://essentia.upf.edu/): An open-source C++ library for audio analysis and hardware-optimized Music Information Retrieval (MIR).
* [Aubio](https://aubio.org/): A C library focused on real-time audio labeling, including pitch estimation and beat tracking.
* [AudioFlux](https://github.com/libAudioFlux/audioFlux): A high-performance C++/Python library designed for deep learning-based audio feature extraction.
* [Rodio](https://github.com/RustAudio/rodio): The standard Rust audio playback library, utilizing a `Source` trait for audio streaming.
* [Hound](https://github.com/ruuda/hound): A lightweight, encoding/decoding library for WAV files in pure Rust.
  
RustyRose will differ from these projects by focusing on:
* Rust as the implementation language
* Explicit correctness testing
* Comparative benchmarking against Python tools 
* Understanding MIR pipelines at the system level 

### BIBLIOGRAPHY

[1] “librosa — librosa 0.8.0 documentation,” librosa.org. https://librosa.org/doc/latest/index.html‌

[2] C. Raffel et al., “mir_eval: A TRANSPARENT IMPLEMENTATION OF COMMON MIR METRICS.” Accessed: Feb. 07, 2024. [Online]. Available: https://colinraffel.com/publications/ismir2014mir_eval.pdf

[3] "mir_eval Documentation," Readthedocs.io, 2026. https://mir-eval.readthedocs.io/latest/index.html/latest/index.html (accesed Feb. 10, 2026

[4] “JAMS — jams 0.3.5 documentation,” Readthedocs.io, 2026. https://jams.readthedocs.io/en/stable/ (accessed Feb. 10, 2026).

[5] D. Bogdanov et al., “Essentia: An audio analysis library for music information retrieval,” in Proceedings of the 14th International Society for Music Information Retrieval Conference (ISMIR), Curitiba, Brazil, 2013, pp. 493–498.

[6] “Criterion.rs - Criterion.rs Documentation,” Github.io, 2025. https://bheisler.github.io/criterion.rs/book/criterion_rs.html
‌

[7] H. Lunnikivi, “Transpiling Python to Rust for Optimized Performance.” Accessed: Feb. 10, 2026. [Online]. Available: https://trepo.tuni.fi/bitstream/handle/10024/217564/transpiling_python_to_rust_for_optimized_performance.pdf?sequence=1

[8] py2many, “GitHub - py2many/py2many: Transpiler of Python to many other languages,” GitHub, 2025. https://github.com/py2many/py2many (accessed Feb. 10, 2026).
‌

[9] konchunas, “GitHub - konchunas/pyrs: Python to Rust transpiler,” GitHub, 2025. https://github.com/konchunas/pyrs (accessed Feb. 10, 2026).

[10] Xiao Hu et al., "The MIREX Grand Challenge: A Framework of Holistic User Experience Evaluation in Music Information Retrieval." Accessed: Feb. 10, 2026. [Online]. Available: https://researchcommons.waikato.ac.nz/server/api/core/bitstreams/70c370c7-88a8-45b1-8e15-5142aca05b80/content

[11] George Tzanetakis, 2001, "GTZAN Genre Collection" [Online]. Available: https://www.kaggle.com/datasets/carlthome/gtzan-genre-collection/data

[12] Vassil Panayotov, 2016, "LibriSpeech ASR Corpus". OpenSLR. [Online]. Available: https://www.openslr.org/12

[13] R. Bittner, J. Salamon, M. Tierney, M. Mauch, C. Cannam and J. P. Bello, "MedleyDB: A Multitrack Dataset for Annotation-Intensive MIR Research", in 15th International Society for Music Information Retrieval Conference, Taipei, Taiwan, Oct. 2014.

[14] ewan-xu, "GitHub - ewan-xu/Librosacpp: Librosahttps://github.com/ewan-xu/LibrosaCpp: LibrosaCpp" Accessed: Feb. 10, 2026. [Online] Available https://github.com/ewan-xu/LibrosaCpp (accessed Feb. 10, 2026).

[15]  M. Minin, “Enhancing Python Performance With Rust Integration,” 2025 International Conference on Industrial Engineering, Applications and Manufacturing (ICIEAM), pp. 879–883, May 2025, doi: https://doi.org/10.1109/icieam65163.2025.11028364.

[16] P. Mroczek, J. Mańturz, and M. Miłosz, “Comparative analysis of Python and Rust: evaluating their combined impact on performance,” Journal of Computer Sciences Institute, vol. 35, pp. 137–141, Jun. 2025, doi: https://doi.org/10.35784/jcsi.7050.


TASKS 

Core IO and DSP [https://librosa.org/doc/latest/core.html]

* (basic) Audio Loading,
* (basic) Signal Generattion - A
* (basic) Magnitude-scaling
* (basic) Frequency Unit Conversion - H
* (basic) Time Unit Converstion
* (basic) Frequency range generation - H
* (basic) Miscellaneous - A
* (expected) Time-Domain Processing 
* (expected) Magnitude-Scaling 
* (expected) Time-Domain Processing (auto-correlate, lpc) - H
* (expected) Time-Domain Processing (zero-crossings, mu-compress, mu-expand) - H
* (expected) Music Notation
* (expected) Pitch and Tuning - A
* (advanced) Spectral-Representations (fft, ifft, reassinged spectrum) - H
* (advanced) Spectral-Representations (all cqts) 
* (advanced) Spectral-Representations (iir, fmt, magphase)
* (advanced) Phase-recovery

Feature extraction [https://librosa.org/doc/latest/feature.html]

* (basic) Spectral Features: RMS, zerocrossings-rate, spectral centroid, rolloff, flatness, contrast, bandwidth
* (basic) Feature Manipulation 
* (expected) Spectral Features: Various chroma
* (expected) Spectral Features: MFCCs, Mel-spectrogram
* (expected) Spectral Features: Poly-features, and tonnetz
* (expected) Feature Inverssion 
* (advanced) Rhythm Features

Onset detection [https://librosa.org/doc/latest/onset.html]
* (basic) onset-detect
* (expected) onset-strength, onset-backtrack, onset-strength-multi

Beat track [https://librosa.org/doc/latest/beat.html]
* (advanced) beat track
* (advanced) plp

Effects [https://librosa.org/doc/latest/effects.html]
* (expected): Miscellaneous - A
* (advanced): time stretch and pitch shift - A

Temporal Segmentation [https://librosa.org/doc/latest/segment.html]
* (expected) cross-similarity, recurrance-matrix, recurrence to lag, lag to recurrence
* (advanced) time-lag filter, path-enhance
* (advanced) temporal-clustering

Sequential Modeling [https://librosa.org/doc/latest/sequence.html]
* (expected) transition matrices
* (advanced) viterbi stuff 
* (advanced) dtw, rqa
  

Utilities [https://librosa.org/doc/latest/util.html] 
* (expected) all functions


 
# Holly Gummerson
* PI1 (basic) Frequency Unit Conversion
* PI2 (basic) Frequency range generation
* PI3 (expected) Time-Domain Processing (auto-correlate, lpc) 
* PI4 (expected) Time-Domain Processing (zero-crossings, mu-compress, mu-expand)
* PI5 (advanced) Spectral-Representations (fft, ifft, reassinged spectrum) 

# Jonathan Ami 
* PI1 (basic) Audio Loading,
* PI2 (basic) Signal Generattion - A
* PI3 (expected) Music Notation
* PI4 (expected) Pitch and Tuning - A
* PI5 (advanced) Rhythm Features

# Megan Kratz
* PI1 (basic)
* PI2 (basic)
* PI3 (expected)
* PI4 (expected)
* PI5 (advanced)

# Alexander Williams 
* PI1 (Basic) Signal generation (Core IO and DSP)
* PI2 (Basic) Miscellaneous (Core IO and DSP)
* PI3 (Expected) Pitch and Tuning (Core IO and DSP)
* PI4 (Expected) Miscellaneous effects (Effects)
* PI5 (Advanced) Time stretch and pitch shift (Effects)
