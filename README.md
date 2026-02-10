# RustyRose - An Audio and Music Processing Library in Rust Based on Librosa 

The main goal of this project will be a re-implementation of the librosa library in Python in Rust. 

The goal will be to provide full coverage, rigorous testing, and experimental benchmarking of the performance. In addition, various other types of audio and signal processing will be implemented and different ways of interfacing the system and integrating it with other packages and software. This mega-project is more focused on implementation and programming languages. Compatibility through mir_eval and jams, MIREX task implementation 

### TIMELINE

|Project Setup |Feburary 17, 2026
|---|---|
|Set up repository and project structure|Names
| Write Design Specification | Names
| Plan features and expectations | Names



|Foundations |Mnth, date, yr
|---|---|
| Select DSP and dependencies | Names
| Impliment audio loading and preprocessing | Names
| Impliment window functions and framing | Names

##### *MORE HERE WITH DATES*

|Documentation and Presentation| Mnth, date, year
|---|---|
| Finalize code and testing | Names
| Final report | Names
| Presentation | Names


### TOOLS
This project primarily uses the Rust language and its dependency manager Cargo. We may utilze existing Python tools from the librosa ecosystem where appropriate.

* *decide on libraries used, fft, audio, benchmarking, testing, reference implimentations*
* mir_eval, jams, MIREX

### DATASETS

* [GTZAN Genre Collection](https://www.kaggle.com/datasets/carlthome/gtzan-genre-collection)
* [LibriSpeech ASR Corpus](https://www.openslr.org/12)
* [MedleyDB](https://medleydb.weebly.com/)

### ASSOCIATED LITERATURE
Supporting documents, manuals and resources:

* [Librosa Audio and Music Processing in Python](https://librosa.org/)
* [The Rust Programming Language](https://doc.rust-lang.org/stable/book/title-page.html)

### RELATED WORK
This project has librosa as the primary reference for implimentation of this project. Librosa is a widely adopted Python based MIR toolkit. Other related systems are listed here:
* [Essentia](https://essentia.upf.edu/): An open-source C++ library for audio analysis and hardware-optimized Music Information Retrieval (MIR).
* [Aubio](https://aubio.org/): A C library focused on real-time audio labeling, including pitch estimation and beat tracking.
* [AudioFlux](https://github.com/libAudioFlux/audioFlux): A high-performance C++/Python library designed for deep learning-based audio feature extraction.
* [Rodio](https://github.com/RustAudio/rodio): The standard Rust audio playback library, utilizing a `Source` trait for audio streaming.
* [Hound](https://github.com/ruuda/hound): A lightweight, encoding/decoding library for WAV files in pure Rust.
  
RustyRose will differ from these projects by focusing on:
* Rust as the implementation language
* Explicit correctness testing
* Comparative benchmarking against Python tools 
* Understanding MIR pipelines ar the system level 

### BIBLIOGRAPHY
*these are just some im finding, we will edit these out and make them real citations if they are applicable*
4. [librosa: Audio and Music Signal Analysis in Python](https://d1wqtxts1xzle7.cloudfront.net/40296500/librosa-libre.pdf?1448292194=&response-content-disposition=inline%3B+filename%3Dlibrosa_Audio_and_Music_Signal_Analysis.pdf&Expires=1770433380&Signature=Ro4rQEi7jX-JrRUVgDSHEZEfDLvF-20-LUY3hHCgZw2CCC1ibdunocnnhr~BAAdhygxHfdbyQs-BEfbvjHazLXmkPpsf5CVYmV8wYV89i2DiFfyFPLLZjfnae4YjpPWtZ8~05zdFlZhvoiVGYyQnlJgTs8AGF0agJEIoJBodKij2ScomszLrANOxNGwzpIU6sQnIEJVHaaEI10ZSqlR6jtbYS91u3jOKDuWbxlS5bxU1nyj0q4IWpUF9mvL2oc9CR8vCljraU8fHQ8I3p2wf~ipEqNqWf2c9-sy0VpFWLimINWSQihu0qyN8eGcmGn2E4udXN1SFJwaW4ohvPg6E4w__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA)
7.[TORCHAUDIO 2.1: ADVANCING SPEECH RECOGNITION, SELF-SUPERVISED LEARNING,
AND AUDIO PROCESSING COMPONENTS FOR PYTORCH](https://arxiv.org/pdf/2310.17864)


[1]“librosa — librosa 0.8.0 documentation,” librosa.org. https://librosa.org/doc/latest/index.html
‌
[2]C. Raffel et al., “mir_eval: A TRANSPARENT IMPLEMENTATION OF COMMON MIR METRICS.” Accessed: Feb. 07, 2024. [Online]. Available: https://colinraffel.com/publications/ismir2014mir_eval.pdf

[3]“JAMS — jams 0.3.5 documentation,” Readthedocs.io, 2026. https://jams.readthedocs.io/en/stable/ (accessed Feb. 10, 2026).

[4]D. Bogdanov et al., “Essentia: An audio analysis library for music information retrieval,” in Proceedings of the 14th International Society for Music Information Retrieval Conference (ISMIR), Curitiba, Brazil, 2013, pp. 493–498.

[5]“Criterion.rs - Criterion.rs Documentation,” Github.io, 2025. https://bheisler.github.io/criterion.rs/book/criterion_rs.html
‌
[6]H. Lunnikivi, “Transpiling Python to Rust for Optimized Performance.” Accessed: Feb. 10, 2026. [Online]. Available: https://trepo.tuni.fi/bitstream/handle/10024/217564/transpiling_python_to_rust_for_optimized_performance.pdf?sequence=1

[7]py2many, “GitHub - py2many/py2many: Transpiler of Python to many other languages,” GitHub, 2025. https://github.com/py2many/py2many (accessed Feb. 10, 2026).
‌
[8]konchunas, “GitHub - konchunas/pyrs: Python to Rust transpiler,” GitHub, 2025. https://github.com/konchunas/pyrs (accessed Feb. 10, 2026).


# Holly Gummerson
* PI1 (basic)
* PI2 (basic)
* PI3 (expected)
* PI4 (expected)
* PI5 (advanced)

# Jonathan Ami 
* PI1 (basic)
* PI2 (basic)
* PI3 (expected)
* PI4 (expected)
* PI5 (advanced)

# Firstname Last 
* PI1 (basic)
* PI2 (basic)
* PI3 (expected)
* PI4 (expected)
* PI5 (advanced)

# Firstname Last 
* PI1 (basic)
* PI2 (basic)
* PI3 (expected)
* PI4 (expected)
* PI5 (advanced)