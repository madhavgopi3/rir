# RIR Toolkit

**RIR Toolkit** is a Python framework for room impulse response measurement, acoustic descriptor extraction, and spatial acoustic analysis. It supports swept-sine generation, RIR extraction, octave-band analysis, descriptor calculation, plotting, CSV export, and heatmap visualisation across an 8 * 5 measurement grid.

## Features

* Generate exponential sine sweeps and inverse filters
* Process generated or external sweep recordings
* Align recordings using cross-correlation
* Extract RIRs through deconvolution
* Trim and normalise impulse responses
* Calculate EDT, T20, T30, C50, C80, D50, and centre time
* Apply Lundeby-style noise-floor estimation
* Perform octave and fractional-octave band analysis
* Generate plots, CSV files, and spatial heatmaps
* Includes a simple GUI for running the pipeline

## Installation

```bash
git clone https://github.com/your-username/open-rir-mapper.git
cd open-rir-mapper
pip install -r requirements.txt
```

## Usage

Recorded `.wav` files should be placed in the recordings folder, using grid-style names such as:

```text
A1.wav
A2.wav
B1.wav
C4.wav
E8.wav
```

Run the main pipeline:

```bash
python main.py
```

OR

Run the graphical interface:

```bash
python app.py
```

## Output

The program saves results in the `output/` folder, including:

* Raw and trimmed RIR audio files
* Recorded signal plots
* Spectrograms
* Frequency responses
* Energy decay curves
* Acoustic descriptor CSV files
* Spatial heatmaps

## Project Structure

```text
main.py                  Main processing pipeline
app.py                   Graphical user interface
config.py                Measurement and analysis global settings
sweep_gen.py             Sweep and inverse filter generation
audio_io.py              Audio loading, saving, and resampling
alignment.py             Sweep alignment
deconvolution.py         RIR extraction
rir_processing.py        RIR trimming and normalisation
acoustic_descriptors.py  Acoustic descriptor calculation
band_analysis.py         Octave and fractional-octave analysis
visualisation.py         Plotting and heatmap generation
external_sweep.py        External sweep workflow
```

Developed as part of a thesis project on open Python-based room acoustic measurement and spatial analysis.
