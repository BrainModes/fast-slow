<p align='center'>
    <img src= 'aux/header.jpg'>
</p>

<p align="left">
    <a href="https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12" alt="License-EUPL-1.2-or-later">
        <img src="https://img.shields.io/badge/license-EUPL--1.2--or--later-green" /></a>
</p>

# Learning how network structure shapes decision-making for bio-inspired computing
## Excitation-inhibition balance fitting of brain network models

Michael Schirner, Gustavo Deco & Petra Ritter
Berlin Institute of Health at Charité – Universitätsmedizin Berlin
Universitat Pompeu Fabra

Contact: [Michael Schirner](mailto:michael.schirner@bih-charite.de) and [Petra Ritter](mailto:petra.ritter@bih-charite.de)
Brain Simulation Group: [brainsimulation.org](https://www.brainsimulation.org/) 

## Overview

Complete code for Schirner, Deco & Ritter, "**Learning how network structure shapes decision-making for bio-inspired computing**", Nature Communications (2023). 
In this study we developed a new fitting algorithm for brain network models based on the Excitation-Inhibition balance between simulated brain areas. The algorithm fits brain models to empirical functional connectivity, based on a structural network description (typically obtained via diffusion-weighted MRI tractography).

## Contents

- brain-network-models/
	- Brain network model simulation code in C, Python and MATLAB
- data-preprocessing/
	- Code for processing of SC, FC, brain model input, and g-factor (Note: the code for computing the g-factor was adapted from Dubois et al. https://doi.org/10.1098/rstb.2017.0284)

## Usage

- **Large- and multi-scale model**: Both the large-scale model (379-regions full-brain parcellation based on Glasser/Human Connectome Project Atlas) and the multiscale model (PFC and PPC regions of the large-scale model are coupled to the DM circuit) are implemented in Python and C versions (see folder `brain-network-models`). We recommend to start with the Python version for initial familiarization as it is more easily readable than the C code which contains numerous optimizations for fast execution.
- **Compilation:** To compile the C code any current C compiler can be used (explicit linking with the `math.h` library needed, e.g. via `-lm`). For this study we used `GCC 9.4` with the following command: 
  ```
  gcc -Wall -Wextra -pedantic -std=c99 -msse2 -O3 -ftree-vectorize -ffast-math -funroll-loops -fomit-frame-pointer -m64 -lm -pthread multiscale-BNM-DM.c -o multiscale-BNM-DM
  ```
- **Input data**: Empirical connectome matrices (structural connectivity and functional connectivity) are brought into input format via the scripts `preprocess_HCP_data.m` and `Generate_BNM_input_LREFFI.m`.
- **MRI Processing**: 
	- **Structural connectivity (SC):** the scripts `step1_generate_scripts.m`, `step2_generate_diffusion_mask.m`, `step3_aggregateSC.m`, `step4_extractSC.m` provide the workflow to perform diffusion-weighted MRI tractography with [MRtrix](https://www.mrtrix.org/) on HCP data.
	- **Functional connectivity (FC):** the scripts `step1_generate_scripts.m` and `step2_extract_FC.m` perform the workflow to extract region-average fMRI time series and compute FC matrices.

## Data

All data used in this study was derived from the Human Connectome Project Young Adult study available in the repository https://db.humanconnectome.org/data/projects/HCP_1200. The derived data generated in this study are available under restricted access due to data privacy laws, access can be obtained within a timeframe of one month from the corresponding authors M.S. and P.R. as processing and sharing is subject to the European Union General Data Protection Regulation (GDPR), requiring a written data processing agreement, involving the relevant local data protection authorities, for compliance with the standard contractual clauses by the European Commission for the processing of personal data under GDPR ([https://commission.europa.eu/publications/standard-contractual-clauses-controllers-and-processors-eueea_en](https://commission.europa.eu/publications/standard-contractual-clauses-controllers-and-processors-eueea_en)). The data processing agreement and dataset metadata are available in EBRAINS ([https://search.kg.ebrains.eu/instances/88507924-8509-419f-8900-109accf1414b](https://search.kg.ebrains.eu/instances/88507924-8509-419f-8900-109accf1414b)).

## License

Copyright © 2023 Charité Universitätsmedizin Berlin. This software includes technology that is the subject of pending patent application No. 23 166 587.8 at the European Patent Office. The patent application has been filed by the Charité Universitätsmedizin Berlin. This software is licensed under the terms of the European Union Public Licence (EUPL) version 1.2 or later.

Please cite the Nature Communications paper when using the software.
