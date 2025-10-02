# Social Index (Marimo) Notebook
Marimo notebook-based guide for behavioral analysis leveraging a cumulative Z-score index calculated from input metrics.

## Overview

An applet that provides an interactive dashboard built with [Marimo](https://marimo.io/) to analyze behavioral data, developed primarily for the operant social stress (OSS) procedure used in the Golden Lab (see [Navarrete and Schneider et al., 2024](https://www.sciencedirect.com/science/article/pii/S0006322324000337)), but should be adaptable to other behavioral tasks with subject-linked measurements.  

It allows users to upload a CSV dataset, select relevant metrics, and compute a cumulative Z-score index. The resulting index and its relationship with the input metrics can be explored through various embedded explorers and visualizations, with the ability to save parameters used and plots generated.


## Installation and Usage

First, clone or download the repository to your local machine.

### Using `uv`
Currently, installation steps have been tested primarily on Windows.

Follow the steps below with a terminal inside the cloned/downloaded repository folder to set up environment and run the notebook.

1. **Inside the cloned/downloaded folder**:
    ```bash
    uv sync
    ```
    This should create the environment and install the dependencies.

2. **Running the notebook**  
    
    Once installed, to run the notebook:
    ```bash
    uv run marimo run oss_app.py
    ```
    Or can also directly activate environment and then run:
    ```bash
    source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
    marimo run oss_app.py
    ```

### Using `conda`

1.  **Create and activate conda environment**:
    ```bash
    conda create -n sidx-env python=3.11
    conda activate sidx-env
    ```
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3. **Run the notebook**
    ```bash
    marimo run oss_app.py
    ```

### Running the App

Once the environment is set up and activated, run the Marimo notebook with the following command:



## WIP / Planned Features

-   [ ] Index distribution scatter plots
-   [ ] Index-metric correlation plots
-   [x] PCA biplots
-   [ ] Component correlation matrix

## License

This project is licensed under the MIT License.

## Authors

*   **Kevin N. Schneider**
    *   GitHub: [@kevsch88](https://github.com/kevsch88)
    *   Email: kevsch88@gmail.com
