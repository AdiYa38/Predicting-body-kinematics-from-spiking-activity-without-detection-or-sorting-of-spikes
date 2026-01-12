# Topographic Organization in the Hippocampus

This repository contains the codebase for analyzing neurophysiological data, specifically focusing on Unit Activity and Multi-Unit Activity (MUA). The project is structured to handle data importation, rate map generation, and location prediction (decoding).

## Project Structure

The codebase is organized into three main modules:
1. **General** (Core utilities and base logic)
2. **Unit Activity** (Single unit analysis)
3. **MUA** (Multi-Unit Activity analysis)

---

### 1. General (Core Modules)
These files form the foundation of the project, handling data ingestion, core calculations, and primary testing.

* **`data.py`**
    Functions for importing and preprocessing raw neurophysiological data from source files.
* **`heatmaps.py`**
    Core logic for generating, smoothing, and manipulating rate maps (heatmaps).
* **`prediction.py`**
    Algorithms and helper functions used for the location prediction  process.
* **`test.py`**
    A visualization script to inspect the rate map creation process and test the prediction pipeline using a single unit.
* **`test_prediction.py`**
    A comprehensive test script for running location predictions across multiple units.

### 2. Unit Activity
*Contains scripts specific to single-unit analysis.*



### 3. MUA (Multi-Unit Activity)
Files dedicated to the generation, analysis, and decoding of MUA signals.

* **`MUA.py`**
    Functions to create MUA signals from raw recordings and perform basic analysis.
* **`mua_test.py`**
    Test script for the MUA generation process. It also evaluates spatial consistency between different maps.
* **`mua_predictions.py`**
    Execution script for testing location predictions based specifically on MUA signals.
* **`MUA_corr.py`**
    Generates a correlation matrix between different MUA channels and exports the results to an Excel file.
* **`mua_ripple.py`**
    A specialized script that detects and removes ripple events from channels, then recalculates and prints the updated rate maps.
