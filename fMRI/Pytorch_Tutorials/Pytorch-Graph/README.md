# Graph Neural Networks for Brain Connectivity Analysis

This repository provides code to analyze brain connectivity using Graph Neural Networks (GNNs). The primary goal is to construct and visualize brain graphs derived from fMRI data, based on brain atlases and connectivity matrices. This analysis explores group-level differences in brain connectivity patterns between adults and children.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
- [Visualization](#visualization)
- [Next Steps](#next-steps)
- [References](#references)

## Overview

Graph Neural Networks (GNNs) are utilized here for their ability to model complex relationships between brain regions. Each node in the graph represents a brain region, and the edges represent functional connectivity between regions, based on fMRI data. The key steps include:

- Loading brain atlas and fMRI data.
- Transforming connectivity matrices into graph structures.
- Visualizing these graphs to interpret brain network patterns.


## Installation

To run the code, you'll need Python and the following libraries:
```bash
pip install numpy pandas networkx seaborn matplotlib plotly nilearn torch torch_geometric
```

## Data Preparation

### 1. Load Brain Atlas and fMRI Data

The code begins by loading the Difumo Atlas, which provides brain region maps, and an fMRI dataset. Each subject's connectivity matrix is extracted, and regions are classified as either "adult" or "child" for group-level analysis.

### 2. Connectivity Matrix Transformation

The fMRI data is transformed into connectivity matrices, which are then vectorized by discarding diagonal elements (self-connections) and retaining only the upper triangular part. This transformation reduces the dimensionality of the data while preserving essential connectivity information.

### 3. Group-Level Analysis

Connectivity matrices are averaged across subjects to produce group-level matrices for adults and children. These matrices are further analyzed to identify significant correlations between brain regions.

## Usage

### 1. Reconstruct Connectivity Matrices

The code reconstructs the full connectivity matrices from the vectorized features and separates them into adult and child groups:

```python
reshaped_features = vector_to_symmetric_matrix(X_features, matrix_size=64)
adult_matrix = reshaped_features[pheno['Adult'] == 1]
child_matrix = reshaped_features[pheno['Adult'] == 0]
```

### 2. Visualize 3D Connectome

The code provides a function to visualize the connectome in 3D using Plotly, highlighting significant connections:

```python
visualize_connectome(marker_corr.head(3), coords, marker_labels, cmap_markers='Paired', cmap_edges='cool')
```

### 3. Construct Brain Graphs

To build a graph network, the code uses the group-averaged connectome data and constructs a k-Nearest Neighbours (KNN) graph:

```python
mean_adult_graph = make_group_graph([conn], self_loops=False, k=8, symmetric=True)
```

## Visualization

Visualizations include both 3D connectomes and 2D graph networks based on Yeo's 17 network parcellations:

- **3D Connectome**: Shows connectivity patterns among brain regions.
- **Brain Graph**: Visualizes the graph with nodes colored according to their Yeo network classifications.

### Example: Yeo Networks Graph
```python
visualize_graph = brain_graph_Yeo17(mean_adult_graph, difumo_labels, color_map=None, layout='kamada_kawai', node_size=200, edge_width=1, alpha=0.2, figsize=(25, 25))
```

## Next Steps

The next steps involve integrating this graph structure into a GNN model, potentially using PyTorch Geometric, to predict outcomes or classify subjects based on their connectivity patterns.


![output](https://github.com/user-attachments/assets/45e9f127-94a8-434b-bb9a-052ed45c7a92)

