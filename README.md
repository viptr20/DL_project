# Child-Inspired Dual-Agent Self-Supervised Visual Learning

A deep learning project for visual representation learning on **COIL-100 dataset**, comparing three setups:

- **Supervised learning** (baseline)
- **Self-supervised learning** with ResNet-18 + projection head
- **Dual-agent self-supervised learning** inspired by social learning in children

The project also includes a **Streamlit web demo** for interactive image retrieval: the user uploads an image and the app returns the nearest visual neighbors from the COIL-100 dataset.

This repository was developed as a course project focused on learning meaningful visual embeddings, comparing training strategies, and demonstrating the results with both quantitative experiments and an interactive interface.

## Overview

The core idea is to study whether a dual-agent training setup can produce useful and more category-aware visual representations compared with a standard self-supervised encoder. The three model families are evaluated with linear-probe accuracy and clustering quality, and their behavior is further inspected through nearest-neighbor retrieval in the Streamlit application.

The dual-agent variant is motivated by social learning: two agents process related views, share information indirectly through training objectives, and are encouraged to build more stable representations. This makes the project suitable both as a representation-learning study and as a compact end-to-end demo combining training, evaluation, and deployment.

## Features

- ResNet-18 baseline trained with labels
- Self-supervised ResNet-18 with projection head
- Dual-agent model with shared learning signals
- Linear-probe evaluation on frozen features
- Clustering evaluation with ARI
- Streamlit application for image-to-image retrieval
- Support for comparing qualitative retrieval behavior across models

## Repository Structure

```text
.
├── checkpoints/              # Saved model weights
├── config/                   # Configuration objects and paths
├── data/                     # COIL-100 dataset location
├── reports/                  # Analysis materials
├── src/
│   ├── dataset.py            # COIL-100 dataset wrapper
│   ├── models.py             # ResNet-18 SSL and dual-agent models
│   ├── train.py              # Training entry point(s)
│   ├── eval.py               # Linear-probe / clustering evaluation
    |__ app_ssl.py            # Streamlit web application for SSL
│   └── app_dual.py           # Streamlit web application for Dual-agent
                  
├── experiment_report.csv     # Experiment results table
├── log_experiments_to_csv.py 
└── README.md
```

The exact file names may vary slightly across local versions, but the project follows the common machine-learning repository pattern of separating data loading, model definitions, training, evaluation, and demo code.

## Methods

### 1. Supervised learning

The supervised baseline trains a ResNet-18 directly on object labels from COIL-100. Its purpose is to provide an upper reference point for classification-oriented performance and to show what can be achieved when labels are available.

### 2. Self-supervised learning

The self-supervised setup uses a ResNet-18 encoder with a projection head and contrastive learning. Different augmented views of the same image are encouraged to produce similar feature vectors, while views from different objects are pushed apart. After pretraining, the encoder is frozen and evaluated with a linear probe.

### 3. Dual-agent self-supervised learning

The dual-agent setup extends the self-supervised idea with two agents, typically referred to as Agent A and Agent B. The agents learn from related views and are encouraged to maintain compatible representations, with the goal of producing smoother and more category-aware embeddings than a single-agent SSL model.

In the project concept, this setup is linked to child-inspired social learning: difficult or uncertain examples can be emphasized through curiosity-driven replay, and agreement between agents can act as an additional training signal.

## Dataset

The experiments use **COIL-100**, a multi-view object dataset containing images of 100 objects captured from multiple viewpoints. It is well suited to studying representation learning because the same object appears under many rotations, which makes it useful for classification, clustering, and nearest-neighbor retrieval.

## Evaluation

The project uses two main quantitative metrics:

- **Linear-probe accuracy**: measures how well a simple linear classifier performs on frozen learned features
- **ARI (Adjusted Rand Index)**: measures clustering quality by comparing predicted group structure with ground-truth labels

These two metrics complement each other: linear-probe accuracy reflects class separability for downstream classification, while ARI reflects how cleanly the feature space organizes objects into clusters.

## Experimental Results

Based on the current experiment table, the following trends emerge:

- The **learning-with-a-teacher** baseline gives the strongest overall scores, reaching nearly perfect linear-probe accuracy and the best clustering quality.
- The **self-supervised** model performs remarkably close to the supervised baseline in linear-probe accuracy, showing that useful representations can be learned without labels.
- The **dual-agent** model scores slightly lower on linear-probe accuracy, but in qualitative retrieval it can produce more category-oriented neighbors rather than only near-duplicates of the exact same object.

In other words, the quantitative results favor the supervised baseline, while the qualitative demo highlights a possible representational advantage of the dual-agent setup for broader semantic similarity.

## Streamlit Demo

The repository includes a Streamlit application for interactive visual retrieval. Streamlit turns Python scripts into shareable web apps and is commonly used for machine-learning demos and lightweight model interfaces.

### Demo behavior

1. The user uploads an image.
2. The application computes its feature representation with the selected model.
3. The representation is normalized.
4. Cosine similarity is computed against precomputed COIL-100 feature vectors.
5. The top nearest neighbors are displayed in the browser.

This demo is especially useful for comparing the qualitative behavior of the self-supervised and dual-agent models. In practice, the SSL model tends to retrieve several views of the exact same object, while the dual-agent version may sometimes include other visually similar objects from the same broader category.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the project

### Train a model

Adjust the configuration in the project files and run the appropriate training script, for example:

```bash
python src/train.py
```

If the repository separates training scripts by method, run the corresponding file for:

- supervised training
- self-supervised training
- dual-agent training

### Run evaluation

```bash
python src/eval.py
```

### Run the Streamlit app

```bash
streamlit run src/app_ssl.py OR 
streamlit run src/app_dual.py
```

Then open the local URL printed in the terminal, usually:

```text
http://localhost:8501
```

## Notes on checkpoints

Make sure the selected checkpoint matches the model architecture:

- `ResNet18SSL` must load an SSL checkpoint
- `DualAgentModel` must load a dual-agent checkpoint

A mismatch between checkpoint structure and model class will raise a PyTorch `state_dict` loading error. This is especially important when switching the app between SSL and dual-agent retrieval modes.

## Example findings

A representative observation from the experiments and demo is the following:

- The SSL model often returns multiple views of the same physical object with very high cosine similarity.
- The dual-agent model can still retrieve the same object, but may also include another semantically similar object in the top results.

This suggests that the dual-agent setup may encourage slightly broader category-level grouping, even when it does not outperform the single-agent SSL model on standard linear-probe accuracy.

## Future Work

Three natural next steps for the project are:

1. **Richer social mechanisms**: asymmetric agent roles, curiosity-driven replay, and stronger sample selection strategies.
2. **Broader datasets**: multi-view extensions, video sequences, and unseen objects to test generalization.
3. **Expanded demo**: direct model comparison, dimensionality-reduction visualizations, and user-assisted grouping/labeling.

## Tech Stack

- Python
- PyTorch
- Torchvision
- Streamlit
- PIL / Pillow
- NumPy / Pandas / scikit-learn (depending on evaluation pipeline)

## Acknowledgments

- COIL-100 dataset creators
- PyTorch ecosystem for training and inference tooling
- Streamlit for fast interactive demo development

## Citation

If you use or reference this repository in academic work, cite the project repository and describe the experimental setup clearly, including the dataset, training regime, and evaluation protocol.
