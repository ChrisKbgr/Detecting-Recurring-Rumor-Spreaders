# Rumor Detection with Graph Neural Networks

This project implements a Graph Neural Network (GNN) approach to detect recurring rumor spreaders in social media using the PHEME dataset. The model leverages both user behavioral features and social network structure to identify users who consistently spread misinformation across multiple events.

## Dataset

This project uses the **PHEME dataset** which contains annotated Twitter threads from nine major news events. The dataset includes:
- Source tweets and their reactions
- Veracity annotations (rumor vs non-rumor)
- User metadata and engagement metrics
- Temporal information for propagation analysis

**Dataset Source**: [PHEME Dataset on Kaggle](https://www.kaggle.com/datasets/usharengaraju/pheme-dataset)

## Problem Definition

We define a **recurring rumor spreader** as a user who:
1. Participates in multiple news events
2. Has a rumor ratio > 0.6 (60% of their tweets are labeled as rumors)

The goal is to identify these persistent misinformation spreaders rather than individual false claims, enabling more efficient and proactive mitigation strategies.

## Environment Setup

### Option 1: Using conda (Recommended)

```bash
# Create conda environment
conda env create -f environment.yml

# Activate the environment
conda activate rumor_detection
```

### Option 2: Using pip

```bash
# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
├── model.py                          # Main GNN model implementation
├── pheme_dataset.py                  # Data preprocessing and feature extraction
├── convert_veracity_annotations.py   # Utility script for data conversion
├── requirements.txt                  # Python dependencies
├── environment.yml                   # Conda environment file
├── README.md                         # This file
├── gnn_architecture                  # Model architecture description
├── graphviz.png                      # Model visualization
└── pheme_user_features_multi_event.csv  # Sample processed data
```

## Usage

### 1. Data Preprocessing

If you have the full PHEME dataset, run the preprocessing script:

```bash
python pheme_dataset.py
```

This will:
- Extract user-level features from Twitter threads
- Compute behavioral and network metrics
- Generate sentiment scores using BERT
- Create the processed dataset for model training

### 2. Model Training and Evaluation

Run the main model script:

```bash
python model.py
```

This will:
- Load the processed data
- Train the GNN model with 5-fold cross-validation
- Compare against baseline models (Logistic Regression, SVM)
- Output performance metrics and results

## Model Architecture

The proposed model uses a **GraphSAGE** architecture with:

- **3 GraphSAGE layers** with 128 hidden units
- **Batch normalization** and **ReLU activation**
- **Dropout (0.2)** for regularization
- **Focal loss** to handle class imbalance
- **Adam optimizer** with learning rate 0.003

### Graph Construction

The model constructs a heterogeneous graph where:
- **Nodes**: Users with engineered features
- **Edges**: Social connections (replies/retweets) + k-nearest neighbors in feature space
- **Features**: 11 behavioral and network characteristics

### Key Features

- Total tweets, average tweet length, sentiment scores
- Activity duration, response times, engagement rates
- Network metrics: degree, clustering coefficient
- Rumor ratio and event participation

## Results

The GNN model achieves superior performance compared to traditional machine learning approaches:

| Model | Macro-F1 | Precision | Recall | ROC-AUC |
|-------|----------|-----------|--------|---------|
| Dummy Classifier | 0.506 | 0.506 | 0.506 | N/A |
| Logistic Regression | 0.508 | 0.540 | 0.555 | 0.572 |
| SVM | 0.492 | 0.538 | 0.552 | 0.581 |
| **GNN (Proposed)** | **0.626** | **0.627** | **0.628** | **0.667** |

**Key Improvements:**
- 23.2% improvement in Macro-F1 score over Logistic Regression
- Better balanced precision-recall performance
- Superior ROC-AUC and PR-AUC scores

## Dependencies

### Core Machine Learning
- `scikit-learn>=1.1.0` - Traditional ML models and evaluation
- `pandas>=1.5.0` - Data manipulation and analysis
- `numpy>=1.21.0` - Numerical computations

### Deep Learning & Graph Neural Networks
- `torch>=1.12.0` - PyTorch framework
- `torch-geometric>=2.2.0` - Graph neural network library

### Natural Language Processing
- `transformers>=4.20.0` - BERT for sentiment analysis
- `tokenizers>=0.13.0` - Text tokenization

### Data Processing & Visualization
- `tqdm>=4.64.0` - Progress bars
- `matplotlib>=3.5.0` - Plotting
- `seaborn>=0.11.0` - Statistical visualizations
- `python-dateutil>=2.8.0` - Date parsing

### Development
- `jupyter>=1.0.0` - Jupyter notebooks
- `ipykernel>=6.0.0` - Jupyter kernel
- `notebook>=6.4.0` - Notebook interface

## Key Features

- **Graph Neural Network (GraphSAGE)** for rumor spreader detection
- **5-fold cross-validation** with proper evaluation metrics
- **Comparison with traditional ML baselines** (Logistic Regression, SVM)
- **Comprehensive feature engineering** from social network data
- **BERT-based sentiment analysis** for text understanding
- **Network structure integration** for improved detection accuracy
- **Early stopping** and **threshold tuning** for optimal performance

## Limitations

- **Temporal bias**: Data limited to 2014-2015 events
- **Platform specificity**: Twitter-only data
- **Class imbalance**: Uneven distribution of rumor vs non-rumor tweets
- **Sampling bias**: May not represent broader Twitter population

## Future Work

- **Temporal dynamics**: Incorporate time-series analysis
- **Cross-platform generalization**: Test on other social media platforms
- **Interpretable GNNs**: Develop attention-based architectures
- **Real-time detection**: Adapt for streaming data
- **Causal analysis**: Investigate network position vs behavior causality

## Citation

If you use this code in your research, please cite:

```bibtex
@article{kuehberger2024detecting,
  title={Detecting Recurring Rumor Spreaders in the PHEME Dataset},
  author={Kühberger, Christian},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PHEME dataset creators and contributors
- PyTorch Geometric team for the GNN framework
- Hugging Face for the transformers library

