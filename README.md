# Islamophobic-Hate-Speech-Detection
Islamophobic Hate Speech Detection from Electronic Media
This repository contains the implementation of the paper:

**Islamophobic Hate Speech Detection from Electronic Media using Deep Learning**  
*Qasim Mehmood, Anum Kaleem, Imran Siddiqi*  
Published at *5th Mediterranean Conference on Pattern Recognition and Artificial Intelligence*
--

## 🚀 Overview
We propose a deep learning-based approach for detecting Islamophobic hate speech in tweets.  
The method combines **1D Convolutional Neural Networks (CNN)** for feature extraction and **Bi-directional LSTM** for classification.

- Dataset: labeled tweets (positive = Islamophobic, negative = non-Islamophobic)  
- Preprocessing: Case folding, tokenization, stop-word removal, stemming  
- Embeddings: Word2Vec (300-dim)  
- Model: 7x Conv1D layers + 2x BiLSTM layers + Dense classifier  
- Achieved Accuracy: **90.13%**
---

Islamophobic-Hate-Speech-Detection/
│── README.md                # Overview of the project
│── requirements.txt          # Python dependencies
│── environment.yml           # (optional) for Conda users
│── LICENSE                   # License for usage
│── .gitignore                # Ignore unnecessary files
│
├── data/                     # Dataset folder (add .gitignore if dataset is large/private)
│   ├── raw/                  # Original tweets dataset
│   ├── processed/            # Preprocessed tweets
│   └── README.md             # Dataset description
│
├── src/                      # Source code, All stages of Islamophobic Hate Speech detection are provided in py code.
│   ├── preprocessing.py      # Data cleaning, tokenization, stop-word removal
│   ├── embeddings.py         # Word2Vec or GloVe embeddings
│   ├── model.py              # CNN + BiLSTM implementation
│   ├── train.py              # Training pipeline
│   ├── evaluate.py           # Evaluation metrics (accuracy, F1, confusion matrix)
│   └── utils.py              # Helper functions
│
├── notebooks/                # Jupyter notebooks (exploration & experiments)
│   └── experiments.ipynb
│
├── results/                  # Store experiment results
│   ├── figures/              # Accuracy/loss curves, confusion matrix
│   └── logs/                 # Training logs
│
└── paper/                    # Paper-related files
    └── Islamophobic_Hate_Speech.pdf



## 🗂 Repository Contents
- `src/` → Preprocessing, model, training, and evaluation scripts  
- `data/` → Dataset (not uploaded here due to size/privacy, see instructions)  
- `notebooks/` → Experiment Jupyter notebooks  
- `results/` → Accuracy/loss curves, confusion matrices  
- `paper/` → Original published paper (PDF)  

---

## ⚙️ Installation
```bash
git clone https://github.com/yourusername/Islamophobic-Hate-Speech-Detection.git
cd Islamophobic-Hate-Speech-Detection


📌 Citation
If you use this code, please cite:

@inproceedings{mehmood2021islamophobic,
  title={Islamophobic Hate Speech Detection from Electronic Media using Deep Learning},
  author={Mehmood, Qasim and Kaleem, Anum and Siddiqi, Imran},
  booktitle={5th Mediterranean Conference on Pattern Recognition and Artificial Intelligence},
  year={2021}
}


