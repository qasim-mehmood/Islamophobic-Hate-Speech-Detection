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

- Dataset: ~1290 labeled tweets (positive = Islamophobic, negative = non-Islamophobic)  
- Preprocessing: Case folding, tokenization, stop-word removal, stemming  
- Embeddings: Word2Vec (300-dim)  
- Model: 7 Conv1D layers + 2 BiLSTM layers + Dense classifier  
- Achieved Accuracy: **90.13%**
---

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


