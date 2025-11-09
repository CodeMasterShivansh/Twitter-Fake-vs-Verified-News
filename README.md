**Twitter Fake vs Verified News Detection**
**Overview**

This project is part of an academic research study focused on detecting fake vs verified news on Twitter using machine learning and transformer-based models.
It compares the performance of traditional models (Logistic Regression, SVM) and pre-trained transformer models (BERT, RoBERTa), followed by the development of a hybrid SVM + BERT ensemble for improved accuracy.

**Dataset**

The dataset used is from FakeNewsNet (PolitiFact subset)
, a publicly available repository developed by Arizona State University.

| **Label** | **Samples** | **Avg Title Length** | **Source File**       |
| --------- | ----------- | -------------------- | --------------------- |
| Fake News | 2,345       | 12.4                 | `politifact_fake.csv` |
| Real News | 2,450       | 13.1                 | `politifact_real.csv` |

**Methodology**

Data Preprocessing

•	Text cleaning and normalization

•	Tokenization using Hugging Face tokenizer

•	TF-IDF feature extraction for traditional models

•	Embedding generation for transformer models

Models Used

•	Logistic Regression

•	Support Vector Machine (SVM)

•	BERT (base-uncased)

•	RoBERTa (base)

•	Hybrid SVM + BERT Ensemble

Hybrid Approach Formula

                  P_hybrid = 0.5 × P_SVM + 0.5 × P_BERT​

| **Model**                    | **Accuracy (%)** | **Precision** | **Recall** | **F1-Score** |
| ---------------------------- | ---------------- | ------------- | ---------- | ------------ |
| TF-IDF + Logistic Regression | 80.41            | 0.8723        | 0.6406     | 0.7387       |
| TF-IDF + SVM                 | 84.46            | 0.8361        | 0.7969     | 0.8160       |
| BERT (bert-base-uncased)     | 89.86            | 0.8657        | 0.9062     | 0.8855       |
| RoBERTa (roberta-base)       | 88.51            | 0.8406        | 0.9062     | 0.8722       |
| **Hybrid (SVM + BERT)**      | **95.20**        | **0.95**      | **0.95**   | **0.95**     |


 The hybrid model achieved the highest accuracy of 95.2%.

**Project Structure**

Twitter-Fake-vs-Verified-News/
│
├── data/
│   ├── politifact_fake.csv
│   ├── politifact_real.csv
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_evaluation.ipynb
│
├── results/
│   ├── model_comparison.png
│   ├── confusion_matrix.png
│
├── report/
│   ├── Twitter_Fake_vs_Verified_News_Research_Paper.pdf
│
└── README.md


**Technologies Used**

•	Python

•	Scikit-learn

•	Hugging Face Transformers

•	TensorFlow / PyTorch

•	Matplotlib

•	Pandas, NumPy


**Citation**

If you use this work, please cite:

S. Vosoughi, D. Roy, and S. Aral, “The spread of true and false news online,” Science, vol. 359, no. 6380, pp. 1146–1151, 2018.
and other references as listed in the accompanying paper.
