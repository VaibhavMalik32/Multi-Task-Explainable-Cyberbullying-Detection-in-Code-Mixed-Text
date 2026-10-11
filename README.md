# Explainable Multi-Task Deep Learning for Code-Mixed Hinglish Cyberbullying Detection

**Course:** Deep Neural Networks (AIML203)  
**Author:** Vaibhav Arun Malik

## 📌 Project Overview

Cyberbullying detection in code-mixed languages such as Hinglish (Hindi + English) is challenging due to irregular spelling, regional slang, transliteration, and complex linguistic structures. Conventional classification models often behave like black boxes, providing predictions without explaining which parts of a sentence influenced their decisions.

This project presents an end-to-end **Explainable AI (XAI) pipeline** based on a **Multi-Task Learning (MTL) Transformer architecture**. Beyond detecting cyberbullying, the system identifies toxic subwords that contribute to its predictions and classifies the intended target category.

## 🧠 Model Architecture

The system uses a shared Transformer encoder with three task-specific prediction heads.

- **Base Model:** `l3cube-pune/hing-bert` — a pretrained language model designed for Hinglish.
- **Task 1 — Cyberbullying Detection:** Binary classification into Bully and Non-Bully.
- **Task 2 — Target Classification:** Multi-class classification into Religion, Gender, Personal, Profession, and Other.
- **Task 3 — Rationale Extraction:** Token-level identification of text segments associated with cyberbullying.

The shared encoder enables the tasks to learn complementary linguistic representations while the separate heads address their individual prediction objectives.

## 📊 Performance and Scientific Validation

The model was evaluated using **Stratified 5-Fold Cross-Validation**, dynamic class weighting with Scikit-learn, and early stopping to address class imbalance and reduce overfitting.

### Final Champion Model Metrics

| Task | Reported Performance |
|---|---:|
| Cyberbullying Detection | 86.04% |
| Target Classification | 65.20% |
| Rationale Extraction (XAI) | 82.92% |

*Note: The metric names, averaging methods, and evaluation protocol should be specified in the accompanying report to make these results reproducible and interpretable.*

## 🛠️ Key Engineering Features

### 1. Label Consolidation
Consolidated 11 sparse dataset labels into five broader target categories to improve label consistency and provide a more manageable classification task.

### 2. Dynamic Subword Reassembly
Implemented custom logic to reconstruct fragmented Transformer subwords into readable Hinglish text during inference. For example, token fragments such as `kut`, `##ti`, and `##ya` can be reassembled into their corresponding word.

### 3. Spurious Correlation Mitigation
Implemented a Hinglish stopword filter to suppress grammatical noise, including words such as `tu` and `hai`, when presenting rationale highlights. This helps make the displayed explanations more focused on potentially toxic expressions.

### 4. Multi-Task Learning
Combines sentence-level bullying detection, target classification, and token-level rationale extraction in a shared-encoder architecture.

### 5. Cross-Validation and Early Stopping
Uses a five-fold training procedure and early stopping to support model selection and reduce the risk of overfitting.

## 🚀 How to Run the Project

### Prerequisites

- Python installed on your system
- pip package manager
- Sufficient memory and storage for the model and dataset

### 1. Install Dependencies

Clone the repository and navigate into the project directory if you have not already done so. Then install the required packages:

```bash
pip install -r requirements.txt
```

### 2. Data Preprocessing

Clean the raw BullyExplain dataset, map the target labels into the five superclasses, and prepare the rationale annotations.

```bash
python preprocess.py
```

Ensure that the input dataset is available at the path expected by the preprocessing script.

### 3. Train the Model

Run the training pipeline, which performs five-fold cross-validation with early stopping and saves the selected champion model weights.

```bash
python train.py
```

**Important:** Training may require substantial memory, compute time, and disk space. The exact requirements depend on the dataset, batch size, and hardware.

### 4. Run Live Inference

Test the trained model on a Hinglish sentence:

```bash
python predict.py
```

Enter a comment when prompted to view the predicted bullying status, target category, and highlighted rationale.

## 🔍 Sample Explainable AI Output

**Input:**

```text
tu gadha hai kya
```

**Illustrative output:**

```text
=============================================
PREDICTION RESULTS
=============================================
Text:          tu gadha hai kya
Bully Status:  Bully
Target Class:  Religion & Community
Rationale:     Identified toxic words -> [gadha]
=============================================
```

*The output above illustrates the intended inference format. Actual predictions depend on the trained model and input text.*

## 📁 Project Structure

```text
.
├── data/
│   ├── clean_data.csv
│   └── BullyExplain.xlsx
├── saved_model/
│   ├── model.pt
│   └── target_classes.npy
├── model.py
├── preprocess.py
├── train.py
├── predict.py
├── utils.py
├── requirements.txt
├── training_plots.png
└── README.md
```

The exact files may vary depending on the current state of the repository. The dataset, model checkpoint, and generated plots should be present only if they are included in the project distribution.

## 🎯 Applications

- Cyberbullying screening for code-mixed social media text
- Explainable text classification for low-resource language settings
- Research into multi-task learning for harmful-content detection
- Analysis of linguistic rationales behind model predictions

## ⚠️ Limitations and Responsible Use

- Code-mixed Hinglish contains regional variations, sarcasm, context-dependent expressions, and evolving slang that may affect model performance.
- A highlighted rationale is a model-generated explanation, not definitive proof of harmful intent.
- Target classification and bullying predictions may be incorrect, particularly for ambiguous or context-dependent statements.
- The model should support human review rather than serve as the sole basis for moderation or punitive decisions.

## 🔬 Reproducibility

For reproducible evaluation, document the dataset version, train-validation-test protocol, random seeds, metric definitions, and model hyperparameters. If reporting five-fold cross-validation results, clarify whether the displayed scores are fold averages, held-out test results, or results from a separately selected champion model.

## 👨‍💻 Author

**Vaibhav Arun Malik**  
B.Tech — Computer Science and Engineering  
Deep Neural Networks (AIML203)
