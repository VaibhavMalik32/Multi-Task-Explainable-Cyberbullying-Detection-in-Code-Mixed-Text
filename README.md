# Explainable Multi-Task Deep Learning for Code-Mixed Hinglish Cyberbullying Detection

**Course:** Deep Neural Networks (AIML203)  
**Author:** Vaibhav Arun Malik  

## 📌 Project Overview
Detecting cyberbullying in code-mixed languages like "Hinglish" (Hindi + English) is fundamentally challenging due to irregular spelling, regional slang, and complex syntax. Standard models act as "black boxes," failing to explain their predictions. 

This project presents an end-to-end **Explainable AI (XAI)** pipeline using a **Multi-Task Learning (MTL)** Transformer architecture. It doesn't just flag toxicity; it mathematically highlights the exact toxic subwords (the rationale) used in the attack.

## 🧠 Model Architecture
The system is built on a Shared-Encoder Multi-Task Architecture utilizing a domain-specific Transformer:
* **Base Model:** `l3cube-pune/hing-bert` (Pre-trained on Indian social media).
* **Task 1 (Sentence Level):** Binary Classification (Bully vs. Non-Bully).
* **Task 2 (Sentence Level):** Multi-Class Target Classification (Religion, Gender, Personal, Profession, Other).
* **Task 3 (Token Level - XAI):** Rationale Extraction (Highlighting specific toxic subwords).

## 📊 Performance & Scientific Validation
To ensure maximum reliability and combat data imbalance, the model was trained using **Stratified 5-Fold Cross-Validation**, dynamic Scikit-Learn class weights, and Early Stopping. 

**Final Champion Model Metrics:**
| Metric | Accuracy |
| :--- | :--- |
| **Bullying Detection** | 86.04% |
| **Target Demographic** | 65.20% |
| **Rationale Extraction (XAI)** | 82.92% |

## 🛠️ Key Engineering Features
* **Dimensionality Reduction:** Consolidated 11 sparse dataset labels into 5 robust Super Classes to stabilize the loss function.
* **Dynamic Subword Reassembly:** Engineered custom logic to stitch fragmented BERT subwords (e.g., `kut`, `##ti`, `##ya`) back into readable Hinglish text during inference.
* **Spurious Correlation Mitigation:** Implemented a Hinglish Stopword Filter (ignoring grammatical noise like "tu", "hai") to ensure only actual toxic words are highlighted.

## 🚀 How to Run the Project

**1. Install Dependencies**
```bash
pip install -r requirements.txt

2. Data Preprocessing
Cleans the raw BullyExplain dataset, maps the Super Classes, and extracts rationale strings.
Bash
python preprocess.py

3. Train the Model
Runs the 5-Fold Cross-Validation loop and automatically saves the absolute best-performing champion model weights to saved_model/model.pt.
Bash
python train.py

4. Live Inference (Prediction)
Test the model on live Hinglish text with active stopword filtering and dynamic subword reconstruction.
Bash
python predict.py

🔍 Sample XAI Output
Plaintext
Enter Hinglish comment: tu gadha hai kya

=============================================
🧐 PREDICTION RESULTS
=============================================
📝 Text:          tu gadha hai kya
🚨 Bully Status:  Bully
🎯 Target Class:  Religion & Community
🔍 Rationale:     Identified toxic words -> [gadha]
=============================================

📁 Project Structure
Plaintext
├── data/                   # Contains raw and clean_data.csv
├── saved_model/            # Stores the champion model.pt and target_classes.npy
├── model.py                # The PyTorch Multi-Task Neural Network class
├── preprocess.py           # Data cleaning and class reduction logic
├── train.py                # 5-Fold training loop with Early Stopping
├── predict.py              # Live inference and XAI token-reassembly
├── utils.py                # Text cleaning helper functions
├── requirements.txt        # Python dependencies
└── final_training_plots.png # Graph of Loss/Accuracy trends