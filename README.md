# 🏥 Reliability-Aware Human–AI Collaboration for Safe Medical Diagnostics

## 📌 Overview
This project presents a **Reliability-Aware Human–AI Collaborative Framework** for medical diagnostics.  
Unlike traditional AI systems that directly map predictions to decisions, this system introduces a **risk-based workflow** that integrates:

- Prediction probability  
- Model uncertainty  
- Clinical severity  

to guide **safe and informed human decision-making**.

---

## ⚠️ Problem Statement
Conventional medical AI systems:

- Rely heavily on prediction probability  
- Ignore uncertainty and clinical consequences  
- Encourage automation bias  
- Fail to ensure decision safety  

---

## 💡 Proposed Solution
We propose a **Risk-Tiered Human-in-the-Loop (HITL) framework** that:

- Computes risk using **probability, uncertainty, and severity**
- Adjusts decisions based on **prediction reliability**
- Assigns priority levels:
  - 🟢 Routine  
  - 🟡 Review  
  - 🔴 Urgent  
- Ensures **clinician involvement in critical cases**

---

## ⚙️ Key Features

- 🧠 CNN-based medical image classification (Pneumonia Detection)
- 📊 Monte Carlo Dropout for uncertainty estimation
- 📈 Reliability-aware risk scoring mechanism
- 👨‍⚕️ Human-in-the-loop decision support system
- 📊 Interactive Streamlit dashboard
- 🔄 Real-time comparison:
  **Traditional AI vs Collaborative AI**

---

## 📊 Key Results

- ✅ **Accuracy:** 95%
- 🔻 **Missed critical cases:** ↓ 73.5%
- 🚫 **False urgent alerts:** ↓ 100%
- ⚡ **Review efficiency:** ↑ 27.1%

> ⚠️ Note: Improvements achieved **without increasing model accuracy**, highlighting the importance of workflow design.

---

## 🎯 Key Insight
> **Clinical safety is determined by workflow design, not just model accuracy.**

---

## 🛠️ Tech Stack

- Python  
- PyTorch  
- Streamlit  
- NumPy  
- OpenCV  
- Matplotlib  

---

## 📂 Dataset

- Chest X-ray Pneumonia Dataset (Kaggle)  
- Used for rapid prototyping and demonstration  

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```
---
## 📊 System Workflow
Image → Model Prediction → Uncertainty Estimation → Risk Scoring → Priority Assignment → Human Decision

---
## ⚠️ Disclaimer

This is a research-oriented prototype designed to demonstrate
Human–AI collaboration, not a clinical diagnostic tool.
---

## ⭐ Why This Project Matters

Moves AI from decision-maker → decision-support system

Reduces critical diagnostic failures

Prevents automation bias

Improves clinical safety without changing model accuracy

---
