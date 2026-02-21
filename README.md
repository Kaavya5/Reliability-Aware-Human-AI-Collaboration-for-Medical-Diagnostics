
🔍 Overview

This project presents a Reliability-Aware Human-AI Collaborative Framework for medical diagnostics. Unlike traditional AI systems that directly map predictions to decisions, this system introduces a risk-based workflow that incorporates uncertainty and clinical severity to guide human decision-making.

🚨 Problem

Conventional medical AI systems:

Rely heavily on prediction probability

Ignore uncertainty and clinical consequences

Encourage automation bias

Fail to ensure decision safety

💡 Solution

This project proposes a Risk-Tiered Human-in-the-Loop (HITL) framework that:

Computes risk using probability, uncertainty, and severity

Adjusts decisions based on prediction reliability

Assigns priority levels (Routine / Review / Urgent)

Ensures clinician involvement in critical cases

⚙️ Key Features

CNN-based medical image classification (Pneumonia detection)

Monte Carlo Dropout for uncertainty estimation

Reliability-aware risk scoring

Human-in-the-loop decision support

Interactive Streamlit dashboard

Real-time comparison:

Traditional AI vs Collaborative AI

📊 Key Results

Missed critical cases reduced by ~23%

Unnecessary escalations reduced by ~18%

Review efficiency improved by ~27%

No change in baseline accuracy (~91%)

🎯 Key Insight

This project demonstrates that workflow design, not just model accuracy, determines clinical safety.

🧰 Tech Stack

Python

PyTorch

Streamlit

NumPy, OpenCV, Matplotlib

📁 Dataset

Chest X-ray Pneumonia dataset (Kaggle)

Used for rapid prototyping and demonstration

🚀 How to Run
pip install -r requirements.txt
streamlit run app.py

⚠️ Note

This is a research-oriented prototype intended to demonstrate Human-AI collaboration, not a clinical diagnostic tool.