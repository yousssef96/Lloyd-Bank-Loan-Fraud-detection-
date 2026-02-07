# 🏦 Lloyd Bank Loan Assessment Portal
### *End-to-End Machine Learning Service for Credit Risk Prediction*

[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Deployment-Docker-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/Model_Tracking-MLflow-0194E2?style=flat&logo=mlflow&logoColor=white)](https://mlflow.org/)

## 📌 Project Overview
This project provides a professional-grade solution for assessing loan application risks. It transitions a standard machine learning model from a Jupyter Notebook into a **decoupled production architecture**. 

By separating the **FastAPI Backend** (the "Brain") from the **Streamlit Frontend** (the "Face"), the system mimics how real-world financial institutions like Lloyd Bank deploy scalable, secure internal tools for credit officers.



## 🛠️ Tech Stack
* **Modeling:** Python, Scikit-Learn, Pandas, XGBoost.
* **Backend API:** FastAPI (Asynchronous, Pydantic data validation).
* **Frontend UI:** Streamlit (Interactive dashboard).
* **Model Management:** MLflow (Experiment tracking & versioning).
* **Data Quality (Great Expectations):** Implemented automated data validation to ensure incoming features (ranges, types, and nulls) match professional banking standards before prediction.
* **DevOps:** Docker (Containerization), Render & Streamlit Cloud (Deployment).

## 🚀 Key Features
* **Real-time Inference:** Input 29 distinct credit features and get an instant risk assessment.
* **Data Validation:** Uses Pydantic schemas to ensure incoming API data is mathematically sound before hitting the model.
* **Decoupled Architecture:** The API can be used by the Streamlit UI, mobile apps, or other bank services simultaneously.
* **Containerized Environment:** Fully Dockerized for "works on my machine" consistency across development and production.
