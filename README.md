# Student Performance Prediction – MLOps Practice Project
📌 Project Overview

This project is a small, practice-focused Machine Learning project created to understand and implement core MLOps concepts such as structured code, data ingestion, data transformation, model training pipelines, logging, and exception handling.
The goal of this project is learning and experimentation, not deployment or production usage.

# Objective

To predict student performance using supervised machine learning while practicing:

Modular ML project structure

Pipeline-based workflow

Reusable components

Logging and custom exception handling

Project Structure:
The project is organized in a modular MLOps-style layout where the data/ directory stores raw and processed datasets, catboost_info/ contains CatBoost training artifacts, and the components/ folder includes core ML components such as data_ingestion.py for loading data, data_transformation.py for preprocessing and feature engineering, and model_trainer.py for model training and tuning. The pipeline/ directory holds the end-to-end training_pipeline.py that connects all stages together, while utils.py provides reusable helper functions. Centralized logging and error handling are implemented using logger.py and exception.py, and setup.py enables the project to be used as a package. Exploratory Data Analysis and problem understanding are documented in the EDA and Problem Statement notebook, and final model experimentation and training steps are demonstrated in the MODEL TRAINING.ipynb notebook.

🔧 Key Features

Data Ingestion: Reads raw data and prepares it for processing

Data Transformation: Feature engineering and preprocessing pipeline

Model Training: ML model training with hyperparameter tuning

Pipeline Design: Clean separation of ingestion, transformation, and training

Logging: Centralized logging for tracking execution flow

Exception Handling: Custom exception class for better debugging

📊 EDA

Exploratory Data Analysis is performed in Jupyter Notebook to:

Understand feature distributions

Identify correlations

Define the problem statement clearly

⚙️ Technologies Used

Python

Pandas, NumPy

Scikit-Learn

CatBoost

Jupyter Notebook

🚫 What This Project Does NOT Include

No model deployment

No CI/CD pipeline

No Docker or cloud integration

These are intentionally excluded as this project is focused on core MLOps concepts, not production deployment.

🎓 Learning Outcome

Through this project, I gained hands-on experience with:

Structuring ML projects professionally

Writing clean, reusable ML pipelines

Handling errors and logs in ML workflows

Understanding how MLOps fits into real-world ML systems

