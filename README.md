# CAPSTONE-PROJECT-24MBMA74-JASNA S

# Data Analytics and Machine Learning Applications in the BFSI Sector Using Databricks

# Overview

This capstone project focuses on applying Apache Spark and Databricks to solve critical challenges in the Banking, Financial Services, and Insurance (BFSI) sector through data analytics and machine learning.

The project covers five real-world BFSI use cases that collectively demonstrate how large-scale financial data can be used to derive business insights, predict risks, and detect anomalies in real time.

Each use case targets a specific functional area within BFSI — from customer segmentation and loan risk to fraud detection and branch performance — providing a holistic view of how data-driven intelligence can transform financial operations.

# Project Objectives

To leverage big data analytics for solving business challenges in BFSI.

To build and evaluate machine learning models for prediction and anomaly detection.

To apply real-time analytics for monitoring financial transactions and performance.

To design a scalable analytical workflow using Apache Spark on Databricks.

# Use Case Overview
1. Customer Transaction Analysis

Goal: Understand customer spending patterns, transaction behavior, and channel performance.

Approach: Used Spark SQL and DataFrames to analyze total spend, channel share, and customer segmentation.

Key Insights:

Digital channels contributed to over 65% of total spend.

Peak transactions occurred during evening hours.

Helped design targeted marketing and loyalty strategies.

2. Loan Default Prediction

Goal: Predict the probability of loan default based on customer attributes and repayment history.

Approach: Implemented Logistic Regression using Spark MLlib after cleaning and feature engineering.

Key Insights:

Achieved ~85% model accuracy.

Credit history and income-to-loan ratio were the most influential predictors.

Enabled proactive risk-based lending and better credit management.

3. Insurance Claim Analysis

Goal: Identify fraudulent or suspicious insurance claims through data analytics.

Approach: Performed exploratory analysis and statistical filtering using Spark DataFrames.

Key Insights:

Detected anomalies in claim-to-premium ratios and duplicate submissions.

Helped reduce manual verification time by ~35%.

Strengthened the insurer’s fraud prevention framework.

4. Real-Time Credit Card Fraud Detection

Goal: Detect fraudulent transactions as they occur in real time.

Approach: Used Spark Structured Streaming to process continuous transaction data streams.

Key Insights:

Detected high-frequency, high-value, and geo-inconsistent transactions.

Achieved detection latency of under 3 seconds.

Enhanced fraud response speed and customer trust.

5. Branch Performance Evaluation

Goal: Assess and compare branch-level performance using profitability and customer satisfaction KPIs.

Approach: Applied Z-score normalization to categorize branches.

Key Insights:

Classified branches as High-performing, Average, or Underperforming.

Highlighted key areas for resource optimization and improvement.

Supported data-driven strategic planning at branch level.

# Methodology

Data Ingestion: Loaded structured BFSI datasets into Databricks.

Data Preprocessing: Cleaned data, handled nulls, encoded categorical variables.

Feature Engineering: Derived KPIs and behavior metrics.

Model Development: Built classification and anomaly detection models using MLlib.

Evaluation & Visualization: Used Spark SQL and Matplotlib for insights and plots.

# Key Results

Improved fraud detection accuracy and reduced response time.

Enhanced credit risk assessment and loan portfolio management.

Enabled real-time analytics for transaction monitoring.

Delivered branch-level performance insights for decision-making.

Showcased the scalability of Databricks + Spark for large BFSI datasets.

# Technology Stack

Platform - Databricks Community Edition
Framework - Apache Spark 3.5
Language - PySpark, Python
Libraries -	Spark SQL, MLlib, Pandas, NumPy, Matplotlib
Data Types	- Structured and Semi-structured BFSI data

# Conclusion

This project demonstrates how big data analytics and machine learning can transform BFSI operations by providing real-time, predictive, and data-driven intelligence.
By using Databricks and Apache Spark, complex financial data can be processed efficiently to improve fraud detection, risk assessment, and strategic decision-making.
