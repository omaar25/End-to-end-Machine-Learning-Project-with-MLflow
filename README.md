# Predictive Maintenance Web Application

## Overview
This project develops a web application for predictive maintenance using synthetic data, employing machine learning models to predict equipment failure and diagnose potential causes. The application provides real-time analytics to enhance proactive maintenance strategies and reduce unexpected equipment downtime.

## Features
- **Failure Prediction:** Predicts the likelihood of equipment failure based on factors such as temperature, rotational speed, torque, and tool wear.
- **Failure Type Classification:** Classifies potential failure types when failure risk is detected, assisting in targeted maintenance response.

## Project Motivation
Predictive maintenance offers transformative benefits in reducing operational costs, enhancing uptime, and improving equipment reliability. This application aims to empower end-users with actionable insights, enabling a shift from reactive to proactive maintenance. 

## Success Criteria
To gauge project success, we will assess the following:
1. **Model Performance:** Evaluated based on precision, recall, and F1 score, ensuring accurate and relevant failure predictions.
2. **User Experience:** The web application is designed for ease of use, with a focus on responsiveness and intuitive navigation.
3. **Operational Impact:** Demonstrated reduction in unplanned downtime and maintenance expenses through timely maintenance interventions.

## System Architecture
The application consists of three main layers:
1. **Data Ingestion and Processing**: Synthetic dataset pre-processing and feature engineering are automated to streamline model input.
2. **Machine Learning Models**: A suite of machine learning models, including Logistic Regression, SVM, Decision Trees, and Random Forests, were developed with configuration managed via YAML files for easy parameter adjustments.
3. **Web Application Interface**: The user interface allows real-time input of process parameters, displaying risk assessments and maintenance recommendations in an interactive format.
