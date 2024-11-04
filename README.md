# Machine Learning Prediction Web Application

##  Project Overview

This project is a dynamic Flask web application designed to provide machine learning predictions for predictive maintenance.


![Example GIF](https://github.com/omaar25/Predictive-Maintenance-Web-Application/raw/main/app.gif)


## Project Motivation
Predictive maintenance offers transformative benefits in reducing operational costs, enhancing uptime, and improving equipment reliability. This application aims to empower end-users with actionable insights, enabling a shift from reactive to proactive maintenance. 

## Success Criteria
To gauge project success, we will assess the following:
1. **Model Performance:** Evaluated based on precision, recall, and F1 score, ensuring accurate and relevant failure predictions.
2. **User Experience:** The web application is designed for ease of use, with a focus on responsiveness and intuitive navigation.
3. **Operational Impact:** Demonstrated reduction in unplanned downtime and maintenance expenses through timely maintenance interventions.

## Key Features
- **Machine Learning Pipeline**: Implements a robust and scalable pipeline that includes data ingestion, data preprocessing, feature engineering, and model inference to deliver accurate predictions.
- **User-Friendly Interface**: An intuitive web interface that allows users to input data and receive instant predictions, enhancing user experience and engagement.
- **Containerization with Docker**: The entire application is containerized, ensuring consistency across development and production environments and simplifying deployment.
- **Azure Integration**: Continuous Integration/Continuous Deployment setup with GitHub Actions to automate testing and deployment to Azure Container Registry.

## Technologies Used

- **Programming Language**: Python, CSS, JavaScript, HTML
- **Web Framework**: Flask
- **Containerization**: Docker
- **Cloud Services**: Azure, Azure Container Registry
- **CI/CD**: GitHub Actions
- **Machine Learning**: Custom pipeline built with relevant ML libraries (e.g., scikit-learn, pandas)

##  Getting Started

### Prerequisites

Before running the application, ensure you have the following:

- Docker installed on your machine.
- An Azure account with access to Azure Container Registry.
- Python 3.x for local development.

### Installation Steps

1. **Clone the repository**:
```
git clone https://github.com/omaar25/Predictive-Maintenance-Web-Application.git
cd Predictive-Maintenance-Web-Application
```

2. **Build the Docker image**:

```
docker build -t ml-predictor-app .
```

3. **Run the Docker container**:

```
docker run -d -p 8080:8080 ml-predictor-app
```
4. **Access the application**:
Open your browser and navigate to 
```
http://localhost:8080
```

## How It Works
1. Data Input: Users can input data via the web interface.
2. Prediction Display: When data is submitted, the application processes the input through the custom pipeline to display prediction instantly.

## Continuous Integration and Deployment
Automated workflows using GitHub Actions ensure that every push to the main branch triggers a build and deployment pipeline, enhancing code quality and deployment efficiency.
The application is pulled from Azure Container Registry and runs seamlessly in any environment.
