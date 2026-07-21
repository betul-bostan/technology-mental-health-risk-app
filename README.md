# Technology and Mental Health Risk App

An end-to-end data science and machine learning project that examines the relationship between technology usage, lifestyle habits, social interaction, and mental health.

The repository includes data preparation, exploratory data analysis, feature engineering, classification model development, and an interactive Streamlit application.

> **Disclaimer:** This project was developed for educational and portfolio purposes. It is not a medical diagnosis tool and must not replace professional mental health advice.

## Project Overview

The project combines multiple datasets related to:

- Technology and screen usage
- Social media habits
- Sleep and physical activity
- Work and study hours
- Caffeine consumption
- Positive and negative online interactions
- Social support
- Mental health indicators

The processed data is used to train a machine learning pipeline that estimates whether the entered lifestyle profile belongs to a higher-risk group.

## Main Features

- Integration of multiple mental-health and technology datasets
- Data cleaning and preprocessing
- Missing-value handling
- Exploratory data analysis
- Feature engineering
- HistGradientBoosting classification pipeline
- Saved model for reusable inference
- Interactive Streamlit interface
- Personalized lifestyle recommendations
- Turkish-language application interface

## Application Inputs

The Streamlit application collects information such as:

- Age
- Gender
- Social support
- Caffeine consumption
- Main social media platform
- Total screen time
- Social media usage
- Gaming time
- Work or study hours
- Sleep duration
- Physical activity
- Positive digital interactions
- Negative digital interactions

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib
- Matplotlib
- Seaborn
- Jupyter Notebook

## Repository Structure

```text
technology-mental-health-risk-app/
├── models/
│   └── hgb_pipeline.joblib
├── 01_data_preparation_and_eda.ipynb
├── 02_model_training_and_evaluation.ipynb
├── app.py
├── model_kontrol.py
├── transforms.py
├── Merged_Optimized_Mental_Health_20k_NO_ROW_DROP.csv
├── Tech_Use_Stress_Wellness.csv
├── mental_health_and_technology_usage_2024.csv
├── mental_health_social_media_dataset.csv
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/betul-bostan/technology-mental-health-risk-app.git
cd technology-mental-health-risk-app
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

```bash
streamlit run app.py
```

The application will open in your browser.

## Machine Learning Workflow

1. Load multiple technology and mental health datasets
2. Standardize column names and data formats
3. Handle missing and zero values
4. Merge related datasets
5. Perform exploratory data analysis
6. Create model-ready features
7. Train and evaluate classification models
8. Save the selected HistGradientBoosting pipeline
9. Load the saved pipeline in the Streamlit application
10. Present the prediction with lifestyle recommendations

## Prediction Output

The application presents:

- A model-based probability score
- A higher-risk or healthier-profile classification
- Personalized recommendations based on entered habits

Some display-level rules are used in addition to the raw model probability to make extreme lifestyle values more visible. These rules are application-level adjustments and are not clinical thresholds.

## Data Files

The repository currently includes several source and processed CSV files.

Future versions should:

- Document the original source of each dataset
- Specify dataset licenses
- Move large raw files to an external data source where appropriate
- Keep only a small sample dataset in the repository

## Future Improvements

- Add verified model-performance metrics
- Add confusion matrix and ROC-AUC results
- Add application screenshots
- Deploy the Streamlit application
- Add automated tests
- Add model explainability
- Refactor reusable preprocessing code
- Replace display-level rules with a formally validated calibration method
- Add dataset sources and licenses

## Author

**Betül Bostan**

- [GitHub](https://github.com/betul-bostan)
- [LinkedIn](https://www.linkedin.com/in/bet%C3%BCl-bostan-2105942b2/)
