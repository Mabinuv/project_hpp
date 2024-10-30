# House Price Prediction 

This is a simple ML project for house price prediction. The project is scoped to take a public available dataset and 
apply machine learning techniques and predict the sale price of unknown houses 

More than the approach of simply applying machinelearning on random dataset, the whole scope
of work is 

* Understand and study Ml techniques and concepts as we progress in this project
* To follow a structural approach in implementing a project

The machine learning models used in this project is:
* Random Forest
* XGBoost
* GradientBoost
* Support vector regressor

### Directory Structure

```plaintext
HPP/
├── data/
│   ├── raw/                # Raw data files
│   └── processed/          # Processed data files
├── notebooks/
│   ├── EDA.ipynb
│   └── Modeling.ipynb
├── models/
│   └── best_model.joblib
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── evaluate_model.py
├── tests/
│   └── test_models.py
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

## Requirements

Make sure to install the necessary dependencies before running the script:

```bash
python preprocess.py /Project/HPP/data/train.csv  --test_size 0.3
```

