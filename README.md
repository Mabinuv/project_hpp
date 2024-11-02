
# House Price Prediction 
####  A simple learning project

### Description
This is a simple ML project for house price prediction. The project is scoped to take a public available dataset of house prices and 
apply machine learning techniques and predict the sale price of unknown houses 

More than the approach of simply applying machinelearning on random dataset, the whole scope
of work is:-

* Understand and study Ml techniques and concepts as we progress in this project
* To follow a structural approach in implementing a project

The machine learning models used in this project is:
* Random Forest
* XGBoost
* GradientBoost
* Support vector regressor

### Dataset
This is public available dataset obtained from [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

### Requirements
```bash
$ pip install -r requirements.txt
```


### Directory Structure
```plaintext
HPP/
├── data/
│   ├── raw/                # Raw data files
│   └── processed/          # Processed data files
├── notebooks/
│   ├── EDA.ipynb           #Done
│   └── Modeling.ipynb
├── models/
│   └── best_model.joblib
├── src/
│   ├── func.py
│   ├── preprocess.py           # Done
│   ├── train_model.py
│   └── model.py
│ 
├── tests/
│   └── test_models.py
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

### Outline
Set up the python environment and run the requirement.txt to setup necessary
dependencies

The train.csv and test.csv files are avaliable int he \data folder. These 
files are sourced from kaggle. Run the preprocess.py file to do necessary 
data preprocessing
  * Handling Null values
  * Handling numerical and categorical values
  * Scaling and splitting train and test datasets
  * Function returns saving the processed train_df.csv and 
  test_df.csv files for evaluation.
```bash
python preprocess.py /Project/HPP/data/train.csv  --test_size 0.3
```
Train the preprocessed datasets on different models. The script should train
four different model and save the best model based on the root mean squared error(rmse).


```bash
python train.py /Project/HPP/data/train_df.csv  /Project/HPP/data/test_df.csv
```
Results
  * Models trained on different hyperparameters like No_of_estimaters, learning_rate, epsilon
    (which can be further explored)
  * XGBoost model performed better when compared with other models
 
The XGBoost model shows high performing model for the house price prediction 
task, Which indicates that it provides the more reliable 
predictions when compared to other models. Random forest and gradient boost also 
performs well and could be considered a strong alternative, 
but it falls slightly behind 
XGBoost, while SVR underperforms significantly and is 
not suitable for this task.

## Learning concepts

### Models
- Random Forest
    * Random forest is an ensemble learning model.
    * Is built on the concept of decision trees -> these itself are individual models that split data(features) into branches
    * RF has multiple decision trees, trained on random subsets
    * The results of these DT are then combined (Averaged for regression task)
    * concepts: Bootstrap sampling
    * Benefits:-
      * improved accuracy - because trained on multiple random trees -> reduces overfitting
      * manage high dimensional data
    * Downside
      * less interpretability -> complex in nature
      * Computationally intensive

- Gradient Boost
  * It is ensemble learning model. [Level-wise growth]
  * Unlike RF model, where each tree is trained independently, gradient boosting 
  focuses on minimizing the overall error by learning from mistakes 
  in a step-by-step manner.
    * Start with simple Decision trees
    * Make the normal predictions and calculate the RESIDUAL(errors)
    * A new tree is then trained to predict these residuals,
    focusing on the parts of the data where the previous model performed poorly. 
    So basically, the new model tries to minimize the error of the prior model 
    by “boosting” its predictions learned from its prior model
    * Update the model => new model's prediction + prev model's prediction adopting a learning rate
    to generalise better
    * This iterative process is repeated until specified number of trees added
  * In GB, the features are split heuristically, which makes is inefficient handling 
  large datasets.

- XGBoost - Extreme gradient boosting
  * Works under the same principal as gradient boost, adding models sequentially 
  to correct error. [Depth-wise growth]
  * optimized and enhanced version of gradient boost
  * Includes regularization techniques L1(Lasso) and L2(Ridge) to prevent
  overfitting when hanlding large and complex datasets
  * Uses "weighted quantile sketch algorithm" to find optimal splits quickly
  * Supports early stopping mechanism built into the model

- Support vector regressor - Margin-based model
  * SVR is based on support vector machines (SVM), which aim to find an 
  optimal hyperplane that best fits the data within a margin(Controlled by the parameter epsilon).
  It focuses on minimizing error while allowing a certain tolerance around predictions. 
  * The margin factor is based around the EPSILON parameter. i.e. points within the margin ARE NOT penalized,
  while the ones outside are considered while minimizing the error.
  * It handles kernel functions like polynomial or RBF to handle non-linearity
  * it is computationally expensive when compared with XGBoost


  ![](/Users/mabin/PycharmProjects/Project/HPP/model/Evaluation Plot.png "Evaluation plots of four different models")