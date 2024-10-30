from model import *
from func import *
import pandas as pd
import os
import joblib
import argparse


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Training models and saving the best model fro inference')
    parser.add_argument("Train_file",type=str, help="The train data ")
    parser.add_argument("Test_file",type=str, help="The test file")
    args = parser.parse_args()

    df_train = pd.read_csv(args.Train_file)
    df_test = pd.read_csv(args.Test_file)

    X_train = df_train.drop('SalePrice', axis=1)
    y_train = df_train['SalePrice']

    X_test = df_test.drop('SalePrice', axis=1)
    y_test = df_test['SalePrice']

    rf = RandomForest()
    rf.train(X_train, y_train)
    y_pred = rf.predict(X_test)
    rf_mse, rf_mae, rf_r2, rf_rmse = rf.evaluate(y_test, y_pred)

    xg_model = XGBoost()
    xg_model.train(X_train, y_train)
    y_pred_xgb = xg_model.predict(X_test)
    xg_mse, xg_mae, xg_r2, xg_rmse = xg_model.evaluate(y_test, y_pred_xgb)

    GB_model = GradientBoosting()
    GB_model.train(X_train, y_train)
    y_pred_gb = GB_model.predict(X_test)
    gb_mse, gb_mae, gb_r2, gb_rmse = GB_model.evaluate(y_test, y_pred_gb)

    svm_model = SVM()
    svm_model.train(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    svm_mse, svm_mae, svm_r2, svm_rmse = svm_model.evaluate(y_test, y_pred_svm)

    ## Saving best model
    save_folder = "/Users/mabin/PycharmProjects/Project/HPP/model"
    os.makedirs(save_folder, exist_ok=True)
    model_scores = {"RF" : rf_rmse,
             "XGB" : xg_rmse,
             "GB" : gb_rmse,
             "SVM" : svm_rmse}

    # Find the model with the lowest RMSE
    best_model_name = min(model_scores, key=model_scores.get)
    best_rmse = model_scores[best_model_name]
    print(f"Best model: {best_model_name} with RMSE: {best_rmse}")

    model_filename = os.path.join(save_folder, f"best_{best_model_name}_model.joblib")
    joblib.dump(xg_model, model_filename)
    print(f"Best model ({best_model_name}) saved as '{model_filename}' with RMSE: {best_rmse}")

# Plot the error metrics for all for models
plot_metrics(rf_mse, xg_mse, gb_mse, svm_mse,
             rf_mae, xg_mae, gb_mae, svm_mae,
             rf_r2, xg_r2, gb_r2, svm_r2,
             rf_rmse, xg_rmse, gb_rmse, svm_rmse)