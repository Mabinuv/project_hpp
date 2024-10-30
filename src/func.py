import matplotlib.pyplot as plt


def plot_metrics(rf_mse, xg_mse, gb_mse, svm_mse, rf_mae, xg_mae, gb_mae, svm_mae, rf_r2, xg_r2, gb_r2, svm_r2, rf_rmse, xg_rmse, gb_rmse, svm_rmse):
    models = ['RF', 'XGB', 'GB', 'SVM']
    mse_scores = [rf_mse, xg_mse, gb_mse, svm_mse]
    mae_scores = [rf_mae, xg_mae, gb_mae, svm_mae]
    r2_scores = [rf_r2, xg_r2, gb_r2, svm_r2]
    rmse_scores = [rf_rmse, xg_rmse, gb_rmse, svm_rmse]

    fig, axes = plt.subplots(1, 4, figsize=(15, 5), sharey=True)  # Three subplots side-by-side

    # MSE Plot
    axes[0].barh(models, mse_scores, color=['blue', 'orange', 'green', 'red'], alpha=0.9)
    axes[0].set_title('MSE')
    axes[0].set_xlabel('Mean Squared Error')
    axes[0].set_ylabel('Models')

    # MAE Plot
    axes[1].barh(models, mae_scores, color=['blue', 'orange', 'green', 'red'], alpha=0.9)
    axes[1].set_title('MAE')
    axes[1].set_xlabel('Mean Absolute Error')

    # R² Plot
    axes[2].barh(models, r2_scores, color=['blue', 'orange', 'green', 'red'], alpha=0.9)
    axes[2].set_title('R² Score')
    axes[2].set_xlabel('R²')

    # RMSE Plot
    axes[3].barh(models, rmse_scores, color=['blue', 'orange', 'green', 'red'], alpha=0.9)
    axes[3].set_title('RMSE Score')
    axes[3].set_xlabel('RMSE')

    plt.tight_layout()
    plt.show()


