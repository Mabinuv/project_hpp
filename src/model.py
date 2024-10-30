from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor


class BaseModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def process_data(self, X, y, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        print(f"{self.__class__.__name__} Model Trained")

    def predict(self, X_test):
        prediction = self.model.predict(X_test)
        return prediction

    def evaluate(self, y_test, y_pred):
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        # print(f"{self.__class__.__name__} Model Evaluated")
        # print(f"Mean Squared Error: {mse} \nMean Absolute Error: {mae}\nR2 Score: {r2} \nRoot "
        #       f"Mean Squared Error: {rmse}")
        return mse, mae, r2, rmse


class RandomForest(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = RandomForestRegressor(n_estimators=1000, random_state=111, max_features=70)


class XGBoost(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = XGBRegressor(n_estimators=1000, random_state=111, learning_rate=0.1)


class SVM(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = SVR(kernel='linear', C=1.0, epsilon=0.1)


class GradientBoosting(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = GradientBoostingRegressor(n_estimators=1000, random_state=111, learning_rate=0.1)
