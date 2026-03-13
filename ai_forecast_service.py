import os
import joblib


MODEL_PATH = "forecast_model.pkl"


class AIForecastService:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            self.model = None

    def predict_cases(
        self,
        temperature,
        humidity,
        rainfall,
        cases_count,
        severity_avg
    ):
        if self.model is None:
            raise RuntimeError("Forecast model not found. Train it first.")

        X = [[
            float(temperature),
            float(humidity),
            float(rainfall),
            float(cases_count),
            float(severity_avg)
        ]]

        pred = float(self.model.predict(X)[0])

        if pred < 3:
            risk = "منخفض"
        elif pred < 8:
            risk = "متوسط"
        else:
            risk = "مرتفع"

        return {
            "predicted_cases": round(pred, 2),
            "risk_level_ar": risk
        }
