import pickle
import numpy as np

with open("model/my_model.pkl", "rb") as f:
    model = pickle.load(f)

X_test = np.random.rand(1, 10)  # 10 features
print("Prediction:", model.predict(X_test))
print("Probabilities:", model.predict_proba(X_test))
