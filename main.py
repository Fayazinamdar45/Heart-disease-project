import os
import joblib

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.ensemble import RandomForestRegressor

MODEL_FILE = "Heart_model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(attributes):
    heart_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy="median")),
        ('standardize', StandardScaler())
    ])

    column_transform = ColumnTransformer([
        ("transformed", heart_pipeline, attributes.columns.tolist())
    ])

    return column_transform


if not os.path.exists(MODEL_FILE):
    data = pd.read_csv("heart.csv")

    data['ca_cat'] = pd.cut(data['ca'], bins=[-1, 0, 1, 2, 3, 4], labels=[1, 2, 3, 4, 5])
    
    splitted = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    
    for train, test in splitted.split(data, data['ca_cat']):
        strat_train_set = data.loc[train].drop('ca_cat',axis=1)
        strat_test_set = data.loc[test].drop('ca_cat',axis=1).to_csv("input.csv",index=False)

    heart = strat_train_set.copy()

    heart_features = heart.drop('target',axis=1)
    heart_labels = heart['target']

    pipeline = build_pipeline(heart_features)
    heart_prepared = pipeline.fit_transform(heart)

    # Random Forest
    model = RandomForestRegressor()
    model.fit(heart_prepared, heart_labels)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)

else:
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    inputs = pd.read_csv("input.csv")

    # Separate actual labels from input features
    actual_targets = inputs["target"]
    input_features = inputs.drop("target", axis=1)

    # Use the already-fitted pipeline
    transformed_input = pipeline.transform(input_features)

    # Predict using the trained model
    predictions = model.predict(transformed_input)
    predicted_targets = np.rint(predictions).astype(int)

    accuracy = accuracy_score(actual_targets, predicted_targets)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Confusion matrix:")
    print(confusion_matrix(actual_targets, predicted_targets))

    pd.DataFrame({
        "actual_target": actual_targets,
        "predicted_target": predicted_targets
    }).to_csv("predictions.csv", index=False)

    print("Results written to file successfully.")