import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import pickle
import os

UNIVERSAL_FEATURES = [
    'Age', 'Gender', 'MaritalStatus', 'DistanceFromHome',
    'Department', 'JobRole', 'JobLevel', 'JobSatisfaction',
    'OverTime', 'MonthlyIncome', 'PercentSalaryHike',
    'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole',
    'YearsSinceLastPromotion', 'WorkLifeBalance', 'JobInvolvement',
    'EnvironmentSatisfaction', 'RelationshipSatisfaction',
    'NumCompaniesWorked', 'TrainingTimesLastYear', 'BusinessTravel'
]

CATEGORICAL_COLS = [
    'Gender', 'MaritalStatus', 'Department',
    'JobRole', 'OverTime', 'BusinessTravel'
]

def load_and_preprocess(df, target_col='Attrition'):
    df = df.copy()

    # Keep only available universal features
    available = [f for f in UNIVERSAL_FEATURES if f in df.columns]
    df = df[available + [target_col]]

    # Fill missing values
    for col in df.columns:
        if col == target_col:
            continue
        try:
            if df[col].dtype == object or str(df[col].dtype) == 'string':
                df[col] = df[col].fillna(
                    df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                )
            elif pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna('Unknown')
        except:
            df[col] = df[col].fillna('Unknown')

    # Encode target
    df[target_col] = (df[target_col].astype(str).str.strip().str.lower() == 'yes').astype(int)

    # Encode categoricals
    encoders = {}
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    return df, encoders, available

def train_model(df, target_col='Attrition'):
    df_processed, encoders, available_features = load_and_preprocess(df, target_col)

    X = df_processed.drop(columns=[target_col])
    y = df_processed[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    min_class = y_train.value_counts().min()
    if min_class >= 6:
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_prob)

    model_data = {
        'model': model,
        'encoders': encoders,
        'features': list(X.columns),
        'available_features': available_features,
        'accuracy': report['accuracy'],
        'auc': auc
    }

    os.makedirs('model', exist_ok=True)
    with open('model/universal_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)

    print(f"✅ Model trained! Accuracy: {report['accuracy']*100:.1f}% | AUC: {auc:.4f}")
    return model_data