import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer


train_df=pd.read_csv('C:\\Users\\vaish\\OneDrive\\Desktop\\Loan-approval-prediction\\credit_risk_dataset.csv')



train_df.isnull().sum()

train_df.describe()

train_df.info()

train_df.corr(numeric_only=True)

# Create a holdout test set from the available dataset (no separate test file present)
train_df_raw, test_df_raw = train_test_split(train_df, test_size=0.2, stratify=train_df['loan_status'], random_state=42)

# Keep test IDs for submission and reset indices (fall back to row index if 'id' not present)
if 'id' in test_df_raw.columns:
    test_id = test_df_raw['id'].reset_index(drop=True)
    # Drop the id column from the data used for modeling
    train_df = train_df_raw.drop('id', axis=1).reset_index(drop=True)
    test_df = test_df_raw.drop('id', axis=1).reset_index(drop=True)
else:
    test_id = pd.Series(range(len(test_df_raw)), name='id')
    train_df = train_df_raw.reset_index(drop=True)
    test_df = test_df_raw.reset_index(drop=True)

# Drop unwanted columns if they exist
cols_to_drop = ['person_age','cb_person_cred_hist_length']
for col in cols_to_drop:
    if col in train_df.columns:
        train_df = train_df.drop(columns=[col])
    if col in test_df.columns:
        test_df = test_df.drop(columns=[col])

# Distribution of loan amounts (EDA)
sns.histplot(train_df['loan_amnt'], bins=30, kde=True)
plt.title('Loan Amount Distribution')
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(train_df.corr(numeric_only=True), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Count plot for loan status
sns.countplot(x='loan_status', data=train_df)
plt.title('Loan Status Count')
plt.show()

# Create new features on both training and holdout sets
for df_ in (train_df, test_df):
    df_['income_to_loan_ratio'] = df_['person_income'] / df_['loan_amnt']
    df_['loan_per_emp_year'] = df_['loan_amnt'] / (df_['person_emp_length'] + 1)

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_ohe_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

label_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_pipeline, make_column_selector(dtype_include=['int64', 'float64'])),
        ('cat_ohe', cat_ohe_pipeline, ['person_home_ownership', 'loan_intent', 'cb_person_default_on_file']),
        ('label', label_pipeline, ['loan_grade'])
    ],
    remainder='drop'
)

pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Prepare X and y from the training portion
X = train_df.drop(columns=['loan_status'])
y = train_df['loan_status']

feature_cols = X.columns.tolist()

# Fit the preprocessing pipeline only on training data
X_preprocessed = pipeline.fit_transform(X)
# ensure pipeline output contains no NaNs before SMOTE
if np.isnan(X_preprocessed).any():
    raise ValueError("Pipeline output contains NaN values after preprocessing — check imputers.")
# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Initialize SMOTE
smote = SMOTE(random_state=42)

# Fit SMOTE to the training data
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Check the class distribution after SMOTE
print("Original class distribution:")
print(y_train.value_counts())

print("\nResampled class distribution:")
print(pd.Series(y_resampled).value_counts())

# Train a Random Forest model with the resampled data
model = RandomForestClassifier(random_state=42)
model.fit(X_resampled, y_resampled)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')


# Initialize the Gradient Boosting Classifier
gb_model = GradientBoostingClassifier(random_state=42)

# Fit the model on the resampled data
gb_model.fit(X_resampled, y_resampled)

# Make predictions on the test set
y_pred_gb = gb_model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred_gb))
print(f'Accuracy: {accuracy_score(y_test, y_pred_gb)}')

# Use RandomizedSearchCV with early stopping and parallel jobs (faster than an exhaustive GridSearch)
param_dist = {
    'n_estimators': [50, 100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.6, 0.8, 1.0]
}

est = GradientBoostingClassifier(random_state=42, n_iter_no_change=10, validation_fraction=0.1)
random_search = RandomizedSearchCV(
    est,
    param_distributions=param_dist,
    n_iter=20,           # limits total fits
    cv=3,                # fewer folds
    scoring='accuracy',
    n_jobs=-1,           # use all cores
    verbose=2,
    random_state=42
)

try:
    random_search.fit(X_resampled, y_resampled)
    print("Best parameters:", random_search.best_params_)
    best_gb_model = random_search.best_estimator_
except KeyboardInterrupt:
    print("Search interrupted — falling back to previously fitted gb_model")
    best_gb_model = gb_model

# Make predictions with the best model
y_pred_best = best_gb_model.predict(X_test)

# Evaluate the best model
print(classification_report(y_test, y_pred_best))
print(f'Accuracy: {accuracy_score(y_test, y_pred_best)}')


# Prepare holdout test set features: drop label if present and align columns
X_test_df = test_df.drop(columns=['loan_status']) if 'loan_status' in test_df.columns else test_df.copy()
# Add any missing training columns with NaN and reorder
for col in feature_cols:
    if col not in X_test_df.columns:
        X_test_df[col] = np.nan
X_test_df = X_test_df[feature_cols]

X_test_preprocessed = pipeline.transform(X_test_df)
y_pred = best_gb_model.predict(X_test_preprocessed)


# Create a submission DataFrame
submission = pd.DataFrame({
    'id': test_id,  # Replace test_id with your actual test ID array
    'loan_status': None  # Placeholder for predictions
})

# Now, make predictions using your model
y_pred = best_gb_model.predict(X_test_preprocessed)

# Assign predictions to the submission DataFrame
submission['loan_status'] = y_pred


# Save the best model (GridSearchCV already refits on the provided training data) and the preprocessing pipeline
joblib.dump(best_gb_model, 'gradient_boosting_model.pkl')
joblib.dump(pipeline, 'preprocessing_pipeline.pkl')

# Save the submission file for the holdout test set
submission.to_csv('submission.csv', index=False)
print('Saved model, pipeline, and submission (submission.csv).')

# --- New: load external new data, predict and add a column with predictions ---
def predict_applications(df, save_path=None):
    # ensure an id column exists
    if 'id' not in df.columns:
        df.insert(0, 'id', range(len(df)))
    # drop same unwanted columns if present
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])
    # create engineered features
    df['income_to_loan_ratio'] = df['person_income'] / df['loan_amnt']
    df['loan_per_emp_year'] = df['loan_amnt'] / (df['person_emp_length'] + 1)
    # drop id/label for prediction
    drop_cols = ['id']
    if 'loan_status' in df.columns:
        drop_cols.append('loan_status')
    X_new = df.drop(columns=drop_cols)
    # Ensure same columns/order as training (fill missing with NaN)
    for col in feature_cols:
        if col not in X_new.columns:
            X_new[col] = np.nan
    X_new = X_new[feature_cols]
    # transform and predict
    X_new_preprocessed = pipeline.transform(X_new)
    preds = best_gb_model.predict(X_new_preprocessed)
    df['predicted_loan_status'] = preds
    if save_path:
        df.to_csv(save_path, index=False)
    return df

# -- check for a single application saved as JSON --
single_json = 'new_application.json'
if os.path.exists(single_json):
    single_df = pd.read_json(single_json, orient='records')
    out = predict_applications(single_df, 'new_application_with_prediction.csv')
    print(f'Saved single application predictions to new_application_with_prediction.csv')

# -- fall back to existing batch CSV handling --
new_data_path = r'new_applications.csv'
if os.path.exists(new_data_path):
    new_df = pd.read_csv(new_data_path)
    out = predict_applications(new_df, 'new_applications_with_predictions.csv')
    print('Saved predictions to new_applications_with_predictions.csv')
else:
    print(f'{new_data_path} not found — skipped batch new data prediction.')