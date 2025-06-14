import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Step 1: Loading the data
train_df = pd.read_csv(r"C:\Users\Priyatanshu Ghosh\Documents\Summer Analytics\hacktrain.csv")
test_df = pd.read_csv(r"C:\Users\Priyatanshu Ghosh\Documents\Summer Analytics\hacktest.csv")

# Step 2: Dropping the 'Unnamed: 0' column if it exists
train_df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
test_df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')

# Step 3: Separating features and target
X_train_raw = train_df.drop(columns=['ID', 'class'])
y_train_raw = train_df['class']
X_test_raw = test_df.drop(columns=['ID'])

# Step 4: Imputing missing values using mean
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train_raw)
X_test_imputed = imputer.transform(X_test_raw)

# Step 5: Feature engineering – rolling window stats & first-order differences
def add_features(X):
    df = pd.DataFrame(X)
    
    # Basic statistics
    df['mean'] = np.mean(X, axis=1)
    df['std'] = np.std(X, axis=1)
    df['min'] = np.min(X, axis=1)
    df['max'] = np.max(X, axis=1)
    
    # First-order differences (captures sudden changes)
    df['diff_1'] = np.mean(np.diff(X, axis=1), axis=1)
    
    # Rolling window stats (captures local trends)
    df['rolling_mean'] = df.iloc[:, :-1].rolling(window=3, axis=1).mean().mean(axis=1)
    df['rolling_std'] = df.iloc[:, :-1].rolling(window=3, axis=1).std().mean(axis=1)

    df.columns = df.columns.astype(str)
    return df

X_train_fe = add_features(X_train_imputed)
X_test_fe = add_features(X_test_imputed)

# Step 6: Scaling features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_fe)
X_test_scaled = scaler.transform(X_test_fe)

# Step 7: Encoding labels
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train_raw)

# Step 8: Fine-tuning Logistic Regression hyperparameters
param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'class_weight': [None, 'balanced']}
grid_search = GridSearchCV(LogisticRegression(max_iter=5000), param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train_encoded)

best_model = grid_search.best_estimator_

# Step 9: Predict on test set
y_test_pred_encoded = best_model.predict(X_test_scaled)
y_test_pred = le.inverse_transform(y_test_pred_encoded)

# Step 10: Save predictions to submission file
submission = pd.DataFrame({
    'ID': test_df['ID'],
    'class': y_test_pred
})
submission.to_csv("ndvi_submission_4.csv", index=False)

print("✅ Submission file saved as 'ndvi_submission_4.csv'")