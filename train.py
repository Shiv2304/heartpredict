import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE  # For handling class imbalance
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import pickle

# Load your dataset with 1024 rows
print("Loading the dataset...")
data = pd.read_csv('heart_data.csv')  # Ensure this matches your file name
print(f"Number of records loaded: {len(data)}")  # Verify the row count

# Check for missing values
print("Checking for missing values in the dataset...")
print(data.isnull().sum())

# Replace any '?' with NaN (just in case)
data = data.replace('?', np.nan)

# Handle missing values
print("Handling missing values...")
numeric_columns = data.select_dtypes(include=np.number).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())

# Data cleaning: Adjust outlier removal
print("Cleaning data (removing outliers)...")
data = data[(data['trestbps'] >= 70) & (data['trestbps'] <= 220)]
data = data[(data['chol'] >= 80) & (data['chol'] <= 450)]
data = data[(data['thalach'] >= 50) & (data['thalach'] <= 220)]
print(f"Number of records after outlier removal: {len(data)}")

# Remove duplicates
data = data.drop_duplicates()
print(f"Number of records after removing duplicates: {len(data)}")

# Separate features and target
X = data.drop('target', axis=1)
y = data['target']

# Check class distribution before SMOTE
print("Class distribution before SMOTE:")
print(y.value_counts(normalize=True))

# Apply SMOTE to balance the dataset
print("Applying SMOTE to balance the dataset...")
smote = SMOTE(random_state=0)
X, y = smote.fit_resample(X, y)
print(f"Dataset shape after SMOTE: {X.shape}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
pickle.dump(scaler, open('models/scaler.pkl', 'wb'))

# KNN Model: Broader range of k
print("Training KNN model...")
knn_scores = []
for k in range(1, 31):
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train_scaled, y_train)
    knn_score = round(knn_classifier.score(X_test_scaled, y_test), 2)
    knn_scores.append(knn_score)
    print(f"KNN with k={k}: {knn_score}")

best_knn_score = max(knn_scores)
best_k = knn_scores.index(best_knn_score) + 1
print(f"Best KNN model has k={best_k} with accuracy {best_knn_score}")
knn_classifier = KNeighborsClassifier(n_neighbors=best_k)
knn_classifier.fit(X_train_scaled, y_train)

# Cross-validation for KNN
cv_scores_knn = cross_val_score(knn_classifier, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"KNN 5-fold CV accuracy: {cv_scores_knn.mean():.2f} (+/- {cv_scores_knn.std() * 2:.2f})")

pickle.dump(knn_classifier, open('models/knn_model.pkl', 'wb'))

# Decision Tree Model: Deeper trees
print("Training Decision Tree model...")
dt_scores = []
for i in range(1, len(X.columns) + 1):
    dt_classifier = DecisionTreeClassifier(max_features=i, random_state=0, max_depth=20)
    dt_classifier.fit(X_train_scaled, y_train)
    dt_score = round(dt_classifier.score(X_test_scaled, y_test), 2)
    dt_scores.append(dt_score)
print("Decision Tree scores:", dt_scores)

best_dt_score = max(dt_scores)
best_max_features = dt_scores.index(best_dt_score) + 1
print(f"Best Decision Tree model has max_features={best_max_features} with accuracy {best_dt_score}")
dt_classifier = DecisionTreeClassifier(max_features=best_max_features, random_state=0, max_depth=20)
dt_classifier.fit(X_train_scaled, y_train)

# Cross-validation for Decision Tree
cv_scores_dt = cross_val_score(dt_classifier, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"Decision Tree 5-fold CV accuracy: {cv_scores_dt.mean():.2f} (+/- {cv_scores_dt.std() * 2:.2f})")

pickle.dump(dt_classifier, open('models/dt_model.pkl', 'wb'))

# Random Forest Model: Reduced tuning space
print("Training Random Forest model...")
rf_scores = []
estimators = [50, 100, 200]  # Reduced range
max_depths = [10, 20, 30]    # Reduced range
best_rf_score = 0
best_rf_params = {}

for n_estimators in estimators:
    for max_depth in max_depths:
        rf_classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=0)
        rf_classifier.fit(X_train_scaled, y_train)
        rf_score = round(rf_classifier.score(X_test_scaled, y_test), 2)
        print(f"Random Forest with n_estimators={n_estimators}, max_depth={max_depth}: {rf_score}")
        if rf_score > best_rf_score:
            best_rf_score = rf_score
            best_rf_params = {'n_estimators': n_estimators, 'max_depth': max_depth}
        rf_scores.append(rf_score)

print(f"Best Random Forest model has n_estimators={best_rf_params['n_estimators']}, max_depth={best_rf_params['max_depth']} with accuracy {best_rf_score}")
rf_classifier = RandomForestClassifier(n_estimators=best_rf_params['n_estimators'], max_depth=best_rf_params['max_depth'], random_state=0)
rf_classifier.fit(X_train_scaled, y_train)

# Cross-validation for Random Forest
cv_scores_rf = cross_val_score(rf_classifier, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"Random Forest 5-fold CV accuracy: {cv_scores_rf.mean():.2f} (+/- {cv_scores_rf.std() * 2:.2f})")

# Feature Importance from Random Forest
feature_importances = pd.DataFrame(rf_classifier.feature_importances_, index=X.columns, columns=['importance']).sort_values('importance', ascending=False)
print("Feature Importances:\n", feature_importances)

pickle.dump(rf_classifier, open('models/rf_model.pkl', 'wb'))

# SVM Model: Tune C and kernel
print("Training SVM model...")
svm_scores = []
C_values = [0.1, 1.0, 10.0]
kernels = ['rbf', 'linear']
for C in C_values:
    for kernel in kernels:
        svm_classifier = SVC(probability=True, random_state=0, C=C, kernel=kernel)
        svm_classifier.fit(X_train_scaled, y_train)
        svm_score = round(svm_classifier.score(X_test_scaled, y_test), 2)
        print(f"SVM with C={C}, kernel={kernel}: {svm_score}")
        svm_scores.append(svm_score)

best_svm_score = max(svm_scores)
best_svm_idx = svm_scores.index(best_svm_score)
best_C = C_values[best_svm_idx // len(kernels)]
best_kernel = kernels[best_svm_idx % len(kernels)]
print(f"Best SVM model has C={best_C}, kernel={best_kernel} with accuracy {best_svm_score}")
svm_classifier = SVC(probability=True, random_state=0, C=best_C, kernel=best_kernel)
svm_classifier.fit(X_train_scaled, y_train)

# Cross-validation for SVM
cv_scores_svm = cross_val_score(svm_classifier, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"SVM 5-fold CV accuracy: {cv_scores_svm.mean():.2f} (+/- {cv_scores_svm.std() * 2:.2f})")

pickle.dump(svm_classifier, open('models/svm_model.pkl', 'wb'))

# Logistic Regression Model
print("Training Logistic Regression model...")
lr_classifier = LogisticRegression(random_state=0)
lr_classifier.fit(X_train_scaled, y_train)
lr_score = round(lr_classifier.score(X_test_scaled, y_test), 2)
print(f"Logistic Regression accuracy: {lr_score}")

# Cross-validation for Logistic Regression
cv_scores_lr = cross_val_score(lr_classifier, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"Logistic Regression 5-fold CV accuracy: {cv_scores_lr.mean():.2f} (+/- {cv_scores_lr.std() * 2:.2f})")

pickle.dump(lr_classifier, open('models/lr_model.pkl', 'wb'))

# Determine the best model based on test accuracy
model_scores = {
    'KNN': best_knn_score,
    'Decision Tree': best_dt_score,
    'Random Forest': best_rf_score,
    'SVM': best_svm_score,
    'Logistic Regression': lr_score
}

best_model_name = max(model_scores, key=model_scores.get)
best_model_score = model_scores[best_model_name]
print(f"Best model (based on test accuracy) is {best_model_name} with accuracy {best_model_score}")

# Final message
print("Model training and evaluation completed successfully!")
print("ALL models have been saved to the 'models' directory.")