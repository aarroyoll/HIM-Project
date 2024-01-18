import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import numpy as np

# Read the dataset
drug_data = pd.read_excel('/Users/anabanana/Desktop/EPBI_8208/Final Project/Data/DrugData_Subset.xlsx')
print(drug_data)

# All descriptive statistics were done in R

# Correlations
drug_data.corr()

# Replace blank values with zeros in the 'VAL30USE' column  ( blank=did not drive drunk)
drug_data['VAL30USE'].fillna(0, inplace=True)
threshold = 1
drug_data['VAL30USE'] = drug_data['VAL30USE'].apply(lambda x: 1 if x >= threshold else 0)

# Define X & Y
X = drug_data.iloc[:, 1:14]  # all features
Y = drug_data['VAL30USE']    # target output (val30use)

# Create a pipeline with an imputer and SelectKBest
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Use mean imputation for missing values
    ('selector', SelectKBest(score_func=chi2, k=3))
])

# Fit the pipeline with the data
pipeline.fit(X, Y)

# Get the selected features
selected_features = pipeline.named_steps['selector'].get_support(indices=True)
print(f"Selected feature indices: {selected_features}")

# Get the scores and feature names after SelectKBest
df_scores = pd.DataFrame(pipeline.named_steps['selector'].scores_)
df_columns = pd.DataFrame(X.columns)

# Combine feature names and scores into one df
features_scores = pd.concat([df_columns, df_scores], axis=1)
features_scores.columns = ['Features', 'Score']

# Sort features based on their scores
features_scores = features_scores.sort_values(by='Score', ascending=False)
print(features_scores)

# Select features for model
X = drug_data[['IRALCFQ', 'POUNDS', 'WORKSTAT',]]  # top 3 features
Y = drug_data['VAL30USE']  # target output

# Adjust shape of target variable Y using ravel()
Y = Y.values.ravel()

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=100)

# Create a SimpleImputer to handle missing values in features
feature_imputer = SimpleImputer(strategy='mean')  # Use mean imputation for missing values in the features
X_imputed = feature_imputer.fit_transform(X)

# Convert the imputed data back into a df with column names
X_imputed = pd.DataFrame(X_imputed, columns=X.columns)

# train and predict using adjusted data
X_train, X_test, y_train, y_test = train_test_split(X_imputed, Y, test_size=0.4, random_state=100)

# Calculate class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(Y), y=Y)

# Create Logistic Regression model with balanced class weights
logreg = LogisticRegression(class_weight={0: class_weights[0], 1: class_weights[1]})
logreg.fit(X_train, y_train)

# Predict the likelihood of drunk driving
y_pred = logreg.predict(X_test)
print(X_test)  # Test dataset
print(y_pred)  # Predicted values

# Evaluate the model's performance

# Classification metrics

print('Accuracy:', metrics.accuracy_score(y_test, y_pred))
print('Recall:', metrics.recall_score(y_test, y_pred, zero_division=1))
print('Precision:', metrics.precision_score(y_test, y_pred, zero_division=1))
print('CL Report:')
print(metrics.classification_report(y_test, y_pred, zero_division=1))

# Calculate ROC curve and AUC
y_pred_proba = logreg.predict_proba(X_test)[:, 1]
false_positive_rate, true_positive_rate, _ = metrics.roc_curve(y_test, y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

# Plot ROC curve
plt.plot(false_positive_rate, true_positive_rate, label="AUC=" + str(auc))
plt.title('ROC Curve')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

# AUC = 0.8831

# with sex and cigarette use
# Predict drunk driving by sex

features_with_sex = drug_data[['IRSEX', 'IRALCFQ', 'POUNDS', 'WORKSTAT', 'CIGFLAG']]  # add sex and cig use
sex_predictions = logreg.predict(features_with_sex)


