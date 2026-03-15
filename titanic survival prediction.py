import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np

# ==========================================
# 1. LOAD DATA
# ==========================================
df = pd.read_csv('Titanic-Dataset.csv')
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# ==========================================
# 2. DATA PREPROCESSING
# ==========================================
# Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)

# Remove unnecessary columns
df_processed = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Convert categorical variables to numeric
le_sex = LabelEncoder()
le_embarked = LabelEncoder()

df_processed['Sex'] = le_sex.fit_transform(df_processed['Sex'])
df_processed['Embarked'] = le_embarked.fit_transform(df_processed['Embarked'])

print("\nProcessed data:")
print(df_processed.head())

# ==========================================
# 3. PREPARE FEATURES AND TARGET
# ==========================================
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = df_processed[features]
y = df_processed['Survived']

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

# ==========================================

# ==========================================
# 5. TRAIN THE MODEL
# ==========================================
model = RandomForestClassifier()

print("\nTraining the model...")
print("Model training completed!")

# ==========================================
# 6. EVALUATE THE MODEL
# ==========================================


print(f"\n{'='*50}")
print(f"{'='*50}")

print("\nClassification Report:")

print("\nConfusion Matrix:")
x = df[features]
y = df['Survived']

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=42)


model = RandomForestClassifier()

# Train the model
model.fit(x_train, y_train)

#Feature Importance
feature_importance = pd.DataFrame({
     'Feature': features,
     'Importance': model.feature_importances_
}).sort_values('Importance',ascending=False)

print("\nFeature Importance:")
print(feature_importance)
#Load test dataset
test_data = pd.read_csv("test.csv")

#Extract PassengerId
passenger_ids = test_data['PassengerId']

#Select features for prediction
X_Submission = test_data[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]

# Make predictions
predictions = model.predict(X_Submission)

# Create submission file
submission = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived': predictions
})

submission.to_csv('submission.csv', index=False)
print("\nSubmission file created: submission.csv")
print(submission.head(10))
