#!/usr/bin/env python
# coding: utf-8



# Project: Customer Churn Prediction Pipeline (Applied ML)




# Authors: Mahi Chudela, Damilare Ajiboye (team project)




#Part 1: Model 1 Classification Model to compare accuracies of all 4 algorithms 




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, classification_report, confusion_matrix




# Loading dataset 
df = pd.read_csv("../data/Bank.csv")
df.head()




# Droping customer_id column as it's not useful for analysis
df.drop(columns=["customer_id"], inplace=True)




# Encoding categorical columns like 'country' and 'gender'
label_encoders = {}
for col in ["country", "gender"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le




# Standardize numerical columns to bring them to the same scale
scaler = StandardScaler()
num_cols = ["credit_score", "age", "tenure", "balance", "products_number", "estimated_salary"]
df[num_cols] = scaler.fit_transform(df[num_cols])




# Split data for classification (churn prediction)
X = df.drop(columns=["churn"])
y = df["churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




# Train classification models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC()
}

classification_results = {}
fig, ax = plt.subplots(figsize=(10, 6))

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    classification_results[name] = accuracy
    print(f"{name} Accuracy: {accuracy * 100:.2f}%")
    ax.bar(name, accuracy, label=name)

ax.set_title("Classification Model Accuracy Comparison")
ax.set_ylabel("Accuracy")
ax.set_ylim([0, 1])
ax.legend()
plt.show()




# Confusion Matrices for Classification Models
for name, model in models.items():
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()



















#Part2: Model 2: Regression model to predict the credit score correctly on the basis of estimated salary, age, and balance available"




from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score





# Loading dataset 
df = pd.read_csv("../data/Bank.csv")
df.head()




# Droping customer_id column as it's not useful for analysis
df.drop(columns=["customer_id"], inplace=True)




# Encoding categorical columns like 'country' and 'gender'
label_encoders = {}
for col in ["country", "gender"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    




# Standardizing numerical columns to bring them to the same scale
scaler = StandardScaler()
num_cols = ["credit_score", "age", "tenure", "balance", "products_number", "estimated_salary"]
df[num_cols] = scaler.fit_transform(df[num_cols])




# Regression task (credit score Prediction on the basis of balance, salary and age)
X_reg = df[['balance', 'age', 'estimated_salary']]
y_reg = df['credit_score']

# OPTIONAL: Uncomment below if your target is heavily skewed
#y_reg = np.log1p(y_reg)

# Feature Scaling
scaler = StandardScaler()
X_reg_scaled = scaler.fit_transform(X_reg)

# Split the dataset
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg_scaled, y_reg, test_size=0.2, random_state=0)

# Define models
regression_models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    "Support Vector Regressor": SVR()
}

print("\n Regression Results:\n")

# Evaluate models
for model_name, model in regression_models.items():
    model.fit(X_train_reg, y_train_reg)
    y_pred_reg = model.predict(X_test_reg)

    # OPTIONAL: Use below if using log1p on y_reg
    #y_pred_reg = np.expm1(y_pred_reg)
    #y_test_reg_exp = np.expm1(y_test_reg)

    mae = mean_absolute_error(y_test_reg, y_pred_reg)
    mse = mean_squared_error(y_test_reg, y_pred_reg)

    print(f"{model_name}")
    print(f" - Mean Absolute Error: {mae:.4f}")
    print(f" - Mean Squared Error: {mse:.4f}")
    print()




# Store results
mae_scores = []
mse_scores = []
model_names = []

# Collecting data from trained models
for model_name, model in regression_models.items():
    model.fit(X_train_reg, y_train_reg)
    y_pred_reg = model.predict(X_test_reg)
    
    mae = mean_absolute_error(y_test_reg, y_pred_reg)
    mse = mean_squared_error(y_test_reg, y_pred_reg)
    
    mae_scores.append(mae)
    mse_scores.append(mse)
    model_names.append(model_name)

# Plotting the graph
x = range(len(model_names))
bar_width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar([i - bar_width/2 for i in x], mae_scores, bar_width, label='MAE', color='green')
bars2 = ax.bar([i + bar_width/2 for i in x], mse_scores, bar_width, label='MSE', color='red')

# Adding annotations
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 5), textcoords='offset points', ha='center', fontsize=9, color='navy')

for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 5), textcoords='offset points', ha='center', fontsize=9, color='darkred')

# Styling the graph to make it look creative
ax.set_xlabel('Regression Model')
ax.set_ylabel('Error')
ax.set_title('Regression Error Comparison (MAE vs MSE)')
ax.set_xticks(list(x))
ax.set_xticklabels(model_names)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()




#showing the actual and the predicted values through another graph for a better insight


N = 100 #number of random samples to show on the graph so it gets easier to compare the data otherwise all the dots will look miserable
indices = np.random.choice(len(y_test_reg), N, replace=False) # Randomly select N unique indices from the test set so it want give the same type of output everytime

plt.figure(figsize=(10, 6))

for model_name, model in regression_models.items():
    model.fit(X_train_reg, y_train_reg)
    y_pred_reg = model.predict(X_test_reg)

    y_sampled = y_test_reg.iloc[indices]
    y_pred_sampled = y_pred_reg[indices]

    plt.scatter(indices, y_sampled, 
                color='black', alpha=0.6, label='Actual Values' if model_name == "Linear Regression" else "", marker='o')
    plt.scatter(indices, y_pred_sampled, 
                alpha=0.5, label=f'Predicted ({model_name})', marker='x')

plt.title("Actual vs Predicted Credit Scores (Sampled Points)")
plt.xlabel("Sample Index")
plt.ylabel("Credit Score")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
























#Part 3 # MODEL 3: Regression Model to Predict Churn Rate and Identify Highest Churn Countries




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score




#loding the dataset
df = pd.read_csv("../data/Bank.csv")




dfmodel3 = df.drop(['customer_id', 'gender'], axis=1).copy()
# Encodeing country to numeric
dfmodel3['country_encoded'] = LabelEncoder().fit_transform(dfmodel3['country'])




# Selecting features (using country as a feature as our aim not to predict the chrun rate by country)
features = ['credit_score', 'age', 'tenure', 'balance', 'products_number',
            'credit_card', 'active_member', 'estimated_salary', 'country_encoded']
X = dfmodel3[features]
y = dfmodel3['churn']  # Regression target for now




# Standardizing features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)




#Model validation using K-Fold function as per lecturer suggested  
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
model = RandomForestRegressor(random_state=42)




cv_scores = cross_val_score(model, X_scaled, y, cv=kfold, scoring='r2')
print(f"Cross-Validated R² Scores: {cv_scores}")
print(f"Mean R² Score: {np.mean(cv_scores):.4f}")




# Training on full dataset
model.fit(X_scaled, y)
dfmodel3['predicted_churn'] = model.predict(X_scaled)




# Churn by Country Analysis 
churn_by_country = dfmodel3.groupby('country')['predicted_churn'].mean().reset_index()
churn_by_country = churn_by_country.sort_values(by='predicted_churn', ascending=False)




# BPlottingn the bar 
plt.figure(figsize=(8, 6))
sns.barplot(data=churn_by_country, x='predicted_churn', y='country', palette='Reds_r')
plt.title('Showing Predicted Churn Rate by Country')
plt.xlabel('Average Predicted Churn')
plt.ylabel('Country')
plt.tight_layout()
plt.show()




# Highlighing through Choropleth Map to make it creative
fig = px.choropleth(
    churn_by_country,
    locations='country',
    locationmode='country names',
    color='predicted_churn',
    color_continuous_scale='Reds',
    title='Predicted Churn by Country',
    template='plotly_dark'
)
fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
fig.show()




# Clean Visualization: Actual vs Predicted Churn (Sampled) just to highlight how well the model is trained 

# Using test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Creating DataFrame
compare_df = pd.DataFrame({
    'Actual Churn': y_test.values,
    'Predicted Churn': y_pred
}).reset_index(drop=True)

# Sample 150 random points for better visualization otherwise it will look very messy we can change the selection of the data to be taken by changing the radom state to any random number
sampled_df = compare_df.sample(n=150, random_state=42).sort_index()

# Plotting the graph now
plt.figure(figsize=(10, 5))
plt.plot(sampled_df.index, sampled_df['Actual Churn'], 'ro-', alpha=0.6, label='Actual Churn')
plt.plot(sampled_df.index, sampled_df['Predicted Churn'], 'bo-', alpha=0.5, label='Predicted Churn')
plt.title('Comparison: Actual vs Predicted Churn')
plt.xlabel('Sample Index')
plt.ylabel('Churn Value')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

# showing the Correlation
correlation = np.corrcoef(sampled_df['Actual Churn'], sampled_df['Predicted Churn'])[0, 1]
print(f"Correlation (Just for Sample): {correlation:.4f}")
























#Part 4:MODEL 4:  Classification to Identify Key Features That Predict Churn




#importing libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report




#loading the dataset
df = pd.read_csv("../data/Bank.csv")




#creating a new DataFrame called dfmodel4 and dropping customer_id as it is not useful for prediction
dfmodel4 = df.copy()
dfmodel4 = dfmodel4.drop(['customer_id'], axis=1)




#encoding 'gender' and 'country' as numeric values using LabelEncoder
dfmodel4['gender'] = LabelEncoder().fit_transform(dfmodel4['gender'])
dfmodel4['country'] = LabelEncoder().fit_transform(dfmodel4['country'])




#defining the features (X) and the target variable (y)
X = dfmodel4.drop(['churn'], axis=1)
y = dfmodel4['churn']




#now scaling all features to normalize the range using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)




#splitting the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)




# creating a Random Forest classifier model
rf_model = RandomForestClassifier(random_state=42)

#applying K-Fold Cross-Validation with 5 folds
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_model, X_scaled, y, cv=kfold, scoring='accuracy')
print("Cross-Validation Accuracy Scores:", cv_scores)
print("Mean Accuracy:", np.mean(cv_scores).round(4))





# fitting the model on training data for SHAP analysis
rf_model.fit(X_train, y_train)




# I am extracting feature importances from the trained model
importances = rf_model.feature_importances_

# I am creating a DataFrame for visualizing feature importances
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# I am plotting a bar graph to highlight key churn-driving features
plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
plt.title(' Top Features Driving Customer Churn (Random Forest)')
plt.xlabel('Feature Importance Score')
plt.ylabel('Feature')
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()














































