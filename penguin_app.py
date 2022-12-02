import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier

# Load the DataFrame
csv_file = 'penguin.csv'
df = pd.read_csv(csv_file)

# Display the first five rows of the DataFrame
df.head()

# Drop the NAN values
df = df.dropna()

# Add numeric column 'label' to resemble non numeric column 'species'
df['label'] = df['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo':2})


# Convert the non-numeric column 'sex' to numeric in the DataFrame
df['sex'] = df['sex'].map({'Male':0,'Female':1})

# Convert the non-numeric column 'island' to numeric in the DataFrame
df['island'] = df['island'].map({'Biscoe': 0, 'Dream': 1, 'Torgersen':2})


# Create X and y variables
X = df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


# Build a SVC model using the 'sklearn' module.
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
svc_score = svc_model.score(X_train, y_train)

# Build a LogisticRegression model using the 'sklearn' module.
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_score = log_reg.score(X_train, y_train)

# Build a RandomForestClassifier model using the 'sklearn' module.
rf_clf = RandomForestClassifier(n_jobs = -1)
rf_clf.fit(X_train, y_train)
rf_clf_score = rf_clf.score(X_train, y_train)

@st.cache()
def predict(model, island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex):
  if sex == 'Male':
  	sex = 0
  else:
  	sex = 1

  if island == 'Biscoe':
  	island = 0
  elif island == 'Dream':
  	island = 1
  else:
  	island = 2

  penguin_species = svc_model.predict([[island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex]])
  penguin_species = penguin_species[0]
  if penguin_species == 0:
    return "Adelie"
  elif penguin_species == 1:
    return "Chinstrap"
  else:
    return "Gentoo"

# Design the App
st.sidebar.title("Penguin Species Prediction App")      

b_len = st.sidebar.slider("Bill Length (in mm)", float(df["bill_length_mm"].min()), float(df["bill_length_mm"].max()))
b_dep = st.sidebar.slider("Bill Depth (in mm)", float(df["bill_depth_mm"].min()), float(df["bill_depth_mm"].max()))
f_len = st.sidebar.slider("Flipper Length (in mm)", float(df["flipper_length_mm"].min()), float(df["flipper_length_mm"].max()))
b_mass = st.sidebar.slider("Body mass (in g)", float(df["body_mass_g"].min()), float(df["body_mass_g"].max()))

sex = st.sidebar.selectbox('Sex', ('Male', 'Female'))
island = st.sidebar.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))

classifier = st.sidebar.selectbox('Classifier', ('Support Vector Machine', 'Logistic Regression', 'Random Forest Classifier'))

if st.sidebar.button("Predict"):
  if classifier == 'Support Vector Machine':
    species_type = predict(svc_model, island, b_len, b_dep, f_len, b_mass, sex)
    score = svc_model.score(X_train, y_train)

  elif classifier =='Logistic Regression':
    species_type = predict(log_reg, island, b_len, b_dep, f_len, b_mass, sex)
    score = log_reg.score(X_train, y_train)

  else:
    species_type = predict(rf_clf, island, b_len, b_dep, f_len, b_mass, sex)
    score = rf_clf.score(X_train, y_train)
  
  st.write("Species predicted:", species_type)
  st.write("Accuracy score of this model is:", score)


