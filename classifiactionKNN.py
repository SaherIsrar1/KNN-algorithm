#!/usr/bin/env python
# coding: utf-8

# #### Import Libraries

# In[23]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss

# #### Import Libraries

# In[4]:


df = pd.read_csv('KNNAlgorithmDataset.csv')

# In[5]:


print(df.shape)

# In[6]:


print(df.head(5))

# In[8]:


df.info()

# #### Drop Unnecessary Columns

# In[12]:


df = df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')

# In[13]:


print(df.shape)

# #### Check Missing Values

# In[14]:


print( df.isnull().values.any())

# #### Encode Target and Define Features

# In[15]:


le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'])  # Malignant = 1, Benign = 0

X = df.drop(columns=['diagnosis']).values
y = df['diagnosis'].values

# #### Train-Test Split

# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# #### Feature Scaling

# In[17]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# #### Scratch KNN Classifier Implementation

# In[25]:


class ScratchKNNClassifier:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _compute_distances(self, X):
        # Euclidean distance formula calculation
        return np.sqrt(np.sum((X[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]) ** 2, axis=2))

    def predict(self, X):
        distances = self._compute_distances(X)
        predictions = []
        for row_dist in distances:
            nearest_indices = np.argsort(row_dist)[:self.k]
            nearest_labels = self.y_train[nearest_indices]
            # Majority vote
            most_common = Counter(nearest_labels).most_common(1)
            predictions.append(most_common[0][0])
        return np.array(predictions)

    def predict_proba(self, X):
        distances = self._compute_distances(X)
        probas = []
        for row_dist in distances:
            nearest_indices = np.argsort(row_dist)[:self.k]
            nearest_labels = self.y_train[nearest_indices]
            prob_1 = np.sum(nearest_labels == 1) / self.k
            probas.append([1 - prob_1, prob_1])
        return np.array(probas)

# #### Train and Predict

# In[26]:


knn_clf = ScratchKNNClassifier(k=5)
knn_clf.fit(X_train, y_train)
y_pred = knn_clf.predict(X_test)

# #### Calculate Metrics

# In[27]:


accuracy = np.mean(y_pred == y_test) * 100
mse = np.mean((y_test - y_pred) ** 2)
loss = log_loss(y_test, knn_clf.predict_proba(X_test))

# #### Evaluation Results

# In[28]:


print(f"Test Accuracy: {accuracy:.2f}%")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Log Loss: {loss:.4f}")

# #### Target Variable Distribution

# In[29]:


plt.figure(figsize=(6, 4))
sns.countplot(x=y, palette="Set2")
plt.title("Diagnosis Distribution (0: Benign, 1: Malignant)")
plt.xlabel("Diagnosis Class")
plt.ylabel("Count")
plt.show()

# #### KNN Accuracy Across Different K Values

# In[30]:


k_values = range(1, 16)
accuracies = []
for k in k_values:
  knn = ScratchKNNClassifier(k=k)
  knn.fit(X_train, y_train)
  y_pred_k = knn.predict(X_test)
  acc = np.mean(y_pred_k == y_test) * 100
  accuracies.append(acc)

plt.figure(figsize=(8, 5))
plt.plot(
    k_values,
    accuracies,
    marker="o",
    linestyle="-",
    color="b",
    linewidth=2,
    markersize=8,
)
plt.title("KNN Accuracy Across Different K Values")
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Test Accuracy (%)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

# #### Feature Scatter Plot (Radius Mean vs Texture Mean)

# In[31]:


plt.figure(figsize=(8, 6))
plt.scatter(
    df.iloc[:, 0], df.iloc[:, 1], c=y, cmap="coolwarm", alpha=0.7, edgecolor="k"
)
plt.title("Radius Mean vs Texture Mean by Diagnosis")
plt.xlabel("Mean Radius")
plt.ylabel("Mean Texture")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

# In[ ]:



