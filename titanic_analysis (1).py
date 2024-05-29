#!/usr/bin/env python
# coding: utf-8

# In[16]:


# Importar bibliotecas necesarias
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix


# In[17]:


# Cargar el dataset
df = pd.read_csv(r'C:\Users\PC\Titanic-Dataset.csv')


# In[18]:


# Análisis exploratorio de datos
print(df.head())
print(df.describe(include='all'))
print(df.isnull().sum())


# In[19]:


# Visualización de datos
sns.countplot(data=df, x='Survived')
plt.title('Distribución de Supervivientes')
plt.show()

sns.countplot(data=df, x='Pclass', hue='Survived')
plt.title('Supervivencia según Clase')
plt.show()

sns.histplot(data=df, x='Age', bins=20, kde=True)
plt.title('Distribución de Edad')
plt.show()

sns.boxplot(data=df, x='Survived', y='Age')
plt.title('Supervivencia según Edad')
plt.show()

sns.countplot(data=df, x='Sex', hue='Survived')
plt.title('Supervivencia según Sexo')
plt.show()

sns.countplot(data=df, x='Embarked', hue='Survived')
plt.title('Supervivencia según Puerto de Embarque')
plt.show()


# In[7]:


# Preprocesamiento de datos
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
print(df.isnull().sum())

df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
df.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)
    


# In[8]:


# Selección de características
correlation = df.corr()
print(correlation['Survived'].sort_values(ascending=False))

features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
X = df[features]
y = df['Survived']


# In[9]:


# División del dataset en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[10]:


# Entrenamiento del modelo de Regresión Logística
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)


# In[11]:


# Entrenamiento del modelo de Árbol de Decisión
tree_clf = DecisionTreeClassifier(max_depth=4, random_state=42)
tree_clf.fit(X_train, y_train)


# In[12]:


# Evaluación del modelo de Regresión Logística
y_pred_log_reg = log_reg.predict(X_test)
print("Regresión Logística")
print(confusion_matrix(y_test, y_pred_log_reg))
print(classification_report(y_test, y_pred_log_reg))


# In[13]:


# Evaluación del modelo de Árbol de Decisión
y_pred_tree = tree_clf.predict(X_test)
print("Árbol de Decisión")
print(confusion_matrix(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))


# In[14]:


# Visualización de resultados
sns.heatmap(confusion_matrix(y_test, y_pred_log_reg), annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión - Regresión Logística')
plt.show()

sns.heatmap(confusion_matrix(y_test, y_pred_tree), annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión - Árbol de Decisión')
plt.show()

plt.figure(figsize=(10, 6))
feature_importances = pd.Series(tree_clf.feature_importances_, index=features)
feature_importances.nlargest(10).plot(kind='barh')
plt.title('Importancia de las Características - Árbol de Decisión')
plt.show()


# In[ ]:




