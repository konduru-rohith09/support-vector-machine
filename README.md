# Support Vector Machine (SVM) Classifier – Iris Dataset

## Project Overview
This project demonstrates the use of a **Support Vector Machine (SVM)** classifier to predict the species of the Iris flower. SVM is a **supervised machine learning algorithm** that finds the **optimal hyperplane** to separate classes with maximum margin.  

---

## Key Concepts

### 1. Support Vector Machine (SVM)
- Finds a **decision boundary (hyperplane)** separating classes.
- **Support Vectors** are points closest to the hyperplane defining the margin.
- Can handle **linear and non-linear classification** using **kernel functions**.

### 2. Regularization (`C`)
- Controls overfitting by penalizing misclassification.
- **High `C`** → strict, smaller margin, may overfit.
- **Low `C`** → allows misclassifications, larger margin, better generalization.

### 3. Kernel Function
- Transforms data into higher dimensions for non-linear separation.
- Common kernels:
  - `linear` – straight-line separation
  - `poly` – polynomial boundary
  - `rbf` – Gaussian, widely used
  - `sigmoid` – rarely used

### 4. Gamma (`γ`)
- Only for `rbf`, `poly`, `sigmoid`
- Controls influence of a single training point:
  - High `gamma` → affects small region → complex boundary → overfitting
  - Low `gamma` → affects larger region → smoother boundary → underfitting

---


## Dataset
- **Iris dataset**: 150 samples, 4 features (`sepal length`, `sepal width`, `petal length`, `petal width`)  
- **Classes**: `setosa`, `versicolor`, `virginica`
- **Iris dataset**: 150 samples, 4 features (`sepal length`, `sepal width`, `petal length`, `petal width`)  
- **Classes**: `setosa`, `versicolor`, `virginica`


## Scatter Plot – Sepal Length vs Sepal Width

This plot visualizes the relationship between sepal length and sepal width for the three Iris flower species (setosa, versicolor, virginica). It helps to understand the distribution of data before training the SVM classifier.

## Feature-target split:

X = df.drop(['target', 'target_name'], axis=1)
y = df['target']


X contains all features (sepal length, sepal width, petal length, petal width).

y is the target class label (0 = setosa, 1 = versicolor, 2 = virginica).

## Train-test split:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


80% of data is used for training, 20% for testing.

Ensures model evaluation on unseen data.

## SVM model creation:

model = SVC(kernel='linear')


kernel='linear' → uses a straight-line hyperplane for classification.

## Model training:

model.fit(X_train, y_train)


The SVM algorithm learns the optimal hyperplane to separate the classes.

## Making predictions:

y_pred = model.predict(X_test)


Predicts class labels for the test set.

## Model evaluation:

print(accuracy_score(y_test, y_pred))


Calculates the accuracy: proportion of correctly classified test samples.

Gives an overall performance metric for the SVM classifier.

## Purpose:

Train a linear SVM to classify Iris flowers.

Evaluate performance using accuracy to ensure the model generalizes well to unseen data.
