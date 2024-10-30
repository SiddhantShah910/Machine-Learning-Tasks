# Machine Learning Predictions

This repository contains two tasks demonstrating the application of supervised and unsupervised machine learning techniques using Python. 

## Table of Contents
- [Task 1: Prediction using Supervised ML](#task-1-prediction-using-supervised-ml)
- [Task 2: Prediction using Unsupervised ML](#task-2-prediction-using-unsupervised-ml)
- [Technologies Used](#technologies-used)
- [Conclusion](#conclusion)

## Task 1: Prediction using Supervised ML

In this task, we predict scores based on the number of hours studied using linear regression.

### Steps

1. **Import Libraries**:
    ```python
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    ```

2. **Load Dataset**:
    ```python
    df = pd.read_csv('http://bit.ly/w-data')
    df.head()
    ```

3. **Data Visualization**:
    ```python
    plt.scatter(x=df['Hours'], y=df['Scores'])
    plt.title('Relation Between Hours And Scores', fontdict={'weight' : 'bold', 'size' : 18})
    plt.xlabel('Hours')
    plt.ylabel('Scores')
    plt.show()
    ```
    <img src="https://github.com/user-attachments/assets/75583c20-1c9f-45b0-bb91-09310f8a7314" alt="download" width="500"/>

4. **Prepare Data for Training**:
    ```python
    X = df[['Hours']].values
    Y = df[['Scores']].values
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    ```

5. **Train the Model**:
    ```python
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(x_train, y_train)
    ```

6. **Make Predictions**:
    ```python
    y_predict = model.predict(x_test)
    ```

7. **Visualization of Predictions**:
    ```python
    l = (model.coef_ * X) + model.intercept_
    plt.scatter(X, Y)
    plt.plot(X, l, color='g')
    plt.title('Linear Regression Visualizer', fontdict={'weight' : 'bold', 'size' : 18})
    plt.xlabel('Hours')
    plt.ylabel('Scores')
    plt.show()
    ```
      <img src="https://github.com/user-attachments/assets/1671b4b5-592e-45e4-8a6a-631a5ea42286" alt="download (1)" width="500"/>

8. **Evaluate the Model**:
    ```python
    from sklearn.metrics import mean_squared_error
    print('Mean Squared Error : ', mean_squared_error(y_test, y_predict))
    ```
    <img src="https://github.com/user-attachments/assets/47ddfb4e-efdc-4c54-bab8-78a40bb91f9f" alt="download (2)" width="500"/>

### Output
The model predicts that if a student studies for 9.25 hours, the expected score is approximately 93.69.

---

## Task 2: Prediction using Unsupervised ML

In this task, we use the KMeans clustering algorithm to classify the Iris dataset.

### Steps

1. **Import Libraries**:
    ```python
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    ```

2. **Load Dataset**:
    ```python
    iris = pd.read_csv(r'/home/Siddhant/Downloads/Iris.csv')
    iris.head()
    ```

3. **Prepare Data for Clustering**:
    ```python
    x = iris.iloc[:, 1:-1].values
    ```

4. **Determine the Optimal Number of Clusters (Elbow Method)**:
    ```python
    from sklearn.cluster import KMeans
    sse = []
    for i in range(1, 10):
        km = KMeans(n_clusters=i)
        km.fit(x)
        sse.append(km.inertia_)
    ```

5. **Plot Elbow Graph**:
    ```python
    plt.plot(range(1, 10), sse)
    plt.xlabel('Number of Clusters')
    plt.ylabel('SSE')
    plt.title('Elbow Method', fontdict={'weight' : 'bold', 'size' : 18})
    plt.show()
    ```
    <img src="https://github.com/user-attachments/assets/6a7a515f-e1ba-49da-bdfd-1341656a9c9c" alt="download" width="500"/>

6. **Apply KMeans Clustering**:
    ```python
    km = KMeans(n_clusters=3)
    y_predicted = km.fit_predict(x) 
    iris['Cluster'] = y_predicted
    ```

7. **Visualize Clusters**:
    ```python
    plt.scatter(iris_1['SepalLengthCm'], iris_1['SepalWidthCm'], label='Iris-setosa', color='r')
    plt.scatter(iris_2['SepalLengthCm'], iris_2['SepalWidthCm'], label='Iris-versicolor', color='g')
    plt.scatter(iris_3['SepalLengthCm'], iris_3['SepalWidthCm'], label='Iris-virginica', color='b')
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:,1], label='Centroid', s=100, color='y')
    plt.legend()
    plt.show()
    ```
    <img src="https://github.com/user-attachments/assets/d60cf541-cb64-46d7-9fa1-ed1ef61b5e97" alt="download (1)" width="500"/>

### Output
The clustering algorithm successfully classified the Iris dataset into three species based on sepal and petal dimensions.

---

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Conclusion
This project demonstrates the fundamental concepts of supervised and unsupervised machine learning through practical examples. The use of linear regression for score prediction and KMeans for clustering provides a solid foundation for further exploration in machine learning.

For any questions or feedback, feel free to reach out!
