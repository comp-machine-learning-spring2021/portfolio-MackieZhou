# CSC 294 Final Portfolio - Mackie Zhou

## Me and My Portfolio

I am a sophomore majoring in Computer Science and Statistical and Data Sciences at Smith College. Welcome to my final portfolio for CSC 294 Computational Machine Learning!

This semester, I got exposed to the amazing world of Machine Learning. Throughout the semester, I have learned clustering, prediction, classification, ensemble methods, efficacy, and neural networks. I am now (more) familiar with numpy, pandas, scikit learn, matplotlib.pyplot, seaborn and some other python libraries. Besides, algorithms and coding skills, I have learned and discussed fairness and ethical issues in machine learning with peers in class and have written a summary essay of a conference paper on the topic of machine learning fairness. This is the the most interesting, challenging, and rewarding class I took in the spring of 2021.

In this final portfolio, I will focus on applications of machine learning in business and complete two unsupervised machine learning projects and one supervised machine learning project on three different datasets from Kaggle.




## Coding Projects

### 1. Unsupervised ML: High Value Customers Identification

**Keywords:**
unsupervised machine learning, clustering, KMeans, elbowology, efficacy comparison

**Self-defined Functions:** `recompute_centers`, `my_kmeans`, `calculate_within_cluster_sse`, `looping_kmeans_sklearn`, `looping_kmeans_my`

This is a revision of hw2.

The dataset used in this project is the [UK-High value Customers Identification](https://www.kaggle.com/vik2012kvs/high-value-customers-identification) dataset. In this coding project, I will use k-means to segment customers into clusters and identify the group of high-value customers--customers who are loyal to the brand or make high purchases--for a UK-based online retail store. In this project, I will use KMeans to segment customers into clusters. I will use elbowology to choose the best number of clusters and will compare efficacy of my own "cold" implementation of KMeans and the KMeans algorithm from sklearn.




### 2. Customer Segmentation on the Credit Card Dataset (Unsupervised)

**Keywords:** unsupervised machine learning, clustering, hierarchical clustering, spectral clustering, KMeans

**Self-defined Functions:** `replace_null_mean`

The dataset used in this project is [Credit Card Dataset for Clustering](https://www.kaggle.com/arjunbhasin2013/ccdata).

Credit card users also have different behaviors. This credit card dataset contains usage behavior of about 9000 active credit card holders in last 6 months1. The task here is to segment credit card users into clusters based on their different behaviors. In this project, I will try different strategies to segment the credit holders and find the model that makes the most sense. I will try hierarchical clustering, spectral clustering, and KMeans.




### 3. Supervised ML: Company Bankruptcy Prediction

**Keywords:**
supervised machine learning, classification, ensemble method, random forest, cross-validation

**Self-defined Functions:** `split_val_data`, `random_forest_cross_val`

The dataset used in this project is the [Company Bankruptcy Prediction](https://www.kaggle.com/fedesoriano/company-bankruptcy-prediction) dataset, which is a Bankruptcy data from the Taiwan Economic Journal for the years 1999–2009. My goal is to train a random forest model to predict whether a business will go bankrupt. I will use PCA to reduce the dimension of the dataset. I will use cross-validation to choose the best hyperparameters for the random forest model and then fit a model with the "best" set of hyperparameters.

### Picture of unit tests passing for all 3 projects

<img src="tests_passing.png" alt="preview of the dataset" width="900"/>