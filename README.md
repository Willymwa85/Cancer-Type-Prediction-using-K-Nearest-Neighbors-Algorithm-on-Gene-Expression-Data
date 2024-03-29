# Cancer-Type-Prediction-using-K-Nearest-Neighbors-Algorithm-on-Gene-Expression-Data
# Introduction

The provided code is an implementation of the K-Nearest Neighbors (KNN) algorithm for predicting cancer types based on gene expression data. The KNN algorithm is a supervised machine learning technique used for classification problems. It classifies new data points based on their similarity to the nearest neighbors in the training dataset. The code effectively trains a KNN model on gene expression data, evaluates its performance, and provides visualizations to analyze the model's behavior and data distribution.

# Methodology

 - Data Loading: The code reads gene expression data and corresponding cancer labels from CSV files using Pandas and NumPy libraries.
 - Data Preprocessing: The label names are converted to numerical values using the LabelEncoder from the scikit-learn library for ease of computation.
 - Data Splitting: The data is split into training and testing sets using the train_test_split function from scikit-learn.
 - Model Training: A KNN model is created and trained on the training data using the KNeighborsClassifier class from scikit-learn.
 - Model Evaluation: The trained model's accuracy is evaluated on the testing data using the accuracy_score metric from scikit-learn.

# Results and Analysis

The code reports an accuracy of 1.00 (100%) for the KNN model with the default value of k=5 (number of neighbors). It then explores the impact of different values of k on the model's accuracy, ranging from 1 to 20. The results show that the accuracy remains constant at 1.00 for all values of k.

Visualizations are provided to analyze the model's behavior and data distribution:

Accuracy vs. Number of Neighbors (k): A line plot shows the relationship between the number of neighbors (k) and the corresponding model accuracy.
Confusion Matrix: A heatmap visualizes the confusion matrix, displaying the true labels against the predicted labels.
Principal Component Analysis (PCA): A scatter plot visualizes the gene expression data projected onto the first two principal components, colored by cancer type.
Applicability

The code demonstrates the application of the KNN algorithm for cancer type prediction based on gene expression data. This approach can be useful in the field of bioinformatics and cancer research, where accurate classification of cancer types is crucial for diagnosis and treatment planning.

# Conclusion and Recommendation

The provided code successfully implements the KNN algorithm for cancer type prediction and achieves a perfect accuracy of 1.00 on the given dataset. However, it is important to note that the performance may vary on different datasets or in real-world scenarios. Further analysis and evaluation on diverse datasets would be recommended to assess the robustness and generalization capabilities of the model.

Additionally, exploring other machine learning algorithms or ensemble methods, and comparing their performance with KNN, could provide valuable insights and potentially improve the overall predictive accuracy.

# References

Scikit-learn documentation: https://scikit-learn.org/stable/
NumPy documentation: https://numpy.org/doc/
Pandas documentation: https://pandas.pydata.org/docs/
Matplotlib documentation: https://matplotlib.org/stable/contents.html
Seaborn documentation: https://seaborn.pydata.org/
