# medical-hypertension-classification
## Task
The provided dataset is about a medical dataset on hypertension, where the target is the class labels (Positive and Negative) and the rest are the class features.
### The tasks required:
Data Analysis & Data Visualization (Plots and Graphs)

Data Loading and Splitting

Using Multiple Machine Learning Classification (SVM, KNNs, Random Forest, etc.)

Comparison of Performances of different ML classifiers. (Accuracy, Precision, Recall, etc.)

Performance Visualization as an output. (Confusion Matrices, Accuracy Vs. Epochs) 

## Data Handling 
Firstly, the data is loaded into a dataFrame.

These lines define where the data is coming from:

`sheet_id`: This is a unique identifier for a Google Sheet.

`gid`: This is a unique identifier for a specific sheet (tab) within the Google Sheet.

`data_url`: This line constructs a URL that points to the Google Sheet and specifies that the data should be exported in CSV (Comma Separated Values) format.

The `try` and `except` block is used for error handling. If any error occurs during the data loading process, the code inside the except block will be executed, printing an error message along with the specific error (e).

The next part, handling missing values, outliers, and inconsistencies. In this case, the data is numerical. The code searches for any columns with numerical data that have missing values. If it finds any, it calculates the average of the existing values in that column and then uses that average to fill in the missing spots. Afterward, it shows some information about the dataset to confirm that the missing values have been handled.

## Data Visualization

This section of Python code is primarily focused on visualizing the data using two popular Python libraries: `matplotlib` and `seaborn`.
### Creating A Heatmap
```
plt.figure(figsize=(2, 10))
sns.heatmap(df.corr()[['target']].sort_values(by='target', ascending=False), annot=True, cmap='coolwarm', fmt=".2f")
```
`df.corr()`: Calculates the correlation between all numerical columns in the DataFrame df.

`[['target']]`: Selects only the correlations with the 'target' column.

![image](https://github.com/user-attachments/assets/e0c71579-fdce-4609-a4ea-c15e6a97d34c)


### Creating Histograms
`num_cols = len(df.columns)`: Gets the number of columns in the DataFrame.

`num_rows = (num_cols + 2) // 3`: Calculates the number of rows needed for the subplots to accommodate all histograms.

`fig, axes = plt.subplots(...)`: Creates a figure and a grid of subplots.

`axes = axes.ravel()`: Flattens the axes array to make it easier to iterate through.

The for loop iterates through each column (col) of the DataFrame:

`ax = axes[i]`: Selects the current subplot.

`ax.hist(df[col].dropna(), bins=10)`: Creates a histogram of the data in the current column, dropping any missing values (dropna()) and using 10 bins.

In essence, this aims to provide a visual exploration of the data by showing the correlations between features and the target variable (using a heatmap) and the distribution of each feature (using histograms). These visualizations can help in understanding patterns, identifying potential issues, and gaining insights into the data.
![image](https://github.com/user-attachments/assets/ec0f033a-14e4-4f31-a5dd-44fb2a2a71e1)

## Data Transformation
This part of the code focuses on preparing the data for use in a machine learning model by scaling it to a standard range (0 to 1) using MinMaxScaler. .

Scale numerical features: Apply normalization (using `MinMaxScaler`) to bring numerical features to a similar scale to improve model performance.



_In case of the presence of categorical features_

Encode categorical features: Convert categorical features into numerical representations using label encoding.
```
# In case of categorical features:  encoding using Label Encoding
label_encoder = LabelEncoder()

for feature in categorical_features:
    try:
       df[feature] = label_encoder.fit_transform(df[feature])
    except TypeError:
        # Handle potential errors if some categorical values are not comparable
        df[feature] = df[feature].astype(str)  # Convert to string first if needed
        df[feature] = label_encoder.fit_transform(df[feature])
```

## Data splitting

Setting up the data in a way that allows training a machine learning model, to assess its performance during training (using the validation set), and finally evaluate its ability to make predictions on completely new data (using the test set). This process is fundamental to building robust and reliable machine learning models. I used an 80-20 split for both train/test and train/validate sets.

## Training the data

This part of the code systematically trains, evaluates, and compares the preformance of 3 implemented classification algorithms for supervised learning.  I import the necessary classes and functions from the `sklearn` library for creating and evaluating the models, as well as `numpy` for numerical operations.

A dictionary called `classifiers` is created to store the three classification models. Each key in the dictionary is the name of the model, and the value is an instance of the model class.
```
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(random_state=50),
    "KNN": KNeighborsClassifier(n_neighbors=1)
}
```

### The evaluation metrics used:

`accuracy`

`precision`

`recall`

`f1_score`

`confusion matrix`

### Visualizing the results:

The code prints the performance metrics for each classifier to the console.
It also generates visualizations: `Confusion Matrices`: Using `seaborn.heatmap`, it creates a visual representation of the confusion matrix for each model.

`Accuracy Comparison`: Using `matplotlib.pyplot.bar`, it creates a bar chart comparing the accuracy scores of the three models.

In this case, KNN classification has the best accuracy when validating the preformance of our classifiers.

![image](https://github.com/user-attachments/assets/23cd157f-4e2d-40b3-9972-837b8d96027e)

### Testing our model

Testing the K-Nearest Neighbors (KNN) machine learning model's performance on unseen data (the test data) and visualizing the results.

![image](https://github.com/user-attachments/assets/d2ebaf08-f53e-4da1-94c4-558faa57d7b1)

## Building and Training a 1D Convolutional Neural Network (CNN)

This part of the code focuses on creating, training, and evaluating a 1D CNN model using the TensorFlow library. CNNs are commonly used for tasks involving sequential data, like time series or in this case, where the data is reshaped to be sequential.

### It starts by importing necessary libraries from TensorFlow:

`tensorflow` for general TensorFlow functionalities.

`keras` for building and training neural networks.

`layers` for accessing different layer types used in building the CNN.

```
X_train = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
X_val = X_val.values.reshape(X_val.shape[0], X_val.shape[1], 1)
X_test = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)
```
The data `(X_train, X_val, X_test)` is reshaped to be suitable for a 1D CNN. The `reshape` function is used to add an extra dimension to the data, representing the single channel expected by the 1D convolutional layers.

A sequential model is created using `keras.Sequential()`, the model consists of the following layers:

`Conv1D`: A 1D convolutional layer with 64 filters, a kernel size of 3, and ReLU activation. It's the first layer and receives the input data.

`MaxPooling1D`: A max pooling layer that reduces the spatial dimension of the output from the convolutional layer.

`Flatten`: Flattens the output from the pooling layer into a single vector.

`Dense`: A fully connected layer with 100 units and ReLU activation.

`Dense`: The output layer with 1 unit and a sigmoid activation function, suitable for binary classification.

`EarlyStopping` is configured to monitor the validation loss (`val_loss`) and stop training if it doesn't improve for 5 epochs. This helps prevent overfitting.

The model is compiled, specifying the optimizer, loss function, and metrics to track during training:

`optimizer='adam'`: Uses the Adam optimizer for updating model weights.

`loss='binary_crossentropy'`: Uses binary cross-entropy as the loss function, suitable for binary classification.

`metrics=['accuracy']`: Tracks accuracy during training.

The trained model is evaluated on the test data (`X_test`, `y_test`) using evaluate, providing the test loss and accuracy.
Predictions are made on the test data using `predict`, and the predicted probabilities are converted to class labels (0 or 1) based on a threshold of 0.5.

Performance metrics (confusion matrix, precision, recall, F1-score) are calculated using functions from `sklearn.metrics`. The confusion matrix is visualized using a heatmap. The training history (accuracy and validation accuracy over epochs) is plotted to visualize the model's learning progress.
![image](https://github.com/user-attachments/assets/aa41a0c7-ada7-4b91-a8bc-e281d67e8e40)
![image](https://github.com/user-attachments/assets/1f676311-ccde-4a33-9908-a6da0cf3ca60)
