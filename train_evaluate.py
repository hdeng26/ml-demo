from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Print the confusion matrix in a nicer format
def print_confusion_matrix(conf_matrix, class_names):
    """Prints the confusion matrix as a table with class names for better readability."""
    df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
    print(df_cm)

# Generate confusion matrix plot
def plot_confusion_matrix(conf_matrix, class_names, filepath='confusion_matrix.png'):
    plt.figure(figsize=(10,7))
    sns.heatmap(conf_matrix, annot=True, fmt='d',
                xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig(filepath)

# Load Iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% testing

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Initialize KNN classifier
classifier = KNeighborsClassifier(n_neighbors=5)

# Train the model using the training sets
classifier.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = classifier.predict(X_test)

# Evaluating the Model
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the confusion matrix to a CSV file
pd.DataFrame(cm, index=iris.target_names, columns=iris.target_names).to_csv('confusion_matrix.csv', index=True)

print("Confusion matrix saved to 'confusion_matrix.csv'")

# Now use the function to print your confusion matrix
print_confusion_matrix(cm, iris.target_names)

# Now generate and save the confusion matrix plot
plot_confusion_matrix(cm, iris.target_names)
