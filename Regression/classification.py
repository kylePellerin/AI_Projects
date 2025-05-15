"""
Project 4 - CSCI 2400 - Classification

Name: Kyle Pellerin
"""

# Dataset generation
import numpy as np  
import pandas as pd
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
#X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=42)
  
# Dataset visualization
import matplotlib.pyplot as plt
#plt.scatter(X[:, 0], X[:, 1], c=y)
#plt.show()


def run_knn(X_train: np.array, y_train: np.array, X_test: np.array, k: int) -> np.array:
    """
    This runs a Knn on the data provided in X_train, y_train and xtest. With the x data being the points locations
    in the grid and y being the calssifier. The idea is to first standardize the data and then using a varitey of np
    commands, esaily manipualte it to find the nearest k points to the data point we are testing. The function then 
    outputs the arary of predictions whose accuracy, precision, and recall can be tested using the below functions. 
    """
    x_train_mean = X_train.mean(axis=0) #standardize the x data becuase y contians the classes for Knn
    x_train_std = X_train.std(axis=0)
    x_train_standardized = (X_train - x_train_mean) / x_train_std 
    x_test_standardized = (X_test - x_train_mean) / x_train_std
   
    output_predictions =[] #init an output array for the predictions
    
    for x in x_test_standardized:
        #print(x)
        euclidian_dsitances = np.sqrt(np.sum((x_train_standardized - x)**2, axis =1)) #get eculidian distnace of point 
        sorted_distances = np.argsort(euclidian_dsitances)
        k_closest = sorted_distances[:k] #get clossest sorted disntaces up to kth closest distance
        k_closest_vals = y_train[k_closest]
        k_output = np.bincount(k_closest_vals).argmax() #use np.bincount to get the most frequent niehgbor
        output_predictions.append(k_output)
    return np.array(output_predictions) #output the predication array as an np.array



def run_perceptron(X_train: np.array, y_train: np.array, X_test: np.array) -> np.array: 
    """
    DOC
    """
    x_train_mean = X_train.mean(axis=0) #standardize the x data becuase y contians the classes for Knn
    x_train_std = X_train.std(axis=0)
    x_train_standardized = (X_train - x_train_mean) / x_train_std 
    x_test_standardized = (X_test - x_train_mean) / x_train_std

    x_train = np.concatenate([np.ones((x_train_standardized.shape[0], 1)), x_train_standardized], axis=1) #add column of ones
    x_test = np.concatenate([np.ones((x_test_standardized.shape[0], 1)), x_test_standardized], axis=1)
    y_train = np.where(y_train==0, -1, 1) #have to change the zeros to -1s, keep the rest as 1s
    
    training_weights = np.zeros(x_train.shape[1])

    max_it = 500 #max number of iterations, change to check preformacnce 
    run_count =0 #count up to 500
    while run_count < max_it: #iterate over each weight max it times and check if its calssified correctly, if not change the weight matrix
        for i in range(len(x_train)):
            prediction = np.sign(np.dot(training_weights, x_train[i]))
            if prediction != y_train[i]:
                if prediction >= 0: #if were on the low side of the line
                    training_weights = training_weights - x_train[i]
                else: #if were above the line and need to get under it
                     training_weights = training_weights + x_train[i]
        run_count +=1

    prediction = np.sign(np.dot(x_test, training_weights))
    prediction = np.where(prediction<0, 0, 1) #change the -1s back to zeros 
    return prediction

def accuracy(y_true: list, y_pred: list) -> float:
    """
    calcualtes accuracy of a dataset = number of correctly calssified points / number of total points
    """
    correct_classification = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]: #check for each index if they are the same = correctly calssifed 
            correct_classification +=1
    return correct_classification/len(y_true)



def precision(y_true: list, y_pred: list) -> float:
    """
    Calcualtes percision of a dataset = number of true positives/ (Number of true psoitives + number of false positives) 
    """
    true_positive = 0
    false_positive = 0
    for i in range(len(y_true)):
        if y_pred[i] == 1:
            if y_true[i] == 1: #if it was calssified as true and is supposed to be
                true_positive +=1
            else:
                false_positive +=1 #if it was classified as false and supposed to be true 
    if (true_positive ==0 | false_positive ==0):
        return 0.0
    else:
        return (true_positive / (true_positive + false_positive))


def recall(y_true: list, y_pred: list) -> float:
    """
    Calculates the recall of a dataset = Number of true positives / (number of true positives +number of false negatives)
    """
    true_positive = 0
    false_negative = 0
    for i in range(len(y_true)):
        if y_true[i] ==1:
            if y_pred[i] ==1:
                true_positive +=1 #if it was supposed ot be true and calssified as true
            else:
                false_negative +=1 #if it was supposed to be true be true but classified as false
    
    if (true_positive ==0 | false_negative ==0):
        return 0.0     
    else:
        return (true_positive / (true_positive + false_negative))


from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target #1s are bad 0s are good


def run_comparisons() -> pd.DataFrame:
    """
    This takes the breast cancer data and runs it on boht algorhtims and outputs the results in a table 
    """
    test_size = 0.3 # 30% for testing
    split_index = int(len(X) * (1 - test_size)) # Calculate index for splitting

    X_train = np.array(X[:split_index]) # First 70% for training
    X_test = np.array(X[split_index:]) # Remaining 30% for testing
    y_train = np.array(y[:split_index]) 
    y_test = np.array(y[split_index:])

    k_value = 5 #change the k value
    knn_result = run_knn(X_train, y_train, X_test,k_value)
    perception_result = run_perceptron(X_train, y_train, X_test)
 
    #get the knn results
    k_acc = accuracy(y_test, knn_result)
    k_pre = precision(y_test, knn_result)
    k_re = recall(y_test,knn_result)

    #get the perceptron results 
    p_acc = accuracy(y_test, perception_result)
    p_pre = precision(y_test, perception_result)
    p_re = recall(y_test, perception_result)


    results = pd.DataFrame({ #create a dataframe with rows as the algorthim and columns with the results
        "accuracy": [k_acc, p_acc],
        "precision": [k_pre, p_pre],
        "recall": [k_re, p_re]
    }, index=["KNN", "Perceptron"])
    #print(results)
    return results

#calls the run comparisons function
run_comparisons()

"""
PRINT OUT ENABLED
#print(X)
test_size = 0.3 # 30% for testing
split_index = int(len(X) * (1 - test_size)) # Calculate index for splitting

X_train = np.array(X[:split_index]) # First 70% for training
X_test = np.array(X[split_index:]) # Remaining 30% for testing
y_train = np.array(y[:split_index]) 
y_test = np.array(y[split_index:])
print("test")
print(y_test)
test_knn_output =(run_knn(X_train, y_train, X_test, k=5)) 
print(run_knn(X_train, y_train, X_test, k=5))
print("accuracy")
print(accuracy(y_test, test_knn_output))
print("precision")
print(precision(y_test, test_knn_output))
print("recall")
print(recall(y_test, test_knn_output))

print("run Perceptron")
test_perceptron = (run_perceptron(X_train, y_train, X_test))
print(test_perceptron)
print("accuracy")
print(accuracy(y_test, test_perceptron))
print("precision")
print(precision(y_test, test_perceptron))
print("recall")
print(recall(y_test, test_perceptron))
    """

