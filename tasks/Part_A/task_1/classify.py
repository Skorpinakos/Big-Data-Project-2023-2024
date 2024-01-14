import csv
def load_to_memory(original_file_path):
    # Initialize an empty list to store the data
    data_2d_list = []

    # Open the CSV file
    with open(original_file_path, 'r', encoding="utf-8-sig", newline='') as csvfile:
        # Create a CSV reader object
        csv_reader = csv.reader(csvfile,delimiter=",")
        
        # Iterate over each row in the CSV file
        for row in csv_reader:
            # Append each row to the 2D list
            data_2d_list.append(row)

    
    return data_2d_list # Return the resulting 2D list representing the dataset 


### Init
original_file_path = 'tasks/Part_A/task_1/cleaned_and_filled.csv'
#original_file_path = 'tasks/Part_A/task_1/test.csv'
dataset=load_to_memory(original_file_path=original_file_path)
#print(dataset[1][0])

### 

#### we need to remove the 5 parameters used for establishing the fraility score and the score itself
#those are the columns : 1,9,10,16,17,18 #starting at 0, we also remove the id as it is irrelevnt
excluded_columns = [0,1, 9, 10, 16, 17, 18]


classifier_input=[]
classifier_output=[]
for entry in dataset[1:]:#exclude the header
    #print(entry)
    classifier_input.append([])
    for i,column in enumerate(entry):
        if i not in excluded_columns:
            classifier_input[-1].append(float(column))
    classifier_output.append(float(entry[1]))
        

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Assuming 'dataset' is your 2D list and 'desired_outputs' is your 1D list
# X is your feature matrix, and y is your target variable
X = classifier_input
y = classifier_output

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


acc=[]
for i in range(1,40):

    # Initialize the KNN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=i)  # You can adjust the number of neighbors (k) as needed

    # Train the classifier on the training data
    knn_classifier.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = knn_classifier.predict(X_test)

    # Evaluate the accuracy of the classifier
    accuracy = accuracy_score(y_test, predictions)
    acc.append(accuracy)
    #print(f'Accuracy: {accuracy * 100:.2f}% for {i} nearest neighboors')
print(sum(acc)/len(acc))
print(max(acc))

#now lets use a simple feed forward neural network 
def model_eval():
    import numpy as np
    import tensorflow as tf
    #tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from tensorflow.keras.utils import to_categorical
    import matplotlib.pyplot as plt

    # Assuming 'dataset' is your 2D list and 'desired_outputs' is your 1D list
    # X is your feature matrix, and y is your target variable
    features = classifier_input
    labels = classifier_output


    # Convert your data to NumPy arrays
    features = np.array(features)
    labels = np.array(labels)

    # Encode labels to integers
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Convert integers to one-hot encoding
    one_hot_labels = to_categorical(encoded_labels, num_classes=3)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, one_hot_labels, test_size=0.1, random_state=42)

    # Build a simple neural network model using TensorFlow for multi-class classification
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1024, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')  # Output layer with softmax for multi-class classification
    ])

    # Compile the model
    custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)


    model.compile(optimizer=custom_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


    # Store the training history
    history = model.fit(X_train, y_train, epochs=2000, batch_size=64, validation_data=(X_test, y_test),verbose=0)

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss}')
    print(f'Test Accuracy: {accuracy}')
    return accuracy

import concurrent.futures



def run_parallel(times): #run the model training multiple times to evaluate accuracy
    # Number of times to run the function
    num_runs = times

    # Create a ThreadPoolExecutor with the desired number of workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_runs) as executor:
        # Submit the function for execution in parallel
        futures = [executor.submit(model_eval) for _ in range(num_runs)]

        # Collect the results as they become available
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    return results

if __name__ == "__main__":
    parallel_results = run_parallel(16)
    print(parallel_results)
    print(sum(parallel_results)/16)

    #in the 8 runs
    #with predictor dataset 0.6388888955116272 [0.5925925970077515, 0.6851851940155029, 0.6666666865348816, 0.6111111044883728, 0.6111111044883728, 0.5925925970077515, 0.6666666865348816, 0.6851851940155029]
    #with simple averages   0.6296296268701553 [0.6296296119689941, 0.5740740895271301, 0.7222222089767456, 0.6111111044883728, 0.6111111044883728, 0.6481481194496155, 0.5740740895271301, 0.6666666865348816]

    #in the 16 runs
    #with predictor dataset 0.6435185223817825 [0.5925925970077515, 0.6851851940155029, 0.6666666865348816, 0.5555555820465088, 0.6111111044883728, 0.6481481194496155, 0.7222222089767456, 0.6666666865348816, 0.6851851940155029, 0.6851851940155029, 0.6296296119689941, 0.6851851940155029, 0.6666666865348816, 0.6111111044883728, 0.5555555820465088, 0.6296296119689941]
    #with simple averages   0.6238425895571709 [0.5925925970077515, 0.5370370149612427, 0.5925925970077515, 0.7222222089767456, 0.6851851940155029, 0.6111111044883728, 0.6111111044883728, 0.6111111044883728, 0.7037037014961243, 0.5925925970077515, 0.6851851940155029, 0.6111111044883728, 0.5925925970077515, 0.6296296119689941, 0.6296296119689941, 0.5740740895271301]
    


    #the knn gives the same result in both datasets
    