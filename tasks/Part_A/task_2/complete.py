

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
            # Replace empty cells with None

            fixed_row = [None if cell == '' else cell for cell in row]
            data_2d_list.append(fixed_row)
    
    return data_2d_list # Return the resulting 2D list representing the dataset 

### Init
original_file_path = 'tasks/Part_A/task_1/cleaned.csv'
dataset=load_to_memory(original_file_path=original_file_path)
#print(dataset[1][0])
header=dataset[0]
entries=dataset[1:]
### 


### check for double entries
#print(header)
#print(len(entries))
my_set=set()
for entry in entries:
    my_set.add(entry[0])

#print(len(my_set))
#
    
### find entries with missing values and those who are full
full_entries=[]
cnt=0
for entry in entries:
    #print(entry)
    if None in entry:
        cnt+=1
    else:
        full_entries.append(entry[1:])
#print(cnt) we find that out of 540 the 271 entries have missing fields 
        
#we will use half the entries (that don't have missing fields) to train a predictor (one for each column) to apply on entries with missing fields
# Function to convert each cell to float
def cast_to_float_2d(string_list_2d):
    float_list_2d = []
    for row in string_list_2d:
        float_row = [float(cell) for cell in row]
        float_list_2d.append(float_row)
    return float_list_2d

# cast to float
full_entries= cast_to_float_2d(full_entries)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

dataset_array = np.array(full_entries)
#print(dataset_array.shape)

prediction_column_index=27 #(starting with 0)

# Split the dataset into features (X) and target variable (y)
all_columns_except_given  = np.concatenate((dataset_array[:, :prediction_column_index], dataset_array[:, prediction_column_index+1:]), axis=1)
X = all_columns_except_given # Features (all columns except the last one)
y = dataset_array[:, prediction_column_index]   # Target variable (last column)
#print(X.shape)
#exit()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Standardize the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a simple neural network model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))  # Output layer with 1 neuron for regression task

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train_scaled, y_train, epochs=1000, batch_size=32, validation_split=0.1)

# Evaluate the model
mse = model.evaluate(X_test_scaled, y_test)
print(f'Mean Squared Error: {mse}')



test=np.array([[3102,1,1,83,0,0.0,1,0,1,1,1,11.7,1,7.25,2.88,0,0,0,0,0.0,0.0,24.67105263,29.5,98,40.185,10,26,0,2,27,4,4.8,1,7,1,1.0,5.0,20,0.0,5.0,1,1,13,8.5,3,2,4.7,3,1,10.5,6,31,4,1,4]])
print(test[0][prediction_column_index+1])
all_columns_except_given  = np.concatenate((dataset_array[:, :prediction_column_index], dataset_array[:, prediction_column_index+1:]), axis=1)
# Now you can use the trained model to make predictions for new data
new_data = np.array(all_columns_except_given)  # Replace this with your new input data
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)
print(f'Predicted value: {prediction[0][0]}')