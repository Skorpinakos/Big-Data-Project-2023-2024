
from cleanerANDnominaliser import non_num_fields,save_csv
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
            #print(len(row))
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
problematic_entries=[]
cnt=0
for entry in entries:

    if None in entry:
        cnt+=1
        problematic_entries.append(entry) 
    else:
        full_entries.append(entry)
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

def calculate_column_averages(dataset):
    num_columns = len(dataset[0]) if dataset else 0
    column_averages = [sum(col) / len(dataset) for col in zip(*dataset)] if num_columns > 0 else []

    return column_averages
avg_values=calculate_column_averages(full_entries)
#print(avg_values)


import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def train_neural_network(dataset, target_column_index, epochs=1000, batch_size=8):
    # Extract features and target variable
    features = np.array([row[:target_column_index] + row[target_column_index+1:] for row in dataset])
    target = np.array([row[target_column_index] for row in dataset])

    # Split the dataset into training and testing sets
    features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    features_train_scaled = scaler.fit_transform(features_train)
    features_test_scaled = scaler.transform(features_test)

    # Build a simple neural network model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(features_train_scaled.shape[1],)),
        tf.keras.layers.Dense(1)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(features_train_scaled, target_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)

    # Evaluate the model on the test set
    loss = model.evaluate(features_test_scaled, target_test, verbose=0)
    print(f'Test Loss: {loss}')

    return model, scaler

def predict_cell_value(model, scaler, row, target_column_index):
    # Extract features for prediction
    features = np.array(row[:target_column_index] + row[target_column_index+1:])

    # Standardize the features using the same scaler used during training
    features_scaled = scaler.transform([features])

    # Make the prediction
    prediction = model.predict(features_scaled)[0][0]
    if target_column_index in non_num_fields:
        return int(prediction)
    return prediction







def fix_rows(rows):
    fixed_rows=[]
    from collections import defaultdict

    models = defaultdict(lambda: None)

    for i,row in enumerate(rows):
        print("Fixing row ",i)
        #my_row=[1008,2,1,74,10,11.0,0,1,1,1,1,55.0,None,46.0,15.0,0,1,1,1,3.0,0.0,33.95555556,32.7,103,51.4172,11,17,0,0,25,5,3.6,0,3,1,1.0,7.0,630,0.0,0.0,1,1,0,7.4,2,3,5,2,0,0.0,5,25,8,2,2]
        none_indices = [index for index, value in enumerate(row) if value is None] #get indexes where cell is empty
        semi_fixed_row=row.copy()
        for index_i in none_indices: #we fill empty values with avg
            semi_fixed_row[index_i]=avg_values[index_i]

        for index_i in none_indices:
            prediction_column_index=index_i
            if models[index_i]==None:   
                print(f"model for column {index_i} is being created")             
                trained_model, scaler = train_neural_network(full_entries, target_column_index=prediction_column_index)
                models[index_i]=[trained_model,scaler]
            else:
                print(f"model for column {index_i} is cached") 



            # Now, you can use the trained model and scaler to make predictions:
            trained_model,scaler=models[index_i]
            predicted_value = predict_cell_value(trained_model, scaler, semi_fixed_row, target_column_index=prediction_column_index)
            semi_fixed_row[index_i]=predicted_value
        fixed_row=semi_fixed_row.copy()
        fixed_rows.append(fixed_row)
            #print(f'Predicted value: {predicted_value}, Ground Truth: {my_row[prediction_column_index]}')
    return fixed_rows
fixed=fix_rows(problematic_entries)
#for i in fixed:
    #print(i)

final_dataset=fixed+full_entries
final_dataset.sort(key=lambda x: x[0])


save_csv(header=header,data=final_dataset,filename="tasks/Part_A/task_1/cleaned_and_filled.csv")