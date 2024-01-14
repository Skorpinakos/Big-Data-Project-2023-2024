import csv 


def cast_to_float_2d(string_list_2d):
    float_list_2d = []
    for row in string_list_2d:

        float_row = [float(cell) for cell in row]
        float_list_2d.append(float_row)
    return float_list_2d

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



original_file_path_merged = 'tasks/Part_B/merged.csv'
dataset_merged=load_to_memory(original_file_path=original_file_path_merged)

# we will remove the frail column (index = 9) as it is the obtained label. the project description does not elaborate on the 5 columns that produce the frail index
temp=[]
for i in dataset_merged:
    temp.append(i[0:9]+i[9+1:])
dataset_merged=temp


header=dataset_merged[0]
entries=dataset_merged[1:]
print(header)

#we now cast to float
entries=cast_to_float_2d(entries)
print(entries[0])

