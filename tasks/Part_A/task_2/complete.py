

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
    
### find entries with missing values 
cnt=0
for entry in entries:
    #print(entry)
    if None in entry:
        cnt+=1
#print(cnt) we find that out of 540 the 271 entries have missing fields 
