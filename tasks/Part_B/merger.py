import csv 



def is_4digit_number(s):
    return s.isdigit() and len(s) == 4


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
original_file_path_clinical = 'tasks/Part_A/task_1/cleaned_and_filled.csv'
original_file_path_beacons = 'tasks/Part_B/new_dataset.csv'
dataset_clinical=load_to_memory(original_file_path=original_file_path_clinical)
dataset_beacons=load_to_memory(original_file_path=original_file_path_beacons)
#print(dataset[1][0])
header_clinical=dataset_clinical[0]
entries_clinical=dataset_clinical[1:]
header_beacons=dataset_beacons[0]
entries_beacons=dataset_beacons[1:]
### 

#print(header_beacons)
#print(header_clinical)


new_header=header_beacons+header_clinical[1:]
#print(new_header)

merged_datasets=[]

def find_entry(element_to_find,my_list):
    index_i = next((i for i, sublist in enumerate(my_list) if sublist[0] == element_to_find), None)
    return index_i

#print(find_entry('3546',dataset_beacons))


for row in entries_beacons:
    beacons_index=row[0]
    #print(beacons_index)
    clinical_index=find_entry(beacons_index,entries_clinical)
    if clinical_index!=None:
        #print(row,entries_clinical[clinical_index])
        merged_datasets.append(row+entries_clinical[clinical_index][1:])





def save_csv(header, data, filename):
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Write the header
        csv_writer.writerow(header)
        
        # Write the data, replacing None with an empty string
        for row in data:
            csv_writer.writerow(['' if cell is None else cell for cell in row])

save_csv(header=new_header,data=merged_datasets,filename="tasks/Part_B/merged.csv")