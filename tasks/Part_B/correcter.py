import csv 

import Levenshtein






def is_4digit_number(s):
    return s.isdigit() and len(s) == 4


def load_to_memory(original_file_path):
    # Initialize an empty list to store the data
    data_2d_list = []

    # Open the CSV file
    with open(original_file_path, 'r', encoding="utf-8-sig", newline='') as csvfile:
        # Create a CSV reader object
        csv_reader = csv.reader(csvfile,delimiter=";")
        
        # Iterate over each row in the CSV file
        for row in csv_reader:
            # Append each row to the 2D list
            data_2d_list.append(row)

    
    return data_2d_list # Return the resulting 2D list representing the dataset 


### Init
original_file_path = 'Project_Desc/beacons_dataset.csv'
#original_file_path = 'tasks/Part_A/task_1/test.csv'
dataset=load_to_memory(original_file_path=original_file_path)
#print(dataset[1][0])
header=dataset[0]
entries=dataset[1:]
### 

print(header)


### removing faulty lines
clean_entries=[]
for row in entries:
    if is_4digit_number(row[0]) and row[-1]!='':
        clean_entries.append(row)
    else:
        pass
        #print(row[0])
#print(len(clean_entries))
#print(len(entries))


participants={}
room_names=set()
for entry in clean_entries:
    try:
        participants[entry[0]].append(entry.copy())
    except:
        participants[entry[0]]=[]
        participants[entry[0]].append(entry.copy())
    room_names.add(entry[-1])


#print(room_names)

#for i in room_names:
    #print(f"'{i}':'',")


fix_dict={
    'Outdoor':'Outdoor',    
'Bathroim':'Bathroom',   
'Livingroon':'Livingroom', 
'Livingroom1':'Livingroom',
'TV':'Livingroom',
'Garage':'Outdoor',
'Bathroom1':'Bathroom',
'Office1':'Office',
'DinnerRoom':'Diningroom',
'Bathroom-1':'Bathroom',
'Sitingroom':'Livingroom',
'Hall':'Hall',
'Washroom':'Bathroom',
'Livingroom':'Livingroom',
'Garden':'Outdoor',
'Baghroom':'Bathroom',
'Kitchen':'Kitchen',
'Bathroom':'Bathroom',
'Liningroom':'Livingroom',
'Guard':'Hall',
'One':'Livingroom',
'DiningRoom':'Diningroom',
'Bedroom':'Bedroom',
'SittingOver':'Livingroom',
'Kichen':'Kitchen',
'Bsthroom':'Bathroom',
'LivingRoom2':'Livingroom',
'Bedroom2':'Bedroom',
'Kithen':'Kitchen',
'three':'Bathroom',
'bedroom':'Bedroom',
'2ndRoom':'Bedroom',
'LuvingRoom':'Livingroom',
'LivibgRoom':'Livingroom',
'Pantry':'Kitchen',
'Office1st':'Office',
'LeavingRoom':'Livingroom',
'Office-2':'Office',
'livingroom':'Livingroom',
'SittingRoom':'Livingroom',
'Four':'Outdoor',
'Kitcheb':'Kitchen',
'Veranda':'Outdoor',
'DinningRoom':'Diningroom',
'Office2':'Office',
'ExitHall':'Hall',
'Two':'Bedroom',
'Chambre':'Hall',
'Sittingroom':'Livingroom',
'Kitvhen':'Kitchen',
'Leavingroom':'Livingroom',
'Bathroon':'Bathroom',
'Livingroom2':'Livingroom',
'Kitcen':'Kitchen',
'Workroom':'Office',
'Entrance':'Hall',
'Laundry':'Bathroom',
'Office':'Office',
'Sittinroom':'Livingroom',
'Box-1':'Livingroom',
'Living':'Livingroom',
'Storage':'Outdoor',
'K':'Kitchen',
'Bedroom1st':'Bedroom',
'Library':'Office',
'DinerRoom':'Diningroom',
'Leavivinroom':'Livingroom',
'LivingRoom':'Livingroom',
'Desk':'Office',
'Kiychen':'Kitchen',
'Box':'Livingroom',
'SeatingRoom':'Livingroom',
'Sittigroom':'Livingroom',
'T':'Diningroom',
'Bqthroom':'Bathroom',
'Kitchen2':'Kitchen',
'Luvingroom1':'Livingroom',
'Bedroom1':'Bedroom',
'Bedroom-1':'Bedroom',
'Dinerroom':'Diningroom',
'LaundryRoom':'Bathroom',
'kitchen':'Kitchen'
}



from datetime import datetime

#### fix 
participants={}
room_names=set()
for entry_unfixed in clean_entries:
    entry=[0,0,0]
    entry[0]=entry_unfixed[0] ### leave id as is




    datetime_string=entry_unfixed[1]+entry_unfixed[2].replace(":","")  ### replace date and time fields to one unix time field
    # Convert the string to a datetime object
    dt_object = datetime.strptime(datetime_string, "%Y%m%d%H%M%S")

    # Get the Unix time (timestamp)
    unix_time = int(dt_object.timestamp())
    entry[1]=unix_time


    entry[-1]=fix_dict[entry_unfixed[-1]]   ### fix room names
    try:
        participants[entry[0]].append(entry.copy())
    except:
        participants[entry[0]]=[]
        participants[entry[0]].append(entry.copy())
    room_names.add(entry[-1])


print(room_names) ### ceck new room name set

for participant in sorted(list(participants.keys())): 

    participants[participant].sort(key=lambda x: x[1]) ### sort entries for each partiucipant based on time


