import csv
import matplotlib.pyplot as plt


def plot_distribution(numbers,title):
    plt.hist(numbers, bins=50, color='blue', edgecolor='black', alpha=0.7)
    plt.title('Distribution '+title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


def is_number(s):
    try:
        complex(s) # for int, long, float and complex
    except ValueError:
        return False

    return True

def has_non_digit_elements(str_list):
    for element in str_list:
        if not is_number(element) and element.strip()!="":
            return True
    return False

def gather_field_ranges(head,entries):
    sets=[set() for _ in range(len(head))]
    for entry in entries:
        for i,value in enumerate(entry):
            sets[i].add(value)
    return sets #list of sets containing possible values for each field

def check_csv_structure(header,entries):
    #print(len(header))
    for i,entry in enumerate(entries):
        #print(len(entry))
        if len(entry)!=len(header):
            print(i,entry)
            return "error on entry: "+str(i)
        
    return True

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
original_file_path = 'Project_Desc/clinical_dataset.csv'
#original_file_path = 'tasks/Part_A/task_1/test.csv'
dataset=load_to_memory(original_file_path=original_file_path)
#print(dataset[1][0])
header=dataset[0]
entries=dataset[1:]
### 

### Diagnostics
structure_test_result=check_csv_structure(header=header,entries=entries)
if (structure_test_result!=True):
    print(structure_test_result)
    exit(1)
###
    
### Nominalise
    
# find ranges for fields
sets=gather_field_ranges(head=header,entries=entries)

#find which fields are non numerical
non_num_fields=[]
for i,range_of_field in enumerate(sets):
    if has_non_digit_elements(range_of_field):
        #print(header[i],range_of_field)
        #print(i,header[i])
        non_num_fields.append(i) #list of indexes corresponding to columns

#print(non_num_fields)

#create nominalisation dicts

nom_dicts=[ #list indexes correspond to columns
    {"Frail":2,"Non frail":0,"Pre-frail":1},
    {"M":-1,"F":+1},
    {"No":0,"Yes":1},
    {'Sees well':0, 'Sees moderately':1, 'Sees poorly':2},
    {'Hears poorly':2, 'Hears well':0, 'Hears moderately':1},
    {"No":0,"Yes":1},
    {'<5 sec':0,'>5 sec':1},
    {'FALSE':0, 'TRUE':1},
    {"No":0,"Yes":1},
    {"No":0,"Yes":1},
    {"No":0,"Yes":1},
    {"No":0,"Yes":1},
    {'Permanent sleep problem':2, 'Occasional sleep problem':1, 'No sleep problem':0},
    {"No":0,"Yes":1},
    {"No":0,"Yes":1},
    {"No":0,"Yes":1},
    {"No":0,"Yes":1},
    {'2 - Bad':1, '4 - Good':3, '5 - Excellent':4, '3 - Medium':2, '1 - Very bad':0},
    {'2 - A little worse':1, '3 - About the same':2, '4 - A little better':3, '5 - A lot better':4, '1 - A lot worse':0},
    {'No':0, '< 2 h per week':1, '> 5 h per week':3, '> 2 h and < 5 h per week':2},
    {'Never smoked':0, 'Current smoker':2, 'Past smoker (stopped at least 6 months)':1},
]

#replace values
new_entries=[]
for entry in entries:
    new_entries.append([])
    for i,field in enumerate(entry):
        if i in non_num_fields:
            try:
                new_entries[-1].append(nom_dicts[non_num_fields.index(i)][field]) #if existing value in dict then translate it according to dict
            except:
                new_entries[-1].append(None) #if existing value not in dict replace with None
        else:
            new_entries[-1].append(field) #if no nominalisation needed just add old value

###new dataset nominalised
            
### Remove eronious values in columns other than those listed in non_num_fields which have been taken care of through the above dicts
            
#get distributions
fields_distributions=[[] for _ in range(len(header))]
for entry in new_entries:
    for i,field in enumerate(entry):
        if i not in non_num_fields:
            try:
                fields_distributions[i].append(float(field))
            except:
                #fields_distributions[i].append(None)
                #print(field)
                #ignore non float/int values from the visualisation 
                pass
            
            
# we plot distributions to visualise outliers and produce thresholds for eronious values
for i,distr in enumerate(fields_distributions):
    
    if i not in non_num_fields:
        title=header[i]
        #print(i)
        #plot_distribution(distr,title)

# from visualy inspecting the distributions plotted we establish the following thresholds and comments, the thresholds are provided by examining the biggest non-outlying entry other than the outlying ones
# the added "x" in front indicates that we will be capping the values because although outliers they do signify something, no "x" means we will be leaving them empty since they are clear mistakes and the errors do not carry value
thresholds=[
    None,
    None,
    None, ###but there are propably real but extreme outliers, maybe need to add log scale
    "x<30", ### maybe 999 represents living in hospital, 2 entries
    None,
    "x<100", ###maybe the 999 represents inability to lift, 40+ entries are not mistakes
    "x<100", ###maybe the 999 represents inability to get up, ~9 entries
    "x<50",  ###maybe the 999 represents inability to run, ~6 entries
    "x<30",  ###the 999 is propably just a mistake or inability to move in general, ~3 entries
    "<30",  ###the 999 is propably just wrong/ clearly eronius
    "<60",  ###the ~900 entry is physically impossible so clearly eronius
    "<100", ###the 999 entry is clearly eronius
    None,   ###there is one outlier but it is propably real
    ">10",   ###the ~ -400 entry is clearly eronius
    None,
    None,
    None,
    None,
    None,
    None,
    "<50",  ###don't know how to interpet those, propably eronius but 4 entries
    "<200", ###don't know how to interpet those, propably eronius 2 entries
    None,
    "x<200", ###maybe not eronius , it may signify something 10+ entries. there are also outliers
    "x<200", ###maybe not eronius , it may signify something 10+ entries. there are also outliers
    None,
    None,
    None,
    "x<100", ###maybe grandad is alcoholic
    None,
    None,
    None,
    None,
    None
]



#print(len(header)-len(non_num_fields))
#print(len(thresholds))


#cap or remove weird values and keep the rest as they are
def clean_eronius_values(thresholds,non_num_fields,old_entries):
    new_entries=[]
    for entry in old_entries:
        new_entries.append([])
        thres_index=-1
        for i,field in enumerate(entry):
            if i not in non_num_fields:
                try:
                    test=float(field)
                except:
                    new_entries[-1].append(None) #if non numerical value then set it to None and pass
                    continue
                
                thres_index+=1
                threshold=thresholds[thres_index]
                if threshold!= None:
                    if "x" in threshold:
                        threshold=threshold.replace("x","")
                        #mode="cap"
                        if "<" in threshold:
                            threshold=float(threshold.replace("<",""))
                            new_entries[-1].append(min(float(field),threshold))
                        elif ">" in threshold:
                            threshold=float(threshold.replace(">",""))
                            new_entries[-1].append(max(float(field),threshold))
                    elif "x" not in threshold:
                        #mode="remove"
                        if "<" in threshold:
                            threshold=float(threshold.replace("<",""))
                            new_entries[-1].append(float(field) if float(field)<threshold else threshold)
                        elif ">" in threshold:
                            threshold=float(threshold.replace(">",""))
                            new_entries[-1].append(float(field) if float(field)>threshold else threshold)


            else:
                new_entries[-1].append(field) #if clearing is not needed just add old value
    return new_entries

new_entries=clean_eronius_values(thresholds,non_num_fields,new_entries)
### entries cleaned and nominalised, filling empty spots is all it is left


# checking results of the first 2 steps
empty_counter=0
for entry in new_entries:

    for field in entry:
        if str(field).strip()=="":
            print('problem')
        if field==None:
            empty_counter+=1
#print(empty_counter) #we count 742 empty spots out of 29700
#print(len(new_entries)*len(header))