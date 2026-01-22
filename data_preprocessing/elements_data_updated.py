"""
Course: Personal Programming Project(PPP) - Winter semester 2023
Supervisor: Dr. Arun Prakash
Author: Harish Somaghatta
Date: 10 Dec 2023
Description: Fill the missing values of atomic properties with their respective mean values

"""

# Update the missing values of the atomic properties with the mean values and write to the file
def write_updated_mean(atm_prop_data, mean_prop, output_file):
    """
    Fill the missing values with the mean values of the atomic properties.

    Arguments:
    atm_prop_data (dict): Dictionary containing the header and list of atomic properties of each element.
    mean_prop (dict): Dictionary to store mean value of each atomic property.
    output_file (str): Path of output file containing non-missing values.
    
    Returns:
    None
    """
    # Get the header from the dictionary
    header = atm_prop_data["header"]
    
    # Open the file in write mode
    with open(output_file, "w") as file:
        seperator = ","
        # Write the header line to the file
        file.write(seperator.join(header) + "\n")
         
        # Iterate over each line of the file
        for line in atm_prop_data["lines"]:
    
            # Iterate over the column indices
            for i in range(len(line)):
                if line[i] == "":
                    header_name = header[i]
    
                    if header_name in mean_prop:
                        line[i] = str(mean_prop[header_name])
            
            # Write the lines after header to the file
            file.write(seperator.join(line) + "\n") 


def calc_mean(atm_property, atm_prop_mean, atm_prop_data):
    """
    Calculate the mean for the missing values of atomic property

    Arguments:
    atm_property(list)  : List containing the atomic property names which contain missing values.
    atm_prop_mean(dict) : Dictionary to store mean value of each property.
    atm_prop_data(dict) : Dictionary containing the header and list of atomic properties of each element.
     
    Returns:
    atm_prop_mean(dict) : Updated dictionary containing mean value of each property.

    """
    # Get the header from the dictionary
    header = atm_prop_data["header"]

    # Check if the property is in the atomic property list
    if atm_property in header:
       # Get the index of property
       ind = header.index(atm_property)
       # Initialize sum and count values
       sum_, count = 0, 0
       
       # Iterate over each element in the file
       for each_element in atm_prop_data["lines"]:   
           
           # Get the value of the property for the current element
           value = each_element[ind]
           
           # Check if there is any empty values and skip that 
           if value != "":
           
               # Calculate the sum of the property values
               sum_ += float(value) 
               count += 1
    
       # Calculate the mean of the property values
       total_mean = sum_/count
       # Store the mean values in dictionary
       atm_prop_mean[atm_property] = total_mean
    
    # Return the dictionary of atomic properties and their corresponding mean values
    return atm_prop_mean

def atm_prop(atm_prop_data):
       
    """
    Get the atomic property names which contains missing values.

    Arguments:
    atm_prop_data(dict) : Dictionary containing the header and list of atomic properties of each element.

    Returns:
    prop_list(list) : List containing the atomic property names which contain missing values

    """
    # Get the header from the dictionary
    header = atm_prop_data["header"]
        
    # Initilize a set to collect unique column indices 
    prop_set = set()
        
    # Iterate over the each line of the file 
    for line in atm_prop_data["lines"]:
        
        # Iterate over the column indices
        for i in range(len(line)):
                
            # Check if any property value is empty
            if line[i] == "":
                # Add the index of the property to the set 
                prop_set.add(header[i])
                
    # Convert set to list
    prop_list = list(prop_set)

    return prop_list

def read_store_prop(file):
        
    """
    Reads and stores the file data containing atomic properties of each element.

    Arguments:
    file(str) : Path of the csv file containing atomic properties.

    Returns:
    atm_prop_dict(dict) : Dictionary containing the header and list of atomic properties of each element

    """
    
    # Open the file in read mode
    with open(file, 'r') as file:
        
        # Initialize a dictionary to store the atomic properties
        atm_prop_dict = {}
        
        # Split the header line into list of atomic property names and store
        header = file.readline().strip().split(",")
        atm_prop_dict["header"] = header
        
        # Initialize a list to store the list of each element properties
        atm_prop_list = []
        
        # Iterate over each line in the file and append the list of atomic property values into entire properties list 
        atm_prop_list = [line.strip().split(",") for line in file]
        
        # Store the  overall list of atomic properties into the dictionary
        atm_prop_dict["lines"] = atm_prop_list
      
    # Return the dictionary containing header and atomic property values
    return atm_prop_dict

def main_ele_data_updated(input_file, output_file):
    """
    Gets the missing atomic properties information from an input file and writes the missing values filled with mean values to an output file. 

    Arguments:
    input_file (str): Path of an input file containing missing atomic property values.
    output_file (str) : Path of output file containing missing values filled with mean of availables property values.
    
    Returns:
    None
    """
    # Read and store the elements data into a dictionary
    atm_prop_dict = read_store_prop(input_file)
    
    # Get the list of atomic property names containing missing values
    atm_prop_list = atm_prop(atm_prop_dict)
    
    # Initialize a dictionary to store the atomic properties and their corresponding mean values
    atm_prop_mean = {}
    
    # Iterate over the each atomic property in the list
    for prop in atm_prop_list:
        atm_prop_mean_dict = calc_mean(prop, atm_prop_mean, atm_prop_dict)
    
    # Update the missing values with the mean values and write to the file
    write_updated_mean(atm_prop_dict, atm_prop_mean_dict, output_file)

if __name__ == "__main__":
    # Main funciton with input and output file paths.
    main_ele_data_updated("Elements_data.csv", "Elements_data_updated.csv")

"""
# Verification:
# ============   
test_prop_list = ["AtomicNumber", "AtomicRadius"]

test_atm_prop_mean = {}

test_atm_prop_dict = {'header': ['AtomicNumber', 'AtomicMass', 'AtomicRadius', 'Electronegativity'], 
                      'lines': [['4', '1.008', '20', '2.2'], 
                                ['3', '4.002', '10', ''], 
                                ['', '6.941', '', '0.98'], 
                                ['9', '9.012', '40', '1.57'], 
                                ['0', '10.81', '40', '2.04']]}


for test_property in test_prop_list:
    test_atm_prop_mean_dict = calc_mean(test_property, test_atm_prop_mean, test_atm_prop_dict)
    
print(test_atm_prop_mean_dict)

# Expected output:  {'AtomicNumber': 4.0, 'AtomicRadius': 27.5}
# ===============
"""