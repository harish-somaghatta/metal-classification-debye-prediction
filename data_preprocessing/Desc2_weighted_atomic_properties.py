
"""
Course: Personal Programming Project(PPP) - Winter semester 2023
Supervisor: Dr. Arun Prakash
Author: Harish Somaghatta
Date: 14 Dec 2023
Description: Converts atomic properties to weighted atomic properties

"""

import numpy as np

# Calculate mean
def calculate_mean(prop_values):
    
    """
    Compute mean of a list of atomic property values of compound elements

    Arguments:
    prop_values (List): List of atomic property values
        
    Returns:
    mean (Float): Mean of the atomic property values
    """ 
    return sum(prop_values)/len(prop_values)

# Calculate weighted mean
def calculate_weighted_mean(ele_fraction, prop_values):
    """
    Compute weighted mean of list of atomic property values of compound elements

    Arguments:
    prop_values (List): List of atomic property values
    ele_fraction (List): List of element fraction values
        
    Returns:
    weighted_mean (Float): Weighted mean of the atomic property values
    """
    # Initialize the weighted mean
    weighted_mean = 0
    
    # Iterate through the list of element fraction values
    for ele_prop, prop_ in zip(ele_fraction, prop_values):
        weighted_mean += (ele_prop * prop_)/sum(ele_fraction)
        
    return weighted_mean
    

#Calculate Geometric mean
def calculate_geometric_mean(prop_values):
    """
    Compute geometric mean of list of atomic property values of compound elements

    Arguments:
    prop_values (List): List of atomic property values
        
    Returns:
    geometric_mean (Float): Geometric mean of the atomic property values
    """
    geometric_mean = np.prod(prop_values) ** (1/len(prop_values))
    return geometric_mean

# Calculate Weighted Geometric Mean 
def calculate_weighted_geometric_mean(prop_values, ele_fraction):
    """
    Compute weighted geometric mean of list of atomic property values of compound elements

    Arguments:
    prop_values (List): List of atomic property values
    ele_fraction (List): List of element fraction values
    
    Returns:
    geometric_mean (Float): Geometric mean of the atomic property values
    """
    # Initialize the weighted geometric mean
    weighted_geometric_mean = 1
    
    # Iterate through the list of element fraction values
    for ele_prop, prop_ in zip(ele_fraction, prop_values):
        weighted_geometric_mean *=   (prop_ ** ele_prop)
    weighted_geometric_mean = weighted_geometric_mean ** (1/sum(ele_fraction))
    
    return weighted_geometric_mean

# Calculate entropy
def calculate_entropy(prop_fraction):
    """
    Compute entropy of list of atomic fraction values of compound elements

    Arguments:
    prop_fraction (List): List of atomic fraction values
    
    Returns:
    entopy (Float): Entropy of the atomic fraction values
    """
    #Initialize the entropy
    entropy = 0
    
    # Iterate through the atomic property fraction values
    for prop_fra in prop_fraction:
        entropy += (prop_fra * np.log(prop_fra))
        
    return -entropy

# Calculate Weighted entropy
def calculate_weighted_entropy(weight_fraction):
    """
    Compute weighted entropy of list of weighted atomic fraction values of compound elements

    Arguments:
    weight_fraction (List): List of weighted atomic fraction values
    
    Returns:
    weighted_entropy (Float): Weighted entropy of the weighted atomic fraction values
    """
    # Initialize the weighted entropy
    weighted_entropy = 0
    
    # Iterate through the weighted atomic property fraction values
    for weight_fra in weight_fraction:
        weighted_entropy += (weight_fra * np.log(weight_fra))
        
    return -weighted_entropy 

# Calculate Range

def calculate_range(prop_values):
    """
    Compute range of list of atomic property values of compound elements

    Arguments:
    prop_values (List): List of atomic property values
    
    Returns:
    range_ (Float): Range of the atomic property values
    
    """
    
    range_ = max(prop_values) - min(prop_values)

    return range_

# Calculate Weighted Range
def calculate_weighted_range(ele_fraction, prop_values):
    """
    Compute weighted range of list of atomic property values of compound elements

    Arguments:
    prop_values (List): List of atomic property values
    ele_fraction (List): List of element fraction values
        
    Returns:
    weighted_range (Float): Weighted range of the atomic property values
    """
    # Initialize weighted range list
    weighted_range_list = []
    
    # Iterate through the element fraction/atomic property values
    for ele_fra, prop_ in zip(ele_fraction, prop_values):
        weighted_range = (ele_fra * prop_)
        #Append the weighted range values to be used to find miximum or minimum of these values
        weighted_range_list.append(weighted_range)
    weighted_range = max(weighted_range_list) - min (weighted_range_list)
    
    return weighted_range

# Calculate Standard Deviation
def calculate_std_deviation(prop_values, mean ):
    """
    Compute standard deviation of list of atomic property values of compound elements

    Arguments:
    prop_values (List): List of atomic property values
    mean (Float): Mean of atomic property values
        
    Returns:
    std_deviation (Float): Standard deviation of the atomic property values
    """
    # Initialize the standard deviation 
    std_deviation = 0
    
    # Iterate through the atomic property values
    for val in prop_values:
        std_deviation += ((val - mean) ** 2)
    std_deviation = np.sqrt((std_deviation) / len(prop_values))
    
    return std_deviation

# Calculate Weighted Standard Deviation
def calculate_weighted_std_deviation(ele_fraction, prop_values, weighted_mean):
    """
    Compute weighted standard deviation of list of atomic property values of compound elements

    Arguments:
    ele_fraction (List): List of element fraction values
    prop_values (List): List of atomic property values
    weighted_mean (Float): Weighted mean of atomic property values
        
    Returns:
    std_deviation (Float): Weighted standard deviation of the atomic property values
    """
    # Initialize the weighted standard deviation 
    w_std_dev = 0
    
    # Iterate through the atomic property / element fraction values
    for val, ele_fra in zip(prop_values, ele_fraction):
        w_std_dev += ((val * ele_fra) - weighted_mean)**2
    w_std_dev = np.sqrt((w_std_dev)/sum(ele_fraction))
    
    return w_std_dev

def read_get_ele_count(file_name):
    """
    Reads a file containing element-count pairs, converts each line into an element-count dictionary,
    and returns a list of dictionaries with keys as elements and values as count.

    Arguments:
    file_name (str): Path of the file containing element-count pair.

    Returns:
    ele_count_list (List): A list of dictionaries with keys as elements and values as count.
    """
    # Initialize a list to store element and their count dictionaries
    ele_count_list = []

    # Open the file in read mode
    with open(file_name, 'r') as f:
        lines = f.readlines()

        # Iterate over each line in the file
        for each_line in lines[1:]:
            # Initialize a dictionary to store element-count pair
            ele_count_dict = {}

            # Split the line into a list of element-count pairs
            line_list = each_line.strip().split()

            # Iterate over the list of element-count pairs
            for ele_pair in line_list:
                # Split the pair into element and count
                ele, count = ele_pair.strip().split(":")
                # Append the pair to the dictionary
                ele_count_dict[ele] = int(count)

            # Append the element and their count dictionary to the list
            ele_count_list.append(ele_count_dict)
    
    # Retuen a list of dictionaries with keys as elements and values as count. 
    return ele_count_list

def read_updated_atm_prop(updated_ele_file):
    """
    Reads a file containing updated atomic properties, and returns the header and atomic properties.

    Arguments:
    updated_ele_file (str): Path of the file containing updated atomic properties data.

    Returns:
    """
    
    # Open the file in read mode.
    with open(updated_ele_file, "r") as ele_file:
        
        atomic_properties = []
        lines = ele_file.readlines()
        
        # Iterate over each line of the file
        for line in lines:
            # Split each line into a list
            line_list = line.strip().split(",")
            atomic_properties.append(line_list)
    
    return atomic_properties

# Calculate weighted atomic properties for each atomic property of a compound
def weighted_properties(prop, element_count, atom_properties):
    
    """
    Computes weighted atomic properties for each atomic property of a compound.

    Arguments:
    prop (String): The property for which weighted atomic properties are carried on.
    element_count (Dictionary): A dictionary with keys as elements and values as their counts in a compound.
    atom_properties (list): List of atomic properties for each element.
        
    Returns:
    f"{prop}_weights (Dictionary): Weighted atomic properties for the given atomic properties
    """
    
    #Initialize lists
    prop_list = []              #Initialize list to store property values of an element
    values_list = []            #Initialize list to store count values of an element
    ele_fraction_list = []      #Initialize list to store fraction of elements in a compound
    prop_frac_list = []         #Initialize list to store fraction of atomic property in a compound
    weighted_frac_d_list = []   #Initialize list to store weighted fraction denominator of a compound
    weighted_frac_list = []     #Initialize list to store weighted fraction values of a compound
    
    
    # Loop through the element and their count of each compound
    for key, value in element_count.items():
        
        # Get the index of the symbol in the header
        symbol_index = atom_properties[0].index("Symbol")
        
        for item in atom_properties[1:]:
            if item[symbol_index] == key:
                # Get the index of the property in the header
                prop_index = atom_properties[0].index(prop) 
                prop_value = item[prop_index]
                break
        prop_list.append(float(prop_value))    # Append the atomic property values to prop_list
        values_list.append(value)           # Append the count values to values_list
        
    # Calculate the element fraction of a compound
    for value in values_list:
        
        ele_fraction = value / sum(values_list)
        ele_fraction_list.append(ele_fraction)

    # Calculate the atomic property fraction
    for prop_val in prop_list:
        
        prop_frac = prop_val / sum(prop_list)
        prop_frac_list.append(prop_frac)
    
    # Calculate the denominatior of weighted atomic property fraction
    for ele_fra, prop_fra in zip(ele_fraction_list, prop_frac_list):
        
        weighted_frac_d = ele_fra * prop_fra
        weighted_frac_d_list.append(weighted_frac_d)

    # Calculate the weighted atomic property fraction
    for ele_fra, prop_fra in zip(ele_fraction_list, prop_frac_list):
        
        weighted_frac = ele_fra * prop_fra
        weighted_frac = weighted_frac / sum(weighted_frac_d_list)
        weighted_frac_list.append(weighted_frac)
        
    # Calculate Mean
    mean = calculate_mean(prop_list)

    # Calculate Weighted mean
    weighted_mean = calculate_weighted_mean(ele_fraction_list, prop_list)

    # Calculate Geometric Mean
    geometric_mean = calculate_geometric_mean(prop_list)

    # Calculate Weighted Geometric Mean
    weighted_geometric_mean = calculate_weighted_geometric_mean(prop_list, ele_fraction_list)

    # Calculate entropy
    entropy = calculate_entropy(prop_frac_list)

    # Calculate Weighted entropy
    weighted_entropy = calculate_weighted_entropy(weighted_frac_list)

    # Calculate Range
    range_ = calculate_range(prop_list)
    #print(f"{prop} Range:", range_)

    # Calculate Weighted Range
    weighted_range = calculate_weighted_range(prop_list, ele_fraction_list)

    # Calculate Standard Deviation
    std_deviation = calculate_std_deviation(prop_list, mean)

    # Calculate Weighted Standard Deviation
    w_std_dev = calculate_weighted_std_deviation(ele_fraction_list, prop_list, weighted_mean)
    
    # Return a dictionary of weighted atomic properties     
    return {
        f"{prop}_Mean": mean,
        f"{prop}_Weighted_Mean": weighted_mean,
        f"{prop}_Geometric_Mean": geometric_mean,
        f"{prop}_Weighted_Geometric_Mean": weighted_geometric_mean,
        f"{prop}_Entropy": entropy,
        f"{prop}_Weighted_Entropy": weighted_entropy,
        f"{prop}_Range": range_,
        f"{prop}_Weighted_Range": weighted_range,
        f"{prop}_Standard_Deviation": std_deviation,
        f"{prop}_Weighted_Standard_Deviation": w_std_dev,
    }

def write_atm_prop(prop_list, all_prop_res, raw_compounds_file, output_file):
    """
    Writes atomic properties into a new CSV file along with compounds data.
    
    Arguments:
    prop_list (list): List of numeric atomic properties.
    all_prop_res (list): List of dictionaries containing weighted atomic properties for each compound.
        
    Returns:
    None
    
    """
    
    # Open the compounds data file in read mode.
    with open(raw_compounds_file, "r") as input_file:
        
        # Read all the lines in the file.
        raw_lines = input_file.readlines()
        # Get the header from the first line and split into list.
        raw_header = raw_lines[0].strip().split(",")
        
        # Open new file to write the weighted compounds data including the existing raw data.
        with open(output_file, "w") as output_file:
            
            # Get the list of weighted atomic properties header.
            atm_prop_header = [key for prop_dict in all_prop_res for key in prop_dict.keys()]
            # Create a new header combining the original header and additional weighted atomic properties.
            modified_header = raw_header[:-2] + atm_prop_header + raw_header[-2:]
            # Write the modified heater to the new file.
            output_file.write(",".join(modified_header) + "\n")
            
            # Iterate over each line of the raw file excluding the header line.          
            for i in range(1, len(raw_lines)):
                # Initialize a list to store weighted atomic properties for current weighted atomic property.
                atm_line_val_list = []
                
                # Iterate over each numeric atomic property.
                for prop_i in range(len(prop_list)):
                    
                    # Iterate over each atomic property in the atomic properties dictionary.
                    for key in all_prop_res[prop_i].keys():
                        # Get the atomic property value for each compound.
                        atm_line_val = str(all_prop_res[prop_i][key][i-1])
                        # Append each property value to the list
                        atm_line_val_list.append(atm_line_val)
                    
                # Create each line by combining the raw compounds data line and compound weighted atomic properties. 
                modified_line = raw_lines[i].strip().split(",")[:-2] + atm_line_val_list + raw_lines[i].strip().split(",")[-2:]
                # Write the modified line to the new file
                output_file.write(",".join(modified_line) + "\n")
                
def main_weighted_prop(raw_compounds_file, compounds_data_file, elements_data_file, output_file):
    """
    Main function to calculate weighted atomic properties of compounds write to an output file.

    Arguments:
    raw_compounds_file (str): Path of an input file containing raw compounds data.
    compounds_data_file (str): Path of an input file containing only compound data.
    elements_data_file (str): Path of an input file containing elements and their count data.
    output_file (str): Path of an output file to write weighted atomic properties.

    Returns:
        None
    """
    
    ele_count_list_ = read_get_ele_count(compounds_data_file)
    atm_properties = read_updated_atm_prop(elements_data_file)
    
    # Consider only numeric atomic properties.
    prop_list = [prop for prop in atm_properties[0] if prop != "Symbol"]
    
    # Initialize a dictionary to store weighted atomic properties.
    weighted_property_output = {}
    
    # Initiaze a list to store dictionaries of weighted atomic properties.
    all_prop_res = []
    
    # Loop over the atomic properties 
    for prop in prop_list:
        
        # Initialize a dictionary to store the individual compound property result
        prop_results = {}
        
        
        # Iterate through each element and their count
        for ele_count_dict_ in ele_count_list_:
            
            # Compute the weighted atomic property for each property
            weighted_property_output = weighted_properties(prop, ele_count_dict_, atm_properties)
            
            # Iterates through each property and appends new values to the existing list without overwriting the old values
            for key, value in weighted_property_output.items():
                prop_results[key] = prop_results.get(key, []) + [value]
                
        all_prop_res.append(prop_results)
    # Write compound weighted atomic properties to the file.
    write_atm_prop(prop_list, all_prop_res, raw_compounds_file, output_file)

if __name__ == "__main__":
    main_weighted_prop("raw_compounds_data.csv", "compounds_to_ele_count.csv", "Elements_data_updated.csv", "DPP1_weighted_atomic_prop.csv")
