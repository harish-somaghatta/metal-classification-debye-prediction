
"""
Course: Personal Programming Project(PPP) - Winter semester 2023
Supervisor: Dr. Arun Prakash
Author: Harish Somaghatta
Date: 13 Jan 2024
Description: (i) Ordinal encoding of thermal conductivity values.
             (ii) Label encoding of space group number values.

"""

# Read space group values and sort the values
def read_write_space_group(file, output_f):
    
    """
    Read the space group values from the file and write the label encoded space group values.

    Arguments:
    file (str) : Path of the csv file containing compounds information.
    
    Returns:
    None
    
    """
    # Open the file in read mode
    with open(file, "r") as f:
        
        space_grp_list = [] # Initialize a list to store all space group values
        data = f.readlines() # Read all the lines of the file.
        header = data[0].strip().split(",") # Get the header details.
        space_grp_ind = header.index("spacegroup") # Get the index of the space group
        
        for line in data[1:]: # Iterate over all the lines after header line. 
            space_grp = line.strip().split(",")[space_grp_ind]
            space_grp_list.append(space_grp)
        # Get the list of label encoded space group values. 
        encoded_space_grp_list = encode_space_group(space_grp_list)
        
    with open(output_f, "w") as output_file:
        # Wrie the header to a file
        output_file.write(",".join(header) + "\n")
        
        # Iterate over the all the lines in the file 
        for i, line in enumerate(data[1:]):
            # Split each line in the file to a list
            list_line = line.strip().split(",")
            # Updating the space group value with the encoded space group value
            list_line[space_grp_ind] = encoded_space_grp_list[i]
            # Write the update space group values to the file
            output_file.write(",".join(list_line) + "\n")

def encode_space_group(list_space_group):
    
    """
    Get the list of space group values and return label encoded space group values.

    Arguments:
    space_group_list (list) : List of space group values.
    
    Returns:
    encode_space_group (list) : List of encoded space group values.
    
    """
    space_grp_dict = {} # Initialize a dictionary to store sorted space group values and their indices.
    encode_space_group = [] # Initialize a list to store encoded space group values.
    space_grp_set = set() # Initialize a set to store unique space group values.
    
    for space_grp in list_space_group: # Iterate over space group list values.
        space_grp_set.add(int(space_grp)) # Add each space group value to the set.
    unq_space_grp_list = list(space_grp_set) # Convert list to set to iterate.
    sort_space_grp = sorted(unq_space_grp_list) # Sort the space group values
    # Iterate over sorted unique space group values .
    for i, space_grp in enumerate(sort_space_grp):
        space_grp_dict[space_grp] = i # Append space group values against their index.
    
    # Iterate over list of space group values and append encoded space group values
    for space_group_val in list_space_group:
        encoded_space_group_val = space_grp_dict[int(space_group_val)]
        encode_space_group.append(str(encoded_space_group_val))

    # Return encoded space group values
    return encode_space_group

def encode_thermal_conductivity(thermal_cond_value):
   """
   Encode thermal conductivity value based on it's range.

   Arguments:
   thermal_cond_value (float): Thermal conductivity value to be encoded.

   Returns:
   encoded_value (str): Encoded thermal conductivity value.
   """
   thermanl_conductivty_range = ["low", "medium", "high"]
   
   encode_dict = {}
   
   for ind, k in enumerate(thermanl_conductivty_range):
       encode_dict[k] = str(ind + 1)
   
   
   if 0 < thermal_cond_value <= 10:
        encod_thermal = encode_dict["low"]
   elif 10 < thermal_cond_value <= 100:
        encod_thermal = encode_dict["medium"]
   elif thermal_cond_value > 100:
        encod_thermal = encode_dict["high"]
        
   return encod_thermal
    

def read_write_thermal_cond(output_file):
    
    """
    Write encoded thermal conductivity values to a file

    Arguments:
    file (str) : Path of the csv file containing compounds information
    
    Returns:
    None
    
    """
    # Read the file in read mode 
    with open(output_file, "r") as file:
        
        # Get header and thermal conductivity column index
        lines = file.readlines()
        header = lines[0].strip().split(",")
        thermal_cond_i = header.index("thermal_conductivity")
    
    # Open the file in write mode to include encoded thermal conductivity values
    with open(output_file, "w") as f:
        # Write the header to the file
        f.write(",".join(header) + "\n")
        
        # Iterate over the each line of the file past header line
        for line in lines[1:]:
            
            line_list = line.strip().split(",") # Split the line into list 
            # Encode based on the thermal conductivity range 
            line_list[thermal_cond_i] = encode_thermal_conductivity(float(line_list[thermal_cond_i]))
            f.write(",".join(line_list) + "\n") # Write the encoded values to the output file

def main_ordinal_label_enode(ele_composition_num_ele_f, output_file):
    """
    Main function to encode thermal conductivity values and space group values and, write to an output file.

    Arguments:
    ele_composition_num_ele_f (str): Path of the input file containing compounds data.
    output_file (str): Path of the output file to write the encoded values along with the vailable compounds data.

    Returns:
    None
    """
        
    # Read the space group values and write the label encoded space group values to the file.
    read_write_space_group(ele_composition_num_ele_f, output_file)
    
    # Read the thermal conductivity information from compounds data and write the ordinal encoded values.
    read_write_thermal_cond(output_file)

if __name__ == "__main__":
    main_ordinal_label_enode("DPP4_ele_composition_num_ele.csv","DPP5_label_ordinal_encode_file.csv")

