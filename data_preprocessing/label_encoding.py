
"""
Course: Personal Programming Project(PPP) - Winter semester 2023
Supervisor: Dr. Arun Prakash
Author: Harish Somaghatta
Date: 28 Dec 2023
Description: label encoding of material type(Metal - 1, Non-Metal - 0).

"""

def binary_encode(material_type_list):
    """
    Encode a list of material types into binary form.

    Arguments:
    material_type_list (list): List of material types to be encoded.

    Returns:
    list: List of binary encoded material type values.
    """
    
    # Initialize a dictionary to store unique material types and their corresponding encode values
    encode_dict = {}
    # Initialize a list to store enoded material type values
    binary_encode_list = []
    # Get the unique material types
    list_unique_mat_type = []
    for mat_type in material_type_list:
        if mat_type not in list_unique_mat_type:
            list_unique_mat_type.append(mat_type)
            
    # Iterate over unique material types
    for ind, mat_type in enumerate(list_unique_mat_type):
        encode_dict[mat_type] = ind
    
    # Iterate over the list of material types to be encoded
    for material_type in material_type_list:
        binary_encode_list.append(encode_dict[material_type]) # Append enoded material type values
    
    # Return list of binary encoded material type values
    return binary_encode_list
        
    
def read_write_material_type(file, output_file):
    """
    Read material types from the file, encode them and write in the binary encoded form.
    
    Arguments:
    file (str): Path of the CSV file containing compounds information.
    
    Returns:
    None
    """
    # Open the file in read mode.
    with open(file, "r") as f:
        
        # Initialize a list to store material type
        material_type_list = []
        lines = f.readlines()
        # Split the line into list and get the header information and index of material type column.
        header = lines[0].strip().split(",")
        material_type_index = header.index("material_type")
        
        # Iterate over each line of the file
        for line in lines[1:]:
           # Split each line into a list
           line_list = line.strip().split(",")
           # Append only the element symbols into the initialized list
           material_type_list.append(line_list[material_type_index])
        encoded_material_type = binary_encode(material_type_list)
        
    with open(output_file, "w") as output_file:
        
        # Write the header to the output file.
        output_file.write(",".join(header[1:]) + "\n")
        
        for line, en_mat_type in zip(lines[1:], encoded_material_type):
            
            # Convert each line into list.
            line_list = line.strip().split(",")
            
            line_list[material_type_index] = str(en_mat_type)
            
            # Write binary encoded material type to the file.
            output_file.write(",".join(line_list[1:]) + "\n")
    
def main_label_encode(ind_ele_col_file, output_file):
    """
    Main function to read material type information from a file and write it as label encoded material type.

    Arguments:
    ind_ele_col_file (str): Path of the input file containing individual element columns.
    output_file (str): Path of the output file to write the binary encoded material type.

    Returns:
        None
    """
    # Read the material type information from the file and write as binary encoded material type.
    read_write_material_type(ind_ele_col_file, output_file)


if __name__ == "__main__":
    main_label_encode("DPP2_ind_ele_col.csv", "DPP3_label_encode.csv")
    

