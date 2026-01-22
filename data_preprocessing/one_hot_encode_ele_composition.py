"""
Course: Personal Programming Project(PPP) - Winter semester 2023
Supervisor: Dr. Arun Prakash
Author: Harish Somaghatta
Date: 07 Jan 2024
Description: (i) One-hot encoding based on number of elements and 
             (ii) Element Composition Vector - Contribution of each element to the compound.

"""

# Import the read_get_ele_count function from the 'weighted_atomic_properties' module
from Desc2_weighted_atomic_properties import read_get_ele_count

def one_hot_encode(ele_count_list):
    """
    Encode based on the number of elements in each compound formula.

    Arguments:
    ele_count (list): List of dictionaries containing keys as elements and values as their count.

    Returns:
    binary_vector_list (list): List of lists containing the elements number in encoded form.
    """
    # Initialize a list to store encoded list
    binary_vector_list = []
    
    # Get the maximum number of elements from the element and their count list.
    max_n_ele = max_num_ele(ele_count_list)
        
    # Iterate over all the compound elements and their count.
    for ele_count_dict in ele_count_list:
        
        # Get the number of elements in the current compound formula
        num_ele = len(ele_count_dict)
        # Initialize a binary vector with zeros
        binary_vector = [0] * max_n_ele
        # Based on number of elements, set the respective index to 1.
        binary_vector[num_ele - 1] = 1
        binary_vector_list.append(binary_vector) 
    
    # Return encoded list of number of elements in the compound formula.
    return binary_vector_list


def max_num_ele(ele_count_list):
    """
    Finds the maximum number of elements in the element-count list.

    Arguments:
    ele_count_list (list): A list of dictionaries containing elements and their counts.

    Returns:
    max_num_ele (int): The maximum number of elements in the dictionary of element-count list.
    """
    # Initialize to compute the maximum number of elements in each compound formula.
    max_num_ele = 0
    
    # Iterate over all the compound elements and their count.
    for ele_count_dict in ele_count_list:
        
        # Get the number of elements in the current compound formula 
        num_ele = len(ele_count_dict)
        # Consider as maximum number of elements if the current compound has more number of elements.
        if num_ele > max_num_ele:
            max_num_ele = num_ele
            
    return max_num_ele

def ele_composition_vector(ele_count_list):
    """
    Calculate Element composition vector for each compound formulas.
    
    Arguments:
    ele_count_list (list): A List of dictionaries containing keys as elements and values as their count.
    
    Returns:
    ele_comp_list (list): List of lists containing the element composotion vector for each compound formula.
    
    """
    # Initialize a list to store element composotion vector for each compound formula.
    ele_comp_list = []
    # Get the maximum number of elements from the element and their count list.
    max_n_ele = max_num_ele(ele_count_list)
    
    # Iterate over all the compound elements and their count.
    for ele_count_dict in ele_count_list:
        
        # Calculate the sum of count values of all elements in the current compound.
        sum_count = sum(ele_count_dict.values())
        # Initialize element composition vector with zeros.
        ele_composition_v = [0] * max_n_ele
        # Calculate the element composition vector for each compound and append to the list.
        for ind, count in enumerate(ele_count_dict.values()):
            
            ele_fra = count/sum_count
            ele_composition_v[ind] = ele_fra   
        ele_comp_list.append(ele_composition_v)
    
    return ele_comp_list, max_n_ele
            
def write_element_comp_encoded_n_ele(file, binary_vec_list, ele_comp_vector_list, max_num_ele, output_file):
    """
    Write encoded number of elements and element composition vectors to an output file.

    Arguments:
    file (str): Path of an input file containing headers.
    binary_vec_list (list): List of lists containing encoded number of elements.
    ele_comp_vector_list (list): List of lists containing element composition vectors.
    max_num_ele (int): Maximum number of elements in the given compounds file.
    output_file (str): Path of an output file to write the encoded values.

    Returns:
    None
    """
    
    # Open the file in read mode
    with open(file, "r") as inp_file:
        # Initialize a list to store the header of encoded values based on number of elements
        encod_header_list = []
        # Initialize a list to store the header of molar ratio values 
        ele_comp_header_list = []
        lines = inp_file.readlines()
        # Split the header line to list
        header = lines[0].strip().split(",")
        
    # Iterate over (max_num_ele) for encoded number of elements and element composition vector
    for i in range(max_num_ele ):
        # Header for encoded number of elements
        encode_header = f"{i + 1}_num_ele"
        encod_header_list.append(encode_header)
        # Header for molar ratios
        ele_composition_header = f"element_composition_{i + 1}"
        ele_comp_header_list.append(ele_composition_header)
        
        # Modified header after including encoded number of elements and molar ratios
        modified_header = header[:2] + encod_header_list + ele_comp_header_list + header[2:]
    
    # Open the file in write mode and write encoded number of elements and molar ratios
    with open(output_file, "w") as out_file:
        # Write the modified header to the file
        out_file.write(",".join(modified_header) + "\n")
        
        # Iterate over the each line of the file after header line
        for i, line in enumerate(lines[1:]):
            # Split each line into a list
            line_list = line.strip().split(",")
            # Insert the new columns after 2nd column
            before_encod = line_list[:2]
            after_encod = line_list[2:]
            
            # Use the encoded values list and molar ratio list
            encoded_val = binary_vec_list[i]
            ele_comp_vector = ele_comp_vector_list[i]
            
            # Convert the encoded values and element composition values into string in order to write to the file
            num_to_str = [str(val) for val in encoded_val]
            ele_comp_str = [str(ele_comp) for ele_comp in ele_comp_vector]

            
            # Write the modified lines to the file
            out_file.write(",".join(before_encod + num_to_str + ele_comp_str + after_encod) + "\n")

def main_encode_ele_comp(compounds_to_ele_count_f, label_encode_f, output_file):
    """
    Main function to encode the number of elements in a compound and element composition of compounds and write the encoded to a file.

    Arguments:
    compounds_to_ele_count_f (str): Path of an input file containing only compounds data.
    label_encode_f (str): Path of an input file containing label encoded material type.
    output_file (str): Path of an output file to write the encoded values.

    Returns:
    None
    """
    # Read the file and get list of dictionary containing keys as elements and values as their count.     
    ele_count_list_ = read_get_ele_count(compounds_to_ele_count_f)
    
    # Get the list containing encoded number of elements.
    binary_vector_list = one_hot_encode(ele_count_list_)
    
    # Get the list containing element composition vector and maximum number of elements of all compound formulas.
    element_comp_vector_list, max_num_e = ele_composition_vector(ele_count_list_)
    
    # Write the one-hot encoded vector based on number of elements and element composition vector to the file.
    write_element_comp_encoded_n_ele(label_encode_f, binary_vector_list, element_comp_vector_list, max_num_e, output_file)


if __name__ == "__main__":
    main_encode_ele_comp("compounds_to_ele_count.csv", "DPP3_label_encode.csv", "DPP4_ele_composition_num_ele.csv")

