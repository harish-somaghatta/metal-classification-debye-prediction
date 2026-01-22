"""
Course: Personal Programming Project(PPP) - Winter semester 2023
Supervisor: Dr. Arun Prakash
Author: Harish Somaghatta
Date: 22 Dec 2023
Description: Create individual column for each element and update the number of atoms of a given 
element in the chemical formula 

"""
# Import the read_get_ele_count function from the 'weighted_atomic_properties' module
from Desc2_weighted_atomic_properties import read_get_ele_count

def read_updated_atm_prop(updated_ele_file):
    """
    Reads a file containing updated atomic properties and returns the element symbols.

    Arguments:
    updated_ele_file (str): Path of the file containing updated atomic properties data.

    Returns:
    symbols_list (List): A list of all the unique element symbols.
    """
    
    # Open the file in read mode.
    with open(updated_ele_file, "r") as ele_file:
        # Initialize a list to store element symbols.
        symbols_list = []
        
        lines = ele_file.readlines()
        # Split the header line into list and get the index of element symbols column.
        header_atm_prop = lines[0].strip().split(",")
        symbol_ind = header_atm_prop.index("Symbol")
        
        # Iterate over each line of the file
        for line in lines[1:]:
            # Split each line into a list
            line_list = line.strip().split(",")
            # Append only the element symbols into the initialized list
            symbols_list.append(line_list[symbol_ind])
    
    # Returns elemnets symbol list
    return symbols_list

def individual_ele_columns(element_count_list, symbols_list):
    """
    Create individual column for each element and the numerical value of each column for a given compound 
    should be equal to the number of atoms of the each element.
    
    Arguments:
    element_count_list (List): List of dictionaries containing keys as elements and values as counts for each compound.
    symbols_list (List): A list of all the unique element symbols.

    Returns:
    final_list (List): List of dictionaries containing number of atoms of each element for individual compound.
    
    """
    
    # Initialize a list to store final dictionary values
    final_list = []
    
    # Iterate over each dictionary in the list
    for ele_count_dict in element_count_list:
        
        # Initialize a dictionary to store columns of individual element
        ind_ele_dict = {}
        
        # Iterate over each element symbol
        for element in symbols_list:
            
            # Get the count of the element if the element symbol matches
            current_ele_count = ele_count_dict.get(element, 0)
            
            # Assign the count against the element to the dictionary
            ind_ele_dict[element] = current_ele_count
        
        # Append the each column of the elements for the individual compound to the final list
        final_list.append(ind_ele_dict)
    
    # Return the element columns for each compound.
    return final_list

def write_ind_col(file, ind_ele_col_list, output_file):
    """
    Writes individual comuln of each element along with the existing compounds columns
    
    Arguments:
    file (str): A file containing compounds data 
    ind_ele_col_list (List): List of dictionaries containing number of atoms of each element against individual compound.

    Returns:
    None    
    
    """
    
    # Open the file in read mode
    with open(file, "r") as compounds_file:
        
        # Read the lines of the compound data file
        compound_lines = compounds_file.readlines()
        # Split the header line into a list
        compound_header = compound_lines[0].strip().split(",")
        # Get the element symbols from element-count list
        ind_ele_header = list(ind_ele_col_list[0].keys())
        # Include the element symbol header columns inbetween the compound columns
        modified_header = compound_header[:-2] + ind_ele_header + compound_header[-2:]
        
        # Open the file in write mode
        with open(output_file, "w") as output_file:
            # Write compounds data header along with element symbol columns
            output_file.write(",".join(modified_header) + "\n")
            
            # Iterate over each element column and each line of compounds file
            for each_ele_col, each_comp_line in zip(ind_ele_col_list, compound_lines[1:]):
                # split the each line of the compounds data file into list
                comp_line_list = each_comp_line.strip().split(",")
                # Intialize a list to store element count for the each compound 
                each_comp_row_list = []
                
                # Iterate over each element column 
                for col in ind_ele_header:
                    
                    # Append the count of current element to the list
                    each_comp_row_list.append(str(each_ele_col[col]))
                # Write the each element as an individual column by combining the compounds data 
                output_file.write(",".join(comp_line_list[:-2] + each_comp_row_list + comp_line_list[-2:]) + "\n")

def main_des_2(compounds_to_ele_file, updated_ele_prop_file, weighted_atm_prop_file, output_file):
    """
    Main function to take weighted atomic properties file as input and add element count against each compound, and write to an output file.
    
    Arguments:
    compounds_to_ele_file (str): Path of an input file containing compounds to element count data.
    updated_ele_prop_file (str): Path of an input file containing updated element properties data.
    weighted_atm_prop_file (str): Path of an input file containing weighted atomic properties data.
    output_file (str): Path of an output file to write the individual element columns along with weighted atomic proerties.


    Returns:
    None    
    
    """
    # Get list of element-count dictionary from the file
    element_count_list = read_get_ele_count(compounds_to_ele_file)
    
    # Get the unique element symbols list 
    symbols_list = read_updated_atm_prop(updated_ele_prop_file)
    
    # Get the list of individual element column for each compound
    ind_col_list = individual_ele_columns(element_count_list, symbols_list)
    
    # Write individual comuln of each element along with the existing compounds columns
    write_ind_col(weighted_atm_prop_file, ind_col_list, output_file)

if __name__ == "__main__":
    main_des_2("compounds_to_ele_count.csv", "Elements_data_updated.csv", "DPP1_weighted_atomic_prop.csv", "DPP2_ind_ele_col.csv")

