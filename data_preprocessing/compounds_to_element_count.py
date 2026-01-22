
"""
Course: Personal Programming Project(PPP) - Winter semester 2023
Title: Predicting Debye temperature in Metals using Machine Learning
Supervisor: Dr. Arun Prakash
Author: Harish Somaghatta
Date: 05 Dec 2023
Description: Get elements and their count from chemical formulas of compounds.

"""
# Read the compounds information from raw compounds data file
def read_compounds(file):
    
    """
    Reads only chemical formulas of compounds from the given csv file.

    Arguments:
    file(str) : Path of the csv file containing compound information

    Returns:
    compounds(List) : A list of compound formulas 

    """
    
    # Open the file in read format.
    with open(file, "r") as file:
        
        # Read all the lines of the file.
        data = file.readlines()
        # Get the index of the compounds column header.
        compounds_index = data[0].split(",").index("compound")
        
        
        # Initialize a list to store compound formulas.
        compounds = []
        
        # Iterate over each line of the file and extract only chemical formula of compounds.
        for line in data[1:]: # Skip the header
            compound = line.split(',')[compounds_index]
            compounds.append(compound)
    
    # Return only the compounds formulas list
    return compounds


def compounds_to_elements_count(compound):
    
    """
    Converts the chemical formula of a compound into dictionary of individual elements and their counts

    Arguments:
    compound (str) : Compound chemical formula

    Returns:
    elements_and_count (dict) : A dictionary with keys as elements and values as their counts of a compound.

    """
    
    # Initialize variables
    elements_and_count = {} # Dictionary to store elements and their counts
    element = ""
    count = 0
    
    i = 0
    # Loop through the each character in the compound formula
    while i<len(compound): 
        char = compound[i]
        
        # If the current character is uppercase which means a new element in the compound formula
        if char.isupper():
            
            # If there was already an element, update the element and count in the dictionary
            if element:
                elements_and_count[element] = count
            # Reset count to 1 and update the new character 
            element = char
            count = 1
           
        # If the current character is lowercase, append this character to the already available element character
        elif char.islower():
            element += char
        
        # Check if the current character is a digit
        elif char.isdigit():
            # Initialize a variable to store the count value
            num = char
            # Check for the next character by raising the index
            i += 1
            
            # If the next character is also a digit, then append the same
            while i < len(compound) and compound[i].isdigit():
                num += compound[i]
                i += 1
            # Convert string to integer and store as count
            count = int(num)
            # Change the index to the earlier which was raised above to check the next character
            i = i-1
            
        i += 1
    
    # If any element exist after the loop, update the element and count in the dictionary
    if element:
        elements_and_count[element] = count
    
    # Returns a dictionary with elements and their corresponding count values
    return elements_and_count


# Write elements and count of each compound into a file
def write_ele_count(element_count, output_file):
    
    """
    Writes elements and count of each compound formula into a csv file

    Arguments:
    element_count (list) : A list of dictionaries with keys as elements and values as their counts of a compound. 
    
    Returns:
    None
    
    """
    
    with open(output_file, "w") as file:
        # Write a header in a file
        file.write("element_count\n")
        
        # Iterate over compound formulas and, element and their count
        for ele_count in element_count:
            ele_count_str = " ".join(f"{ele}:{count}" for ele, count in ele_count.items())
            
            # Write element and their count of each compound
            file.write(f"{ele_count_str}\n")

def main_compounds_to_ele_count(input_file, output_file):
    """
    Gets the compound forumla information from an input file and extracts the element and their count, and writes to an output file. 

    Arguments:
    input_file (str): Path of an input file containing compound formulas information.
    output_file (str) : Path of output file containing elements and their count.
    
    Returns:
    None
    """
    # Get list of compound formulas
    compound_f = read_compounds(input_file)
    
    # Initialize a list to store the dictionary items with elements and their counts of all compounds
    element_count = []
    
    # Loop through all the compound formulas 
    for compound in compound_f:
        compounds_to_ele_count = compounds_to_elements_count(compound)
        element_count.append(compounds_to_ele_count)
    
    # Writes elements and their count into a file
    write_ele_count(element_count, output_file)

if __name__ == "__main__":
    # Main funciton with input and output file paths.
    main_compounds_to_ele_count("raw_compounds_data.csv", "compounds_to_ele_count.csv")

