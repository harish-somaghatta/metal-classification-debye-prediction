"""
Course: Personal Programming Project(PPP) - Winter semester 2023
Supervisor: Dr. Arun Prakash
Author: Harish Somaghatta
Date: 17 Jan 2024
Description: Perform unit test for all the functions used in data preprocessing.

"""
# Import necessary libraries
import pytest
import os
import sys
Dpp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','..', 'data_preprocessing'))
sys.path.insert(0, Dpp_dir)
# Import necessary functions to perform unit test.
from compounds_to_element_count import compounds_to_elements_count
from elements_data_updated import calc_mean
from Desc2_weighted_atomic_properties import weighted_properties, read_updated_atm_prop
from Desc1_comp_to_ind_ele import individual_ele_columns
from label_encoding import binary_encode
from one_hot_encode_ele_composition import one_hot_encode, ele_composition_vector
from ordinal_encod import encode_space_group, encode_thermal_conductivity

#==========================================================================================================

def test_compounds_to_elements_count():
    '''
    
    # Purpose of the test: Test the function compounds_to_elements_count to makes sure that it extracts individual elements and their counts from chemical compound formulas correctly.

    # Input: Chemical formulas of compound.[H2So4, M]

    Command to run file: pytest test_data_preprocessing_unit_tests.py
    
    # Expected output: Dictionaries of individual elements as keys and their counts as values for each chemical compound formulas.
        exp_t1 = {'H': 2, 'So': 4}
        exp_t2 = {'M': 1}
    # Obtained output: Dictionaries of individual elements as keys and their counts as values for each chemical compound formulas.
            t1 = {'H': 2, 'So': 4}
            t2 = {'M': 1}
    '''

    t1 = compounds_to_elements_count("H2So4")
    t2 = compounds_to_elements_count("M")
    exp_t1 = {'H': 2, 'So': 4}
    exp_t2 = {'M': 1}
    
    assert(t1 == exp_t1)
    assert(t2 == exp_t2)

#==========================================================================================================

def test_mean_atm_prop():
    '''
    # Purpose: Test the function calc_mean to makes sure that it calculates the mean of atomic properties correctly which fills the missing values.

    # Input:  Atomic propertiers with missing values of Atomic number and Atomic radius.
        test_prop_list = ["AtomicNumber", "AtomicRadius"]    
        test_atm_prop_dict = {'header': ['AtomicNumber', 'AtomicMass', 'AtomicRadius', 'Electronegativity'], 
                              'lines': [['4', '1.008', '20', '2.2'], 
                                        ['3', '4.002', '10', ''], 
                                        ['', '6.941', '', '0.98'], 
                                        ['9', '9.012', '40', '1.57'], 
                                        ['0', '10.81', '40', '2.04']]}
    Command to run file: pytest test_data_preprocessing_unit_tests.py
    
    # Expected Output: AtomicNumber mean: 4.0, AtomicRadius mean: 27.5

    # Obtained Output: AtomicNumber mean: 4.0, AtomicRadius mean: 27.5

    '''
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
        
    expected_output = {'AtomicNumber': 4.0, 'AtomicRadius': 27.5}
    #print(test_atm_prop_mean_dict)
    
    assert(test_atm_prop_mean_dict == expected_output)

#==========================================================================================================

def test_weighted_prop():
    '''
    
    # Purpose: Test the function weighted_properties to make sure weighted atomic properties are correctly calculated.

    # Input: Atomic number, Atomic mass properties with test element and their count dictionary.
            test_prop = ['AtomicNumber', 'AtomicMass']
            test_ele_count_dict = {'S': 2, 'Mn': 4}
    
    Command to run file: pytest test_data_preprocessing_unit_tests.py
    
    # Expected Output: List of dictionaries containing weighted atomic properties for atomic number and atomic mass of given element.
            exp_t = [{'AtomicNumber_Mean': 20.5, 'AtomicNumber_Weighted_Mean': 21.99, 'AtomicNumber_Geometric_Mean': 20.0, 'AtomicNumber_Weighted_Geometric_Mean': 21.5443, 'AtomicNumber_Entropy': 0.66, 'AtomicNumber_Weighted_Entropy': 0.553, 'AtomicNumber_Range': 9.0, 'AtomicNumber_Weighted_Range': 11.3333, 'AtomicNumber_Standard_Deviation': 4.5, 'AtomicNumber_Weighted_Standard_Deviation': 17.4992}, 
                     {'AtomicMass_Mean': 43.499, 'AtomicMass_Weighted_Mean': 47.312, 'AtomicMass_Geometric_Mean': 41.96799113610276, 'AtomicMass_Weighted_Geometric_Mean': 45.909, 'AtomicMass_Entropy': 0.658, 'AtomicMass_Weighted_Entropy': 0.5342, 'AtomicMass_Range': 22.878, 'AtomicMass_Weighted_Range': 25.938, 'AtomicMass_Standard_Deviation': 11.439, 'AtomicMass_Weighted_Standard_Deviation': 38.152}]
            
    # Obtained Output: List of dictionaries containing weighted atomic properties for atomic number and atomic mass of given element. 
                The difference between expected output and obtained output is with in a tolerance of 1e-2
         
    '''
    test_prop = ['AtomicNumber', 'AtomicMass']
    test_ele_count_dict = {'S': 2, 'Mn': 4}
    atm_properties = read_updated_atm_prop(os.path.join(Dpp_dir, "Elements_data_updated.csv"))
    exp_t = [{'AtomicNumber_Mean': 20.5, 'AtomicNumber_Weighted_Mean': 21.999999999999996, 'AtomicNumber_Geometric_Mean': 20.0, 'AtomicNumber_Weighted_Geometric_Mean': 21.544346900318835, 'AtomicNumber_Entropy': 0.6688570623740269, 'AtomicNumber_Weighted_Entropy': 0.5538582294924286, 'AtomicNumber_Range': 9.0, 'AtomicNumber_Weighted_Range': 11.333333333333332, 'AtomicNumber_Standard_Deviation': 4.5, 'AtomicNumber_Weighted_Standard_Deviation': 17.499206331208914}, 
             {'AtomicMass_Mean': 43.499, 'AtomicMass_Weighted_Mean': 47.312, 'AtomicMass_Geometric_Mean': 41.96799113610276, 'AtomicMass_Weighted_Geometric_Mean': 45.90955353826814, 'AtomicMass_Entropy': 0.6581601850866663, 'AtomicMass_Weighted_Entropy': 0.5342456024177491, 'AtomicMass_Range': 22.878, 'AtomicMass_Weighted_Range': 25.938666666666663, 'AtomicMass_Standard_Deviation': 11.439, 'AtomicMass_Weighted_Standard_Deviation': 38.15258688768328}]
    
    for i, test_prop in enumerate(test_prop):
        test_weighted_prop = weighted_properties(test_prop, test_ele_count_dict, atm_properties)
        assert(exp_t[i] == test_weighted_prop)
#==========================================================================================================

def test_ind_col():
    '''
    # Purpose: Test the function individual_ele_columns to make sure that it creates individual element columns and put correspoding count for each compound correctly. 

    # Input: Elements and their count list, and element symbol list.
            test_element_count_list = [{"H":5, "He":2, "B":4},{"Be":2, "Li":9}]
            test_symbols_list = ["H", "He", "Li", "Be", "B", "C", "N", "O"]
            
    Command to run file: pytest test_data_preprocessing_unit_tests.py
    
    
    # Expected Output: List of dictionaries containing individual element columns and their corresponding count for each compound.
            exp_t = [{'H': 5, 'He': 2, 'Li': 0, 'Be': 0, 'B': 4, 'C': 0, 'N': 0, 'O': 0}, 
            {'H': 0, 'He': 0, 'Li': 9, 'Be': 2, 'B': 0, 'C': 0, 'N': 0, 'O': 0}]
    # Obtained Output: [{'H': 5, 'He': 2, 'Li': 0, 'Be': 0, 'B': 4, 'C': 0, 'N': 0, 'O': 0}, 
    {'H': 0, 'He': 0, 'Li': 9, 'Be': 2, 'B': 0, 'C': 0, 'N': 0, 'O': 0}]
    '''
    
    test_element_count_list = [{"H":5, "He":2, "B":4},{"Be":2, "Li":9}]
    test_symbols_list = ["H", "He", "Li", "Be", "B", "C", "N", "O"]
    exp_t = [{'H': 5, 'He': 2, 'Li': 0, 'Be': 0, 'B': 4, 'C': 0, 'N': 0, 'O': 0}, 
    {'H': 0, 'He': 0, 'Li': 9, 'Be': 2, 'B': 0, 'C': 0, 'N': 0, 'O': 0}]
    test_final_list = individual_ele_columns(test_element_count_list, test_symbols_list)
    assert(exp_t == test_final_list)
#==========================================================================================================


def test_material_type():
    '''
    # Purpose: Test the function binary_encode to make sure that it label encode the material types correctly.

    # Input: Material type list with Metal and Non-metal.
    test_material_type_list = ["Metal", "Metal", "Non-Metal", "Non-Metal", "Metal"]
    test_material_type = binary_encode(test_material_type_list)
    
    Command to run file: pytest test_data_preprocessing_unit_tests.py
    
    # Expected Output: List of label encoded material types - 0's and 1's.
        exp_t = [0, 1, 0, 0, 1]
    # Obtained Output: List of label encoded material types - 0's and 1's.
        t = [0, 1, 0, 0, 1]
    '''
    test_material_type_list = ["Non-Metal", "Metal", "Non-Metal", "Non-Metal", "Metal"]
    test_material_type = binary_encode(test_material_type_list)
    exp_t = [0, 1, 0, 0, 1]
    
    assert(test_material_type == exp_t)

test_material_type
#==========================================================================================================

def test_one_hot_encode_ele_comp():
    '''
    # Purpose: Test the functions one_hot_encode and ele_composition_vector to make sure that they compute one-hot encoding based on number of elements and element composition vector based on contribution of each element correctly and maximum number of elements in a compound formula correctly.

    # Input: List containing dictionaries of elements and their count. 
            test_compound_list = [{"Sn" : 4, "Mn" : 3, "H" : 21, "He":10}, {"Li" : 2, "O" : 8}, {"Sn" : 4, "Mn" : 3, "H" : 21}, {"Mn":1}]
    
    Command to run file: pytest test_data_preprocessing_unit_tests.py
    
    # Expected Output: List of one-hot encoded vectors, element composition vector and maximum number of elements in any compound.
            exp_t1 = [[0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]]
            exp_t2 = [[0.10526315789473684, 0.07894736842105263, 0.5526315789473685, 0.2631578947368421], [0.2, 0.8, 0, 0], [0.14285714285714285, 0.10714285714285714, 0.75, 0], [1.0, 0, 0, 0]]
            exp_t3 = 4
    # Obtained Output: 
            exp_t1 = [[0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]]
            exp_t2 = [[0.10526315789473684, 0.07894736842105263, 0.5526315789473685, 0.2631578947368421], [0.2, 0.8, 0, 0], [0.14285714285714285, 0.10714285714285714, 0.75, 0], [1.0, 0, 0, 0]]
            exp_t3 = 4
    '''
    test_compound_list = [{"Sn" : 4, "Mn" : 3, "H" : 21, "He":10}, {"Li" : 2, "O" : 8}, {"Sn" : 4, "Mn" : 3, "H" : 21}, {"Mn":1}]
    test_one_hot_encode = one_hot_encode(test_compound_list)
    exp_t1 = [[0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]]
    
    test_element_composition_vector_list, max_number_elements = ele_composition_vector(test_compound_list)
    exp_t2 = [[0.10526315789473684, 0.07894736842105263, 0.5526315789473685, 0.2631578947368421], [0.2, 0.8, 0, 0], [0.14285714285714285, 0.10714285714285714, 0.75, 0], [1.0, 0, 0, 0]]
    exp_t3 = 4
    
    assert(exp_t1 == test_one_hot_encode)
    assert(exp_t2 == test_element_composition_vector_list)
    assert(exp_t3 == max_number_elements)

#==========================================================================================================


def test_ordianl_label_encode():
    '''
    # Purpose: Test the functions encode_space_group and encode_thermal_conductivity to make sure that they compute label encoding of space group and ordinal encoding of thermal conductivity values correctly.
            test_list = [2, 100, 11, 121, 86, 22]
    # Input: List of space group values and thermal conductivity values.
            exp_t1 = ['0', '4', '1', '5', '3', '2']
    
    Command to run file: pytest test_data_preprocessing_unit_tests.py
    
    # Expected Output: List of label encoded space group and ordinal encoded thermal conductivity values.
            t1 = ['0', '4', '1', '5', '3', '2']
    # Obtained Output: List of label encoded space group and ordinal encoded thermal conductivity values.

    '''
    test_list = [2, 100, 11, 121, 86, 22]
    test_space_group_encod = encode_space_group(test_list)
    exp_t1 = ['0', '4', '1', '5', '3', '2']
    assert(exp_t1 == test_space_group_encod)
    test_k_list = [99, 25, 200, 54, 8]
    exp_t2 = [2, 2, 3, 2, 1]
    for i, k in enumerate(test_k_list):
        test_ordinal_encode = encode_thermal_conductivity(k)
        assert(exp_t2[i] == int(test_ordinal_encode))

#==========================================================================================================








