"""
Course: Personal Programming Project(PPP) - Winter semester 2023
Supervisor: Dr. Arun Prakash
Author: Harish Somaghatta
Date: 20 Jan 2024
Description: Perform integrated test for all the functions used in data preprocessing.

"""

# Import necessary libraries
import numpy as np
import pytest
import os
import sys
Dpp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data_preprocessing'))
sys.path.insert(0, Dpp_dir)

# Import necessary functions to perform integrated test.
from compounds_to_element_count import main_compounds_to_ele_count
from elements_data_updated import main_ele_data_updated
from Desc2_weighted_atomic_properties import main_weighted_prop
from Desc1_comp_to_ind_ele import main_des_2
from label_encoding import main_label_encode
from one_hot_encode_ele_composition import main_encode_ele_comp
from ordinal_encod import main_ordinal_label_enode


#==========================================================================================================


def test_comp_to_ele_count_integrated():
    '''
    # Purpose of the test: Test the main function main_compounds_to_ele_count to make sure that it extracts individual elements and their count from chemical formula of a compound correctly.

    # Input: File containing chemical compound formulas.
                (Mn, I22, Nh4K2, MnSo4I9, MNhSK64)
    
    Command to run file: pytest test_data_preprocessing_integrated.py
    
    # Expected output: File containing elements and their corresponding count of given compounds.
                ['Mn:1', 'I:22', 'Nh:4 K:2', 'Mn:1 So:4 I:9', 'M:1 Nh:1 S:1 K:64']
    # Obtained output: File containing elements and their corresponding count of given compounds.
                ['Mn:1', 'I:22', 'Nh:4 K:2', 'Mn:1 So:4 I:9', 'M:1 Nh:1 S:1 K:64']
    '''
    
    # Get the absolute paths 
    current_dir = Dpp_dir
    input_file = os.path.join(current_dir, "test_compounds_to_elements.csv")
    output_file = os.path.join(current_dir, "test_cases_data_preprocessing", "test_elements_and_count.csv")
    
    # Call the main function with absolute file paths
    main_compounds_to_ele_count(input_file, output_file)
    
    # Open the output file and compare the results
    with open(output_file, 'r') as f:
        data = f.readlines()
        ele_count = [line.strip().split(',')[0] for line in data[1:]]
        expected_op = ['Mn:1', 'I:22', 'Nh:4 K:2', 'Mn:1 So:4 I:9', 'M:1 Nh:1 S:1 K:64']
        assert ele_count == expected_op

#==========================================================================================================


def test_mean_atm_prop_integrated():
    '''
    # Purpose of the test: Test the main function main_ele_data_updated to make sure that it fills the missing values of atomic properties with their corresponding available mean values of the property correctly.

    # Input:  File containing atomic propertiers(Atomic Mass, Atomic Radius, Electronegativity) with missing values.
                AtomicMass = ['1', '2', '4',' ', '3', ' ', '9', '1'], AtomicRadius = ['3', '1', ' ', '5', ' ', '6', '9',' '], Electronegativity = ['8',' ','4', '3',' ',' ',' ', '1']
    
    Command to run file: pytest test_data_preprocessing_integrated.py
    
    # Expected Output: File containing atomic property and missed atomic property values filled with mean of the same.
                {'AtomicNumber': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 
                               'AtomicMass': [1.0, 2.0, 4.0, 3.3333333333333335, 3.0, 3.3333333333333335, 9.0, 1.0], 
                                 'AtomicRadius': [3.0, 1.0, 4.8, 5.0, 4.8, 6.0, 9.0, 4.8], 
                                 'Electronegativity': [8.0, 4.0, 4.0, 3.0, 4.0, 4.0, 4.0, 1.0]}
    # Obtained output: File containing atomic property and missed atomic property values filled with mean of the same.
                {'AtomicNumber': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 
                               'AtomicMass': [1.0, 2.0, 4.0, 3.3333333333333335, 3.0, 3.3333333333333335, 9.0, 1.0], 
                                 'AtomicRadius': [3.0, 1.0, 4.8, 5.0, 4.8, 6.0, 9.0, 4.8], 
                                 'Electronegativity': [8.0, 4.0, 4.0, 3.0, 4.0, 4.0, 4.0, 1.0]}
    '''
    
    main_ele_data_updated(os.path.join(Dpp_dir, "test_miss_elements_prop.csv"), os.path.join(Dpp_dir, "test_cases_data_preprocessing", "test_miss_prop_with_mean.csv"))

    filename = os.path.join(Dpp_dir, "test_cases_data_preprocessing", "test_miss_prop_with_mean.csv")
    data = np.genfromtxt(filename, delimiter = ',', names = True, dtype = float, encoding='utf-8')
    atm_prop_dict = {}
    for prop in data.dtype.names:
        if prop != 'Symbol':
            atm_prop_dict[prop] = data[prop].tolist()
    expected_op = {'AtomicNumber': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 
                   'AtomicMass': [1.0, 2.0, 4.0, 3.3333333333333335, 3.0, 3.3333333333333335, 9.0, 1.0], 
                     'AtomicRadius': [3.0, 1.0, 4.8, 5.0, 4.8, 6.0, 9.0, 4.8], 
                     'Electronegativity': [8.0, 4.0, 4.0, 3.0, 4.0, 4.0, 4.0, 1.0]}
    assert(expected_op == atm_prop_dict)
       


#==========================================================================================================

def test_wtd_atm_prop_integrated():
    '''
    # Purpose of the test: Test the main function main_weighted_prop to make sure that it computes weighted atomic properties for the given atomic properties of compounds.

    # Input: Files containing elements and their count, and dictionary of atomic property names as keys and their corresponding values of compounds as values.
        ['Mn:1', 'I:22', 'Nh:4 K:2', 'Mn:1 So:4 I:9', 'M:1 Nh:1 S:1 K:64']
        {Mn: [1, 3, 8], I: [3, 4.8, 4], Nh: [3.333333333, 5, 3], K: [3.333333333, 6, 4], S: [2, 1, 4] }
    
    Command to run file: pytest test_data_preprocessing_integrated.py
    
    # Expected Output: File containing weighted atomic property names and their corresponding weighted values.
        {'AtomicNumber_Mean': [1.0, 5.0, 5.0], 'AtomicNumber_Weighted_Mean': [1.0, 5.0, 4.666666666666666], 'AtomicNumber_Geometric_Mean': [1.0, 5.0, 4.898979485566356], 'AtomicNumber_Weighted_Geometric_Mean': [1.0, 5.0, 4.5788569702133275], 'AtomicNumber_Entropy': [-0.0, -0.0, 0.6730116670092565], 'AtomicNumber_Weighted_Entropy': [-0.0, -0.0, 0.6829081047004717], 'AtomicNumber_Range': [0.0, 0.0, 2.0], 'AtomicNumber_Weighted_Range': [0.0, 0.0, 0.6666666666666665], 'AtomicNumber_Standard_Deviation': [0.0, 0.0, 1.0], 'AtomicNumber_Weighted_Standard_Deviation': [0.0, 0.0, 3.3333333333333326], 
        'AtomicMass_Mean': [1.0, 3.0, 3.3333333333333335], 'AtomicMass_Weighted_Mean': [1.0, 3.0, 3.3333333333333335], 'AtomicMass_Geometric_Mean': [1.0, 3.0, 3.3333333333333335], 'AtomicMass_Weighted_Geometric_Mean': [1.0, 3.0, 3.333333333333333], 'AtomicMass_Entropy': [-0.0, -0.0, 0.6931471805599453], 'AtomicMass_Weighted_Entropy': [-0.0, -0.0, 0.6365141682948128], 'AtomicMass_Range': [0.0, 0.0, 0.0], 'AtomicMass_Weighted_Range': [0.0, 0.0, 1.1111111111111112], 'AtomicMass_Standard_Deviation': [0.0, 0.0, 0.0], 'AtomicMass_Weighted_Standard_Deviation': [0.0, 0.0, 2.4845199749997664], 
        'AtomicRadius_Mean': [3.0, 4.8, 5.5], 'AtomicRadius_Weighted_Mean': [3.0, 4.8, 5.333333333333333], 'AtomicRadius_Geometric_Mean': [3.0, 4.8, 5.477225575051661], 'AtomicRadius_Weighted_Geometric_Mean': [3.0, 4.8, 5.313292845913056], 'AtomicRadius_Entropy': [-0.0, -0.0, 0.6890092384766586], 'AtomicRadius_Weighted_Entropy': [-0.0, -0.0, 0.6615632381579821], 'AtomicRadius_Range': [0.0, 0.0, 1.0], 'AtomicRadius_Weighted_Range': [0.0, 0.0, 1.333333333333333], 'AtomicRadius_Standard_Deviation': [0.0, 0.0, 0.5], 'AtomicRadius_Weighted_Standard_Deviation': [0.0, 0.0, 3.8873012632302], 
        'Electronegativity_Mean': [8.0, 4.0, 3.5], 'Electronegativity_Weighted_Mean': [8.0, 4.0, 3.333333333333333], 'Electronegativity_Geometric_Mean': [8.0, 4.0, 3.4641016151377544], 'Electronegativity_Weighted_Geometric_Mean': [8.0, 4.0, 3.3019272488946263], 'Electronegativity_Entropy': [-0.0, -0.0, 0.6829081047004717], 'Electronegativity_Weighted_Entropy': [-0.0, -0.0, 0.6730116670092565], 'Electronegativity_Range': [0.0, 0.0, 1.0], 'Electronegativity_Weighted_Range': [0.0, 0.0, 0.6666666666666667], 'Electronegativity_Standard_Deviation': [0.0, 0.0, 0.5], 'Electronegativity_Weighted_Standard_Deviation': [0.0, 0.0, 2.403700850309326]}
    # Obtained Output: File containing weighted atomic property names and their corresponding weighted values.
        'AtomicMass_Mean': [1.0, 3.0, 3.3333333333333335], 'AtomicMass_Weighted_Mean': [1.0, 3.0, 3.3333333333333335], 'AtomicMass_Geometric_Mean': [1.0, 3.0, 3.3333333333333335], 'AtomicMass_Weighted_Geometric_Mean': [1.0, 3.0, 3.333333333333333], 'AtomicMass_Entropy': [-0.0, -0.0, 0.6931471805599453], 'AtomicMass_Weighted_Entropy': [-0.0, -0.0, 0.6365141682948128], 'AtomicMass_Range': [0.0, 0.0, 0.0], 'AtomicMass_Weighted_Range': [0.0, 0.0, 1.1111111111111112], 'AtomicMass_Standard_Deviation': [0.0, 0.0, 0.0], 'AtomicMass_Weighted_Standard_Deviation': [0.0, 0.0, 2.4845199749997664], 
        'AtomicRadius_Mean': [3.0, 4.8, 5.5], 'AtomicRadius_Weighted_Mean': [3.0, 4.8, 5.333333333333333], 'AtomicRadius_Geometric_Mean': [3.0, 4.8, 5.477225575051661], 'AtomicRadius_Weighted_Geometric_Mean': [3.0, 4.8, 5.313292845913056], 'AtomicRadius_Entropy': [-0.0, -0.0, 0.6890092384766586], 'AtomicRadius_Weighted_Entropy': [-0.0, -0.0, 0.6615632381579821], 'AtomicRadius_Range': [0.0, 0.0, 1.0], 'AtomicRadius_Weighted_Range': [0.0, 0.0, 1.333333333333333], 'AtomicRadius_Standard_Deviation': [0.0, 0.0, 0.5], 'AtomicRadius_Weighted_Standard_Deviation': [0.0, 0.0, 3.8873012632302], 
        'Electronegativity_Mean': [8.0, 4.0, 3.5], 'Electronegativity_Weighted_Mean': [8.0, 4.0, 3.333333333333333], 'Electronegativity_Geometric_Mean': [8.0, 4.0, 3.4641016151377544], 'Electronegativity_Weighted_Geometric_Mean': [8.0, 4.0, 3.3019272488946263], 'Electronegativity_Entropy': [-0.0, -0.0, 0.6829081047004717], 'Electronegativity_Weighted_Entropy': [-0.0, -0.0, 0.6730116670092565], 'Electronegativity_Range': [0.0, 0.0, 1.0], 'Electronegativity_Weighted_Range': [0.0, 0.0, 0.6666666666666667], 'Electronegativity_Standard_Deviation': [0.0, 0.0, 0.5], 'Electronegativity_Weighted_Standard_Deviation': [0.0, 0.0, 2.403700850309326]}
    '''
    main_weighted_prop(os.path.join(Dpp_dir, "test_compounds_to_elements.csv"), 
                       os.path.join(Dpp_dir, "test_cases_data_preprocessing", "test_elements_and_count.csv"), 
                       os.path.join(Dpp_dir, "test_cases_data_preprocessing", "test_miss_prop_with_mean.csv"), 
                       os.path.join(Dpp_dir, "test_cases_data_preprocessing", "test_wtd_atm_prop.csv"))

    filename = os.path.join(Dpp_dir, "test_cases_data_preprocessing", "test_wtd_atm_prop.csv")
    data = np.genfromtxt(filename, delimiter=',', names=True, dtype=float, encoding='utf-8')
    data = data[:3]
    atm_prop_dict = {}
    for i, prop in enumerate(data.dtype.names):
        if i >= 3 and i < len(data.dtype.names) - 2:
            atm_prop_dict[prop] = data[prop].tolist()
    expected_out = {'AtomicNumber_Mean': [1.0, 5.0, 5.0], 'AtomicNumber_Weighted_Mean': [1.0, 5.0, 4.666666666666666], 'AtomicNumber_Geometric_Mean': [1.0, 5.0, 4.898979485566356], 'AtomicNumber_Weighted_Geometric_Mean': [1.0, 5.0, 4.5788569702133275], 'AtomicNumber_Entropy': [-0.0, -0.0, 0.6730116670092565], 'AtomicNumber_Weighted_Entropy': [-0.0, -0.0, 0.6829081047004717], 'AtomicNumber_Range': [0.0, 0.0, 2.0], 'AtomicNumber_Weighted_Range': [0.0, 0.0, 0.6666666666666665], 'AtomicNumber_Standard_Deviation': [0.0, 0.0, 1.0], 'AtomicNumber_Weighted_Standard_Deviation': [0.0, 0.0, 3.3333333333333326], 
    'AtomicMass_Mean': [1.0, 3.0, 3.3333333333333335], 'AtomicMass_Weighted_Mean': [1.0, 3.0, 3.3333333333333335], 'AtomicMass_Geometric_Mean': [1.0, 3.0, 3.3333333333333335], 'AtomicMass_Weighted_Geometric_Mean': [1.0, 3.0, 3.333333333333333], 'AtomicMass_Entropy': [-0.0, -0.0, 0.6931471805599453], 'AtomicMass_Weighted_Entropy': [-0.0, -0.0, 0.6365141682948128], 'AtomicMass_Range': [0.0, 0.0, 0.0], 'AtomicMass_Weighted_Range': [0.0, 0.0, 1.1111111111111112], 'AtomicMass_Standard_Deviation': [0.0, 0.0, 0.0], 'AtomicMass_Weighted_Standard_Deviation': [0.0, 0.0, 2.4845199749997664], 
    'AtomicRadius_Mean': [3.0, 4.8, 5.5], 'AtomicRadius_Weighted_Mean': [3.0, 4.8, 5.333333333333333], 'AtomicRadius_Geometric_Mean': [3.0, 4.8, 5.477225575051661], 'AtomicRadius_Weighted_Geometric_Mean': [3.0, 4.8, 5.313292845913056], 'AtomicRadius_Entropy': [-0.0, -0.0, 0.6890092384766586], 'AtomicRadius_Weighted_Entropy': [-0.0, -0.0, 0.6615632381579821], 'AtomicRadius_Range': [0.0, 0.0, 1.0], 'AtomicRadius_Weighted_Range': [0.0, 0.0, 1.333333333333333], 'AtomicRadius_Standard_Deviation': [0.0, 0.0, 0.5], 'AtomicRadius_Weighted_Standard_Deviation': [0.0, 0.0, 3.8873012632302], 
    'Electronegativity_Mean': [8.0, 4.0, 3.5], 'Electronegativity_Weighted_Mean': [8.0, 4.0, 3.333333333333333], 'Electronegativity_Geometric_Mean': [8.0, 4.0, 3.4641016151377544], 'Electronegativity_Weighted_Geometric_Mean': [8.0, 4.0, 3.3019272488946263], 'Electronegativity_Entropy': [-0.0, -0.0, 0.6829081047004717], 'Electronegativity_Weighted_Entropy': [-0.0, -0.0, 0.6730116670092565], 'Electronegativity_Range': [0.0, 0.0, 1.0], 'Electronegativity_Weighted_Range': [0.0, 0.0, 0.6666666666666667], 'Electronegativity_Standard_Deviation': [0.0, 0.0, 0.5], 'Electronegativity_Weighted_Standard_Deviation': [0.0, 0.0, 2.403700850309326]}
    assert(expected_out == atm_prop_dict)

#==========================================================================================================


def test_comp_to_ele_integrated():
    
    '''
    # Purpose of the test: Test the main function main_des_2 to make sure that it creates individual column for each element and update the number of atoms of the element in the chemical formula correctly.

    # Input: Files containing elements and their count and the symbols of the elements.
            ['Mn:1', 'I:22', 'Nh:4 K:2', 'Mn:1 So:4 I:9', 'M:1 Nh:1 S:1 K:64']
    
    Command to run file: pytest test_data_preprocessing_integrated.py
    
    # Expected Output: File containing individual element columns and their corresponding count for each compound.
            {'Mn': [1.0, 0.0, 0.0], 'S': [0.0, 0.0, 0.0], 'O': [0.0, 0.0, 0.0], 'Nh': [0.0, 0.0, 4.0], 'I': [0.0, 22.0, 0.0], 'K': [0.0, 0.0, 2.0]}
    # Obtained Output: File containing individual element columns and their corresponding count for each compound.
            {'Mn': [1.0, 0.0, 0.0], 'S': [0.0, 0.0, 0.0], 'O': [0.0, 0.0, 0.0], 'Nh': [0.0, 0.0, 4.0], 'I': [0.0, 22.0, 0.0], 'K': [0.0, 0.0, 2.0]}
    '''
    main_des_2(os.path.join(Dpp_dir, "test_cases_data_preprocessing", "test_elements_and_count.csv"), 
               os.path.join(Dpp_dir, "test_cases_data_preprocessing", "test_miss_prop_with_mean.csv"), 
               os.path.join(Dpp_dir, "test_cases_data_preprocessing", "test_wtd_atm_prop.csv"), 
               os.path.join(Dpp_dir, "test_cases_data_preprocessing", "test_compound_ind_col.csv"))

    filename = os.path.join(Dpp_dir, "test_cases_data_preprocessing", "test_compound_ind_col.csv")
    data = np.genfromtxt(filename, delimiter=',', names=True, dtype=float, encoding='utf-8')
    data = data[:3]
    ind_ele_dict = {}
    for i, ind_ele in enumerate(data.dtype.names):
        if i >len(data.dtype.names) - 11 and i < len(data.dtype.names) - 4:
            ind_ele_dict[ind_ele] = data[ind_ele].tolist()
    expected_output = {'Mn': [1.0, 0.0, 0.0], 'S': [0.0, 0.0, 0.0], 'O': [0.0, 0.0, 0.0], 'Nh': [0.0, 0.0, 4.0], 'I': [0.0, 22.0, 0.0], 'K': [0.0, 0.0, 2.0]}
    assert(expected_output == ind_ele_dict)


#==========================================================================================================

def test_label_encode_integrated():
    '''
    # Purpose of the test: Test the main function main_label_encode to make sure that it label encode the material types correctly.

    # Input: Files containing material type with Metal and Non-metal.
                ['Non-Metal', 'Metal', 'Non-Metal', 'Metal', 'Non-Metal']

    Command to run file: pytest test_data_preprocessing_integrated.py    

    # Expected Output: File containing label encoded material types - 0's and 1's.
                {'material_type': [0.0, 1.0, 0.0, 1.0, 0.0]}

    # Obtained Output: File containing label encoded material types - 0's and 1's.
                {'material_type': [0.0, 1.0, 0.0, 1.0, 0.0]}

    '''
    main_label_encode(os.path.join(Dpp_dir, "test_compounds_to_elements.csv"), 
                      os.path.join(Dpp_dir, "test_cases_data_preprocessing", "test_metal_or_notmetal.csv"))

    filename = os.path.join(Dpp_dir, "test_cases_data_preprocessing", "test_metal_or_notmetal.csv")
    
    data = np.genfromtxt(filename, delimiter=',', names=True, dtype=float, encoding='utf-8')
    label_encode_dict = {}
    for i, mat_type in enumerate(data.dtype.names):
        if i == len(data.dtype.names) - 2:
            label_encode_dict[mat_type] = data[mat_type].tolist()
    expected_op = {'material_type': [0.0, 1.0, 0.0, 1.0, 0.0]}
    assert(expected_op == label_encode_dict)


#==========================================================================================================

def test_one_hot_ele_comp_integrated():
    '''
    # Purpose of the test: Test the main function main_encode_ele_comp to make sure that it computes one-hot encoding based on number of elements and element composition vector based on contribution of each element correctly.

    # Input: Files containing elements and their count.
                ['Mn:1', 'I:22', 'Nh:4 K:2', 'Mn:1 So:4 I:9', 'M:1 Nh:1 S:1 K:64']
    
    Command to run file: pytest test_data_preprocessing_integrated.py
    
    # Expected Output: File containing one-hot encoded columns and element composition columns.
                {'1_num_ele': [1.0, 1.0, 0.0, 0.0, 0.0], '2_num_ele': [0.0, 0.0, 1.0, 0.0, 0.0], '3_num_ele': [0.0, 0.0, 0.0, 1.0, 0.0], '4_num_ele': [0.0, 0.0, 0.0, 0.0, 1.0],
                               'element_composition_1': [1.0, 1.0, 0.6666666666666666, 0.07142857142857142, 0.014925373134328358], 'element_composition_2': [0.0, 0.0, 0.3333333333333333, 0.2857142857142857, 0.014925373134328358], 'element_composition_3': [0.0, 0.0, 0.0, 0.6428571428571429, 0.014925373134328358], 'element_composition_4': [0.0, 0.0, 0.0, 0.0, 0.9552238805970149]}
    # Obtained Output:File containing one-hot encoded columns and element composition columns.
                {'1_num_ele': [1.0, 1.0, 0.0, 0.0, 0.0], '2_num_ele': [0.0, 0.0, 1.0, 0.0, 0.0], '3_num_ele': [0.0, 0.0, 0.0, 1.0, 0.0], '4_num_ele': [0.0, 0.0, 0.0, 0.0, 1.0],
                               'element_composition_1': [1.0, 1.0, 0.6666666666666666, 0.07142857142857142, 0.014925373134328358], 'element_composition_2': [0.0, 0.0, 0.3333333333333333, 0.2857142857142857, 0.014925373134328358], 'element_composition_3': [0.0, 0.0, 0.0, 0.6428571428571429, 0.014925373134328358], 'element_composition_4': [0.0, 0.0, 0.0, 0.0, 0.9552238805970149]}
    '''
    main_encode_ele_comp(os.path.join(Dpp_dir, "test_cases_data_preprocessing", "test_elements_and_count.csv"), 
                         os.path.join(Dpp_dir, "test_cases_data_preprocessing", "test_metal_or_notmetal.csv"), 
                         os.path.join(Dpp_dir, "test_cases_data_preprocessing", "test_one_hot_ele_comp.csv"))

    filename = os.path.join(Dpp_dir, "test_cases_data_preprocessing", "test_one_hot_ele_comp.csv")
    
    data = np.genfromtxt(filename, delimiter=',', names=True, dtype=float, encoding='utf-8')
    data = data
    encode_dict = {}
    for i, encode_comp in enumerate(data.dtype.names):
        if i >= 2 and i < len(data.dtype.names) - 2:
            encode_dict[encode_comp] = data[encode_comp].tolist()
    expected_op = {'1_num_ele': [1.0, 1.0, 0.0, 0.0, 0.0], '2_num_ele': [0.0, 0.0, 1.0, 0.0, 0.0], '3_num_ele': [0.0, 0.0, 0.0, 1.0, 0.0], '4_num_ele': [0.0, 0.0, 0.0, 0.0, 1.0],
                   'element_composition_1': [1.0, 1.0, 0.6666666666666666, 0.07142857142857142, 0.014925373134328358], 'element_composition_2': [0.0, 0.0, 0.3333333333333333, 0.2857142857142857, 0.014925373134328358], 'element_composition_3': [0.0, 0.0, 0.0, 0.6428571428571429, 0.014925373134328358], 'element_composition_4': [0.0, 0.0, 0.0, 0.0, 0.9552238805970149]}
    assert(expected_op == encode_dict)


#==========================================================================================================

def test_ordinal_encode_integrated():
    '''
    # Purpose of the test: Test the main function main_ordinal_label_enode to make sure that it computes label encoding of space group and ordinal encoding of thermal conductivity values correctly.

    # Input: Files containing space group values and thermal conductivity values.
            {Spacegroup: [227, 225, 223, 225, 122]}
            {thermal_conduxtivity: [1.46419, 150, 0.689631, 80, 500]}
            
    Command to run file: pytest test_data_preprocessing_integrated.py
    
    # Expected Output: File containing label encoded space group and ordinal encoded thermal conductivity values.
            {'spacegroup': [3.0, 2.0, 1.0, 2.0, 0.0], 
                'thermal_conductivity': [1.0, 3.0, 1.0, 2.0, 3.0]}
    # Obtained Output: File containing label encoded space group and ordinal encoded thermal conductivity values.
            {'spacegroup': [3.0, 2.0, 1.0, 2.0, 0.0], 
                 'thermal_conductivity': [1.0, 3.0, 1.0, 2.0, 3.0]}
    '''
    main_ordinal_label_enode(os.path.join(Dpp_dir, "test_compounds_to_elements.csv"),
                             os.path.join(Dpp_dir, "test_cases_data_preprocessing", "ordinal_label_encod.csv"))

    filename = os.path.join(Dpp_dir, "test_cases_data_preprocessing", "ordinal_label_encod.csv")
    
    data = np.genfromtxt(filename, delimiter=',', names=True, dtype=float, encoding='utf-8')
    ordinal_sg_dict = {}
    for i, encode_comp in enumerate(data.dtype.names):
        if i >= 1 and i < len(data.dtype.names) - 2:
            ordinal_sg_dict[encode_comp] = data[encode_comp].tolist()
    expected_output = {'spacegroup': [3.0, 2.0, 1.0, 2.0, 0.0], 
                       'thermal_conductivity': [1.0, 3.0, 1.0, 2.0, 3.0]}
    assert(expected_output == ordinal_sg_dict)

#==========================================================================================================

