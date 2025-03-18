import uproot
import pandas as pd
import numpy as np
import pickle
import joblib

# NB model trained on scikit-learn==1.4.1.post1

def query_BDT(data, mother, particles, intermediate, BDT_branch_name, tag=''):
    """
    Applies BDT model to a given dataset.
    
    Parameters:
    - data (pd.DataFrame): The dataset containing the variables.
    - mother (str): Name of the mother particle.
    - particles (list of str): Names of the daughter particles.
    - intermediate (str): Name of the intermediate particle.
    - BDT_branch_name (str): The column name where BDT output will be stored.
    - tag (str, optional): Prefix for variable names (default is '').

    Returns:
    - pd.DataFrame: The modified dataset with the BDT output column added.
    """

    # BDT target features
    BDT_targets = [
        f"{tag}{mother}_ENDVERTEX_CHI2",
        f"{tag}{mother}_IPCHI2_OWNPV",
        f"{tag}{mother}_FDCHI2_OWNPV",
        f"{tag}{mother}_DIRA_OWNPV",
        f"{tag}{particles[0]}_IPCHI2_OWNPV",
        f"{tag}{particles[0]}_TRACK_CHI2NDOF",
        f"{tag}{particles[1]}_IPCHI2_OWNPV",
        f"{tag}{particles[1]}_TRACK_CHI2NDOF",
        f"{tag}{particles[2]}_IPCHI2_OWNPV",
        f"{tag}{particles[2]}_TRACK_CHI2NDOF",
        f"{tag}{intermediate}_FDCHI2_OWNPV",
        f"{tag}{intermediate}_IPCHI2_OWNPV",
    ]
    
    # Load the BDT model
    with open("/users/zw21147/ResearchProject/scripts/validation/BDT_sig_comb_WGANcocktail_newconditions.pkl", "rb") as model_file:
        clf = pickle.load(model_file)[0]["BDT"]

    # with open("/users/zw21147/ResearchProject/scripts/validation/BDT_sig_comb_WGANcocktail_newconditions.pkl", "rb") as model_file:
    #     clf = joblib.load(model_file, mmap_mode='r')

    for target in BDT_targets:
        if target in data:
            data[target] = pd.to_numeric(data[target], errors='coerce')

    # Extract valid samples
    sample = data[BDT_targets].to_numpy()
    nan_rows = np.unique(np.where(np.isnan(sample))[0])

    # Initialize BDT response array
    bdt_responses = np.full(len(sample), np.nan)
    non_nan_rows = np.setdiff1d(np.arange(len(sample)), nan_rows)

    if len(non_nan_rows) > 0:
        bdt_responses[non_nan_rows] = clf.predict_proba(sample[non_nan_rows])[:, 1]

    # Store the BDT response in the dataset
    data[f'{tag}{BDT_branch_name}'] = bdt_responses
    
    return data  # Return the modified dataset

