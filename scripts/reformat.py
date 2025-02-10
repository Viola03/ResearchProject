import uproot
import numpy as np
import pandas as pd
import tools as t
import awkward as ak

### TAKE ROOT FILES AND REFORMATS ###


masses = {} #GeV/c^2
masses[521] = 5.27925
masses[321] = 493.677 * 1e-3
masses[211] = 139.57039 * 1e-3
masses[13] = 105.66 * 1e-3
masses[11] = 0.51099895000 * 1e-3

#file = '/users/zw21147/ResearchProject/datasets_mixed/mixed_Kee_2.root'

############################################################

def extra_conditions(file_name):
    ### Takes a ROOT file and reformats it for training ###
    
    particles = ["K_plus", "e_minus", "e_plus"] 
    mother = 'B_plus'
    
    file = uproot.open(file_name)
    tree = file["DecayTree"] 
    events = tree.arrays(library="pd")

    # Compute additional branches
    # events[f"DIRA_{mother}"] = t.compute_DIRA(events, mother, particles, true_vars=True, true_vertex=False)
    # events[f"DIRA_{mother}_true_vertex"] = t.compute_DIRA(events, mother, particles, true_vars=True, true_vertex=True)

    # FD
    A = t.compute_distance(events, mother, 'vtx', mother, 'orig')
    B = t.compute_distance(events, particles[0], 'orig', mother, 'orig')
    C = t.compute_distance(events, particles[1], 'orig', mother, 'orig')
    D = t.compute_distance(events, particles[2], 'orig', mother, 'orig')

    A = np.asarray(A)
    B = np.asarray(B)
    C = np.asarray(C)
    D = np.asarray(D)

    min_A = 5e-5
    min_B = 5e-5
    min_C = 5e-5
    min_D = 5e-5

    A[np.where(A==0)] = min_A
    B[np.where(B==0)] = min_B
    C[np.where(C==0)] = min_C
    D[np.where(D==0)] = min_D

    events[f"{mother}_FLIGHT"] = A
    events[f"{particles[0]}_FLIGHT"] = B
    events[f"{particles[1]}_FLIGHT"] = C
    events[f"{particles[2]}_FLIGHT"] = D
    
    #TRUEID
    
    events['B_plus_TRUEID'] = 521 #?
    
    events['K_plus_TRUEID'] = 321
    events['e_plus_TRUEID'] = 11
    events['e_minus_TRUEID'] = 11
    
    # Angles
    for particle in particles:
        events[f"angle_{particle}"] = t.compute_angle(events, mother, f"{particle}")

    # B P/PT
    
    events['B_plus_P'] = np.sqrt(events['B_plus_PX_TRUE']**2 + events['B_plus_PY_TRUE']**2 + events['B_plus_PZ_TRUE']**2)
    events['B_plus_PT'] = np.sqrt(events['B_plus_PX_TRUE']**2 + events['B_plus_PY_TRUE']**2)
    
    # M
    
    events['B_plus_M'] = np.sqrt(events['B_plus_PX_TRUE']**2 + events['B_plus_PY_TRUE']**2 + events['B_plus_PZ_TRUE']**2 + masses[521]**2)
    
    return events



def resample_PV(df, column_name, retain_original_size=True, random_state=None):
    """
    Removes zero values from the specified column in the dataframe and resamples 
    from the remaining distribution to match the original size if retain_original_size is True.
    """
    
    # Filter out zero values
    filtered_data = df[df[column_name] != 0][column_name]

    # Determine sample size
    if retain_original_size:
        sample_size = len(df)  # Keep original size (including zeros that were removed)
    else:
        sample_size = len(filtered_data)  # Use only the nonzero count

    # Resample with replacement
    resampled_data = np.random.default_rng(seed=random_state).choice(filtered_data, size=sample_size, replace=True)
    
    ###
    # Plot histograms before and after resampling
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Histogram before resampling
    axes[0].hist(df[column_name], bins=100, alpha=0.7, label='Original')
    axes[0].set_title(f'Original {column_name}')
    axes[0].set_xlabel(column_name)
    axes[0].set_ylabel('Frequency')

    # Histogram after resampling
    axes[1].hist(resampled_data, bins=100, alpha=0.7, label='Resampled')
    axes[1].set_title(f'Resampled {column_name}')
    axes[1].set_xlabel(column_name)
    axes[1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(f'/users/zw21147/ResearchProject/resampled_histogram_{column_name}.png')
    plt.close()
    ###
    
    # Replace the original column with the resampled data
    # Ensure resampled_data is of the same size as the original column
    if len(resampled_data) == len(df[column_name]):
        df[column_name] = resampled_data
    else:
        raise ValueError("Resampled data size does not match the original column size.")
    
    return df


## Extra conditions for mixed samples

#events = extra_conditions(file)
#output_file_path = "/users/zw21147/ResearchProject/datasets_mixed/mixed_Kee_newconditions.root"

## Resampling for full combinatorial sample
file_name = '/users/zw21147/ResearchProject/datasets/combinatorial_select_Kuu_renamed.root'
file = uproot.open(file_name)
tree = file["DecayTree"] 
events = tree.arrays(library="pd")

events = resample_PV(events, 'MOTHER_vtxX_TRUE')
events = resample_PV(events, 'MOTHER_vtxY_TRUE')
events = resample_PV(events, 'MOTHER_vtxZ_TRUE')

# for col in events.columns:
#     print(f"{col}: {type(events[col].values)}")

# Drop columns containing 'COV_'
columns_to_drop = [col for col in events.columns if 'COV_' in col]
events.drop(columns=columns_to_drop, inplace=True)

output_file_path = "/users/zw21147/ResearchProject/datasets/combinatorial_select_Kuu_renamed_resampled.root"

print('Writing to new output file...')

data_dict = {col: events[col].values for col in events.columns}

with uproot.recreate(output_file_path) as f:
    f["DecayTree"] = data_dict
    
print(f"Wrote resampled tree to {output_file_path}")

    