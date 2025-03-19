import uproot
import numpy as np
import pandas as pd
import awkward as ak

rename_dict = {
    "MOTHER": "B_plus",
    "DAUGHTER1": "K_plus",
    "DAUGHTER2": "e_plus",
    "DAUGHTER3": "e_minus",
    "INTERMEDIATE": "J_psi_1S",
}


output_root_file = "/users/zw21147/ResearchProject/datasets/split/validation_renamed.root"

file = uproot.open('/users/zw21147/ResearchProject/datasets/split/validation.root')
tree = file["DecayTree"] 
df = tree.arrays(library="pd")

# new_columns = {}
# for col in df.columns:
#     for old_name, new_name in rename_dict.items():
#         if col.startswith(old_name):
#             new_columns[col] = col.replace(old_name, new_name)

new_columns = {}
for col in df.columns:
    for old_name, new_name in rename_dict.items():
        if old_name in col:  # checks if the substring exists anywhere in the column name
            new_columns[col] = col.replace(old_name, new_name)

df = df.rename(columns=new_columns)

# Identify and remove jagged columns
jagged_cols = [col for col in df.columns if isinstance(df[col].iloc[0], (list, ak.highlevel.Array))]

if jagged_cols:
    print(f"Removing jagged columns: {jagged_cols}")
    df = df.drop(columns=jagged_cols)


# Convert DataFrame to dictionary of numpy arrays
numpy_dict = {col: np.array(df[col]) for col in df.columns}
tree_name = 'DecayTree'

# Save to new ROOT file
with uproot.recreate(output_root_file) as new_file:
    new_file[tree_name] = numpy_dict