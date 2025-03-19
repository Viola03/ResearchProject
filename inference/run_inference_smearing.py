### Inference on RapidSim Tuple

from fast_vertex_quality_inference.processing.data_manager import tuple_manager
from fast_vertex_quality_inference.processing.network_manager import network_manager
import numpy as np
import uproot
import awkward as ak

### NETWORKS ###

rapidsim_PV_smearing_network = network_manager(
					network="example/models/smearing_decoder_model.onnx", 
					config="example/models/smearing_configs.pkl", 
					transformers="example/models/smearing_transfomers.pkl", 
					)

print('Initialised networks')

### DATA ####

file_path = "/users/zw21147/ResearchProject/rapidsim/Kee/kee_tree.root"

with uproot.open(file_path) as f:
    tree = f["DecayTree"]
    # smearing_conditions =  tree.arrays(rapidsim_PV_smearing_network.conditions, library="np")
    df = tree.arrays(library="pd")

### RENAMING ###

rename_dict = {
    "B_plus": "MOTHER",
    "K_plus": "DAUGHTER1",
    "e_plus": "DAUGHTER2",
    "e_minus": "DAUGHTER3",
    "J_psi_1S": "INTERMEDIATE",
}

new_columns = {}
for col in df.columns:
    for old_name, new_name in rename_dict.items():
        if old_name in col: 
            new_columns[col] = col.replace(old_name, new_name)

df = df.rename(columns=new_columns)

# Identify and remove jagged columns
jagged_cols = [col for col in df.columns if isinstance(df[col].iloc[0], (list, ak.highlevel.Array))]

if jagged_cols:
    print(f"Removing jagged columns: {jagged_cols}")
    df = df.drop(columns=jagged_cols)


# Convert DataFrame to dictionary of numpy arrays
numpy_dict = {col: np.array(df[col]) for col in df.columns}

# Apply the selection from conditions to numpy_dict
smearing_conditions = {col: numpy_dict[col] for col in rapidsim_PV_smearing_network.conditions if col in numpy_dict}


### SMEAR PV ###

smearing_conditions_processed = {}

for col in rapidsim_PV_smearing_network.conditions:
     arr_physical = smearing_conditions[col]
     if col in rapidsim_PV_smearing_network.Transformers:
         arr_processed = rapidsim_PV_smearing_network.Transformers[col].process(arr_physical)
     else:
         arr_processed = arr_physical  # if no transformer for col
     smearing_conditions_processed[col] = arr_processed

X = np.column_stack([smearing_conditions_processed[col] for col in rapidsim_PV_smearing_network.conditions])

print('Applying smearing')

smeared_PV_output = rapidsim_PV_smearing_network.query_network(
					['noise', X],
					)

print(smeared_PV_output)

smeared_PV_output.to_csv("rapidsim_smearing_inference.csv", index=False)
