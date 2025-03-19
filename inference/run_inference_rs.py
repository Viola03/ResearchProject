### Inference on RapidSim Tuple

from fast_vertex_quality_inference.processing.data_manager import tuple_manager
from fast_vertex_quality_inference.processing.network_manager import network_manager
import numpy as np
import uproot

### NETWORKS ###

vertexing_network = network_manager(
					network="models/baseline/vertexing_decoder_model.onnx", 
					config="models/baseline/vertexing_configs.pkl", 
					transformers="models/baseline/vertexing_transfomers.pkl", 
                    )

vertexing_encoder = network_manager(
					network="models/baseline/vertexing_encoder_model.onnx", 
					config="models/baseline/vertexing_configs.pkl", 
					transformers="models/baseline/vertexing_transfomers.pkl", 
                    )

rapidsim_PV_smearing_network = network_manager(
					network="example/models/smearing_decoder_model.onnx", 
					config="example/models/smearing_configs.pkl", 
					transformers="example/models/smearing_transfomers.pkl", 
					)

print('Initialised networks')

### DATA ####

file_path = "/users/zw21147/ResearchProject/datasets_mixed/RapidSimSample.root"

with uproot.open(file_path) as f:
    tree = f["DecayTree"]
    # smearing_conditions =  tree.arrays(rapidsim_PV_smearing_network.conditions, library="np")
    vertexing_conditions = tree.arrays(vertexing_network.conditions, library="np")
    df = tree.arrays(library="pd")

print('Data read')

### SMEAR PV ###

# smearing_conditions_processed = {}

# for col in rapidsim_PV_smearing_network.conditions:
#      arr_physical = smearing_conditions[col]
#      if col in rapidsim_PV_smearing_network.Transformers:
#          arr_processed = rapidsim_PV_smearing_network.Transformers[col].process(arr_physical)
#      else:
#          arr_processed = arr_physical  # if no transformer for col
#      smearing_conditions_processed[col] = arr_processed

# X = np.column_stack([smearing_conditions_processed[col] for col in vertexing_network.conditions])

# print('Applying smearing')

# smeared_PV_output = rapidsim_PV_smearing_network.query_network(
# 					['noise', X],
# 					)

# print(smeared_PV_output)

### VERTEXING ###

# Apply each transform to the corresponding column
vertexing_conditions_processed = {}
for col in vertexing_network.conditions:
    arr_physical = vertexing_conditions[col]
    if col in vertexing_network.Transformers:
        arr_processed = vertexing_network.Transformers[col].process(arr_physical)
    else:
        arr_processed = arr_physical  # if no transformer for col
    vertexing_conditions_processed[col] = arr_processed

# single 2D array with the same column order as `vertexing_network.conditions`:
X = np.column_stack([vertexing_conditions_processed[col] for col in vertexing_network.conditions])

print("Processed shape:", X.shape)


vertexing_output = vertexing_network.query_network(
					['noise', X],
					)

print(vertexing_output.shape)

vertexing_output.to_csv("rapidsim_inference.csv", index=False)
