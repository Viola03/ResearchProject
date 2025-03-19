### Simplified Inference for Validation dataset (NOT RapidSim tuple)

from fast_vertex_quality_inference.processing.data_manager import tuple_manager
from fast_vertex_quality_inference.processing.network_manager import network_manager
import numpy as np
import uproot

### NETWORKS ###

vertexing_network = network_manager(
					network="inference/models/baseline/vertexing_decoder_model.onnx", 
					config="inference/models/baseline/vertexing_configs.pkl", 
					transformers="inference/models/baseline/vertexing_transfomers.pkl", 
                    )

vertexing_encoder = network_manager(
					network="inference/models/baseline/vertexing_encoder_model.onnx", 
					config="inference/models/baseline/vertexing_configs.pkl", 
					transformers="inference/models/baseline/vertexing_transfomers.pkl", 
                    )


### DATA ####

file_path = "/users/zw21147/ResearchProject/datasets/split/validation_renamed.root"
with uproot.open(file_path) as f:
    tree = f["DecayTree"]
    vertexing_conditions = tree.arrays(vertexing_network.conditions, library="np")
    df = tree.arrays(library="pd")

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

# vertexing_conditions = data_tuple.get_branches(
# 					vertexing_network.conditions, 
# 					vertexing_network.Transformers, 
# 					numpy=True,
# 					)

vertexing_output = vertexing_network.query_network(
					['noise', X],
					)

print(vertexing_output.shape)

vertexing_output.to_csv("validation_inference.csv", index=False)
