import uproot
import awkward as ak
import pandas as pd

tree = uproot.open("/users/zw21147/ResearchProject/datasets/combinatorial_select_Kuu.root")["DecayTree"]
branches = list(tree.keys())
data = tree.arrays(branches, library="pd")


# Use the first column's length as the expected length
expected_length = None

for col in data.columns:
    # Convert the pandas Series to a list (one element per event)
    try:
        col_list = data[col].to_list()
    except Exception as e:
        print(f"Error converting column '{col}' to a list: {e}")
        continue

    length = len(col_list)
    if expected_length is None:
        expected_length = length

    print(f"Column {col} has {length} entries.")

    # Print the types of the first 5 entries to see if anything looks unusual
    sample_types = [type(x) for x in col_list[:5]]
    print(f"Sample types for {col}: {sample_types}")

    if length != expected_length:
        print(f"WARNING: Column '{col}' has length {length}, which differs from expected length {expected_length}.")