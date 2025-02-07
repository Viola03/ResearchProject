import uproot
import awkward as ak
import numpy as np
import tools as t

def renamed(file_name):
    
    masses = {521: 5279.34}  # B_plus mass in MeV
    particles = ["DAUGHTER1", "DAUGHTER2", "DAUGHTER3"]
    mother = "MOTHER"
    
    # Open the original tree with uproot
    tree = uproot.open(file_name)["DecayTree"]
    
    # Get the list of branch names
    branches = list(tree.keys())
    
    # Define a function that applies your renaming rules
    def rename_branch(branch):
        new = branch
        if "TRUEENDVERTEX_X" in new:
            new = new.replace("TRUEENDVERTEX_X", "vtxX_TRUE")
        elif "ENDVERTEX_X" in new:
            new = new.replace("ENDVERTEX_X", "vtxX")
        elif "TRUEENDVERTEX_Y" in new:
            new = new.replace("TRUEENDVERTEX_Y", "vtxY_TRUE")
        elif "ENDVERTEX_Y" in new:
            new = new.replace("ENDVERTEX_Y", "vtxY")
        elif "TRUEENDVERTEX_Z" in new:
            new = new.replace("TRUEENDVERTEX_Z", "vtxZ_TRUE")
        elif "ENDVERTEX_Z" in new:
            new = new.replace("ENDVERTEX_Z", "vtxZ")
        elif "TRUEORIGINVERTEX_X" in new:
            new = new.replace("TRUEORIGINVERTEX_X", "origX_TRUE")
        elif "ORIGINVERTEX_X" in new:
            new = new.replace("ORIGINVERTEX_X", "origX")
        elif "TRUEORIGINVERTEX_Y" in new:
            new = new.replace("TRUEORIGINVERTEX_Y", "origY_TRUE")
        elif "ORIGINVERTEX_Y" in new:
            new = new.replace("ORIGINVERTEX_Y", "origY")
        elif "TRUEORIGINVERTEX_Z" in new:
            new = new.replace("TRUEORIGINVERTEX_Z", "origZ_TRUE")
        elif "ORIGINVERTEX_Z" in new:
            new = new.replace("ORIGINVERTEX_Z", "origZ")
        elif "TRUEP_X" in new:
            new = new.replace("TRUEP_X", "PX_TRUE")
        elif "TRUEP_Y" in new:
            new = new.replace("TRUEP_Y", "PY_TRUE")
        elif "TRUEP_Z" in new:
            new = new.replace("TRUEP_Z", "PZ_TRUE")
        return new

    # Create the mapping from old branch names to new branch names
    new_branches = [rename_branch(b) for b in branches]
    branch_mapping = dict(zip(branches, new_branches))
    
    # Read the data for all branches into a pandas DataFrame.
    # (Some branches may be jagged and stored as Awkward arrays.)
    data = tree.arrays(branches, library="pd")
    print("Read data...")
    
    # Rename the DataFrame columns
    data.rename(columns=branch_mapping, inplace=True)
    print("Renamed columns...")
    
    
    
    
    
    # Compute additional branches
    
    ## Compute FLIGHT distances
    min_FD = 5e-5
    mother_FD = np.asarray(t.compute_distance(data, mother, 'vtx', mother, 'orig'))
    particle_FDs = {p: np.asarray(t.compute_distance(data, p, 'orig', mother, 'orig')) for p in particles}

    # Prevent zero distances by setting minimum values
    mother_FD[mother_FD == 0] = min_FD
    for p in particles:
        particle_FDs[p][particle_FDs[p] == 0] = min_FD

    data[f"{mother}_FLIGHT"] = mother_FD
    for p in particles:
        data[f"{p}_FLIGHT"] = particle_FDs[p]

    ## Compute angles
    for p in particles:
        data[f"angle_{p}"] = t.compute_angle(data, mother, p)
        print('Computed angles...')

    ## Compute momentum variables
    if {f"{mother}_PX_TRUE", f"{mother}_PY_TRUE", f"{mother}_PZ_TRUE"}.issubset(data.columns):
        data[f"{mother}_P"] = np.sqrt(
            data[f"{mother}_PX_TRUE"]**2 + data[f"{mother}_PY_TRUE"]**2 + data[f"{mother}_PZ_TRUE"]**2
        )
        data[f"{mother}_PT"] = np.sqrt(
            data[f"{mother}_PX_TRUE"]**2 + data[f"{mother}_PY_TRUE"]**2
        )
        data[f"{mother}_M"] = np.sqrt(
            data[f"{mother}_PX_TRUE"]**2 + data[f"{mother}_PY_TRUE"]**2 + data[f"{mother}_PZ_TRUE"]**2 + masses[521]**2
        )
        print("Computed MOTHER P, PT, and M.")
      
    
    
    # Define a helper to convert each element if it is an Awkward array.
    def convert_series(series):
        return [ak.to_list(x) if hasattr(x, "to_list") else x for x in series]
    
    # Convert all columns; the resulting dictionary has only Python objects
    converted_columns = {col: convert_series(data[col]) for col in data.columns}
    
    # Write the converted data to a new ROOT file.
    # Uproot will now see only plain lists (or numpy arrays if possible)
    output_file_path = file_name.replace(".root", "_renamed.root")
    with uproot.recreate(output_file_path) as fout:
        fout["DecayTree"] = converted_columns
    
    print("Wrote renamed tree to", output_file_path)
    return output_file_path

# Example usage:
if __name__ == "__main__":
    renamed("/users/zw21147/ResearchProject/datasets/combinatorial_select_Kuu.root")
