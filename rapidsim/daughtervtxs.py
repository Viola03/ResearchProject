import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uproot

# Load RapidSim
rapidsim_file = "/users/zw21147/ResearchProject/rapidsim/Kee/kee_tree.root"  
tree_name = "DecayTree"

data = uproot.open(rapidsim_file)[tree_name].arrays(library="pd")

# Define estimated track lengths for stable particles
TRACK_LENGTHS = {
    "K_plus": 500.0,  # Kaons typically travel ~50 cm - 5m before interaction
    "e_plus": 200.0,  # Electrons have high interaction but travel some distance
    "e_minus": 200.0   # Same for positrons
}

def compute_end_vertex(row, track_length, prefix):
    """Compute the estimated end vertex based on momentum direction."""
    p_mag = np.sqrt(row[f'{prefix}PX_TRUE']**2 + row[f'{prefix}PY_TRUE']**2 + row[f'{prefix}PZ_TRUE']**2)
    scale = track_length / p_mag if p_mag > 0 else 0
    return row[f'{prefix}origX_TRUE'] + scale * row[f'{prefix}PX_TRUE'], row[f'{prefix}origY_TRUE'] + scale * row[f'{prefix}PY_TRUE'], row[f'{prefix}origZ_TRUE'] + scale * row[f'{prefix}PZ_TRUE']

# Compute end vertices for each particle
for particle in TRACK_LENGTHS.keys():
    data[f'{particle}_vtxX'], data[f'{particle}_vtxY'], data[f'{particle}_vtxZ'] = zip(*data.apply(
        lambda row: compute_end_vertex(row, TRACK_LENGTHS[particle], f'{particle}_'), axis=1))

# Save 
output_file = "kee_with_daughtervtx.root"
with uproot.recreate(output_file) as file:
    file[tree_name] = data

print(f"Modified data saved to {output_file}")

### validation ###

# Load LHCb sample for validation
lhcb_file = "/users/zw21147/ResearchProject/datasets/combinatorial_select_Kuu_renamed.root"  
lhcb_data = uproot.open(lhcb_file)[tree_name].arrays(library="pd")

# match LHCb branch naming convention
daughter_map = {
    "K_plus": "DAUGHTER1_",
    "e_plus": "DAUGHTER2_",
    "e_minus": "DAUGHTER3_"
}

# Plot comparison of end vertex distributions
fig, axes = plt.subplots(3, 3, figsize=(15, 15))

for j, (particle, prefix) in enumerate(daughter_map.items()):
    for i, coord in enumerate(["vtxX", "vtxY", "vtxZ"]):
        rapidsim_hist, bins = np.histogram(data[f'{particle}_{coord}'], bins=100, density=True)
        lhcb_hist, _ = np.histogram(lhcb_data[f'{prefix}{coord}_TRUE'], bins=bins, density=True)
        
        axes[j, i].step(bins[:-1], rapidsim_hist, where='mid', alpha=0.5, label=f'RapidSim {particle}')
        axes[j, i].step(bins[:-1], lhcb_hist, where='mid', alpha=0.5, label=f'LHCb {particle}')
        
        axes[j, i].set_xlabel(coord)
        axes[j, i].set_title(f"{particle} End Vertex {coord} Distribution")
        axes[j, i].set_xlim(-50, 50)
        axes[j, i].set_ylim(0, max(rapidsim_hist.max(), lhcb_hist.max()))
        axes[j, i].legend()

plt.tight_layout()
plt.savefig("end_vertex_distributions.png")
print('Fig saved')
