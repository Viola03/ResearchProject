import uproot
import numpy as np
import pandas as pd
import tools as t

### TAKE ROOT FILES AND REFORMATS ###

masses = {}
masses[321] = 493.677
masses[211] = 139.57039
masses[13] = 105.66
masses[11] = 0.51099895000 * 1e-3

file = '/users/zw21147/ResearchProject/datasets_mixed/mixed_Kee_2.root'

############################################################

def renamed(file_name):
    
    DAUGHTER1_TRUEID = 321
    DAUGHTER2_TRUEID = 11
    DAUGHTER3_TRUEID = 11

    # file_name = 'Kmumu/Kmumu_tree.root'
    # DAUGHTER1_TRUEID = 321
    # DAUGHTER2_TRUEID = 13
    # DAUGHTER3_TRUEID = 13

    particles = ["DAUGHTER1", "DAUGHTER2", "DAUGHTER3"]
    mother = 'MOTHER'
    intermediate = 'INTERMEDIATE'

    print("Opening file...")

    file = uproot.open(f"{directory}/{file_name}:DecayTree")
    branches = file.keys()
    print(branches)
    print('\n')
    new_branches = []

    for branch in branches:
        new_branch = branch

        if "B_plus" in branch:
            new_branch = new_branch.replace("B_plus", "MOTHER")
        if "K_plus" in branch:
            new_branch = new_branch.replace("K_plus", "DAUGHTER1")
        if "e_plus" in branch:
            new_branch = new_branch.replace("e_plus", "DAUGHTER2")
        if "e_minus" in branch:
            new_branch = new_branch.replace("e_minus", "DAUGHTER3")

        if "vtxX" in branch:
            if "TRUE" in branch:
                new_branch = new_branch.replace("vtxX", "TRUEENDVERTEX_X")
            else:
                new_branch = new_branch.replace("vtxX", "ENDVERTEX_X")
            if branch[-4:] == "TRUE":
                new_branch = new_branch[:-5]  # remove _TRUE
        if "vtxY" in branch:
            if "TRUE" in branch:
                new_branch = new_branch.replace("vtxY", "TRUEENDVERTEX_Y")
            else:
                new_branch = new_branch.replace("vtxY", "ENDVERTEX_Y")
            if branch[-4:] == "TRUE":
                new_branch = new_branch[:-5]  # remove _TRUE
        if "vtxZ" in branch:
            if "TRUE" in branch:
                new_branch = new_branch.replace("vtxZ", "TRUEENDVERTEX_Z")
            else:
                new_branch = new_branch.replace("vtxZ", "ENDVERTEX_Z")
            if branch[-4:] == "TRUE":
                new_branch = new_branch[:-5]  # remove _TRUE

        if "origX" in branch:
            if "TRUE" in branch:
                new_branch = new_branch.replace("origX", "TRUEORIGINVERTEX_X")
            else:
                new_branch = new_branch.replace("origX", "ORIGINVERTEX_X")
            if branch[-4:] == "TRUE":
                new_branch = new_branch[:-5]  # remove _TRUE
        if "origY" in branch:
            if "TRUE" in branch:
                new_branch = new_branch.replace("origY", "TRUEORIGINVERTEX_Y")
            else:
                new_branch = new_branch.replace("origY", "ORIGINVERTEX_Y")
            if branch[-4:] == "TRUE":
                new_branch = new_branch[:-5]  # remove _TRUE
        if "origZ" in branch:
            if "TRUE" in branch:
                new_branch = new_branch.replace("origZ", "TRUEORIGINVERTEX_Z")
            else:
                new_branch = new_branch.replace("origZ", "ORIGINVERTEX_Z")
            if branch[-4:] == "TRUE":
                new_branch = new_branch[:-5]  # remove _TRUE

        if "_P" in branch:
            if new_branch == 'MOTHER_P_TRUE' or new_branch == 'MOTHER_PT_TRUE':
                drop_idx = drop(drop_idx)
                continue
            if "TRUE" in branch:
                new_branch = new_branch[:-5]  # remove _TRUE
                new_branch = new_branch[:-3] + '_TRUEP_' + new_branch[-1]

        new_branches.append(new_branch)

    events = file.arrays(branches, library='pd')
    events.columns = new_branches

    # set TRUEID
    events['DAUGHTER1_TRUEID'] = DAUGHTER1_TRUEID
    events['DAUGHTER2_TRUEID'] = DAUGHTER2_TRUEID
    events['DAUGHTER3_TRUEID'] = DAUGHTER3_TRUEID

    ### Flight distances ###

    A = vt.compute_distance(events, mother, 'TRUEENDVERTEX', mother, 'TRUEORIGINVERTEX')
    B = vt.compute_distance(events, particles[0], 'TRUEORIGINVERTEX', mother, 'TRUEENDVERTEX')
    C = vt.compute_distance(events, particles[1], 'TRUEORIGINVERTEX', mother, 'TRUEENDVERTEX')
    D = vt.compute_distance(events, particles[2], 'TRUEORIGINVERTEX', mother, 'TRUEENDVERTEX')

    print(A, B, C, D)

    A = np.asarray(A)
    B = np.asarray(B)
    C = np.asarray(C)
    D = np.asarray(D)

    min_A = 5e-5
    min_B = 5e-5
    min_C = 5e-5
    min_D = 5e-5

    A[np.where(A == 0)] = min_A
    B[np.where(B == 0)] = min_B
    C[np.where(C == 0)] = min_C
    D[np.where(D == 0)] = min_D

    events[f"{mother}_FLIGHT"] = A
    events[f"{particles[0]}_FLIGHT"] = B
    events[f"{particles[1]}_FLIGHT"] = C
    events[f"{particles[2]}_FLIGHT"] = D

    ### Angles ###

    for particle in particles:
        events[f"angle_{particle}"] = vt.compute_angle(events, mother, f"{particle}")

    ### IP / DIRA ###

    true_vertex = False
    events[f"IP_{mother}"] = vt.compute_impactParameter(events, mother, particles, true_vertex=true_vertex)
    for particle in particles:
        events[f"IP_{particle}"] = vt.compute_impactParameter_i(events, mother, f"{particle}", true_vertex=true_vertex)
    events[f"FD_{mother}"] = vt.compute_flightDistance(events, mother, particles, true_vertex=true_vertex)
    events[f"DIRA_{mother}"] = vt.compute_DIRA(events, mother, particles, true_vertex=true_vertex)

    true_vertex = True
    events[f"IP_{mother}_true_vertex"] = vt.compute_impactParameter(events, mother, particles, true_vertex=true_vertex)
    for particle in particles:
        events[f"IP_{particle}_true_vertex"] = vt.compute_impactParameter_i(events, mother, f"{particle}", true_vertex=true_vertex)
    events[f"FD_{mother}_true_vertex"] = vt.compute_flightDistance(events, mother, particles, true_vertex=true_vertex)
    events[f"DIRA_{mother}_true_vertex"] = vt.compute_DIRA(events, mother, particles, true_vertex=true_vertex)

    # ??? How to deal with intermediares ???

    # events['MOTHER_OWNPV_X'] = events['MOTHER_ORIGINVERTEX_Z']
    # events['MOTHER_OWNPV_Y'] = events['MOTHER_ORIGINVERTEX_Y']
    # events['MOTHER_OWNPV_Z'] = events['MOTHER_ORIGINVERTEX_Z']

    # events['INTERMEDIATE_TRUEENDVERTEX_X'] = events['MOTHER_TRUEENDVERTEX_X']
    # events['INTERMEDIATE_TRUEENDVERTEX_Y'] = events['MOTHER_TRUEENDVERTEX_Y']
    # events['INTERMEDIATE_TRUEENDVERTEX_Z'] = events['MOTHER_TRUEENDVERTEX_Z']

    # dist = vt.compute_intermediate_distance(events, intermediate, mother)
    # dist = np.asarray(dist)
    # print(f'fraction of intermediates that travel: {np.shape(dist[np.where(dist > 0)])[0] / np.shape(dist)[0]}')
    # dist[np.where(dist == 0)] = 1E-4
    # events[f"{intermediate}_FLIGHT"] = dist

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
    
    # events['B_plus_TRUEID'] = 
    events['K_plus_TRUEID'] = 321
    events['e_plus_TRUEID'] = 11
    events['e_minus_TRUEID'] = 11
    
    # Angles
    for particle in particles:
        events[f"angle_{particle}"] = t.compute_angle(events, mother, f"{particle}")

    # B P/PT
    
    events['B_plus_P'] = np.sqrt(events['B_plus_PX_TRUE']**2 + events['B_plus_PY_TRUE']**2 + events['B_plus_PZ_TRUE']**2)
    events['B_plus_PT'] = np.sqrt(events['B_plus_PX_TRUE']**2 + events['B_plus_PY_TRUE']**2)
    
    return events


events = extra_conditions(file)

output_file_path = "/users/zw21147/ResearchProject/datasets_mixed/mixed_Kee_newconditions.root"

data_dict = {col: events[col].values for col in events.columns}

with uproot.recreate(output_file_path) as f:
    f["DecayTree"] = data_dict


    