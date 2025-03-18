import ROOT
import random


input_file = "/users/zw21147/ResearchProject/datasets/combinatorial_select_Kuu.root"
output_train = "/users/zw21147/ResearchProject/datasets/split/train.root"
output_test = "/users/zw21147/ResearchProject/datasets/split/test.root"

file = ROOT.TFile.Open(input_file, "READ")
tree = file.Get("DecayTree")  # Replace with actual TTree name if different

train_file = ROOT.TFile(output_train, "RECREATE")
test_file = ROOT.TFile(output_test, "RECREATE")

# Create new trees for train and test
train_tree = tree.CloneTree(0) 
test_tree = tree.CloneTree(0)   


n_entries = tree.GetEntries()
indices = list(range(n_entries))
random.shuffle(indices)  # Shuffle indices

# Split index list (80% train, 20% test)
train_indices = indices[:int(0.8 * n_entries)]
test_indices = indices[int(0.8 * n_entries):]

for i in train_indices:
    tree.GetEntry(i)
    train_tree.Fill()

for i in test_indices:
    tree.GetEntry(i)
    test_tree.Fill()

# Write trees to files
train_file.Write()
test_file.Write()

train_file.Close()
test_file.Close()
file.Close()

print(f"Train file saved at: {output_train}")
print(f"Test file saved at: {output_test}")