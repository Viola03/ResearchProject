import ROOT
import random


input_file = "/users/zw21147/ResearchProject/datasets/combinatorial_select_Kuu_renamed_resampled.root"
output_train = "/users/zw21147/ResearchProject/datasets/split/train.root"
output_test = "/users/zw21147/ResearchProject/datasets/split/validation.root"

file = ROOT.TFile.Open(input_file, "READ")
tree = file.Get("DecayTree")  # Replace with actual TTree name if different

n_entries = tree.GetEntries()
indices = list(range(n_entries))
random.shuffle(indices)  # Shuffle indices

# Split index list (90% train/test handled by the train_edit script, 10% validation)
train_indices = indices[:int(0.9 * n_entries)]
test_indices = indices[int(0.9 * n_entries):]

#Train
train_file = ROOT.TFile(output_train, "RECREATE")
train_file.cd()  # Make train_file the current directory
train_tree = tree.CloneTree(0)

for i in train_indices:
    tree.GetEntry(i)
    train_tree.Fill()
    
# train_tree.Write()
train_tree.Write("", ROOT.TObject.kOverwrite)
train_file.Close()

#Validation
test_file = ROOT.TFile(output_test, "RECREATE")
test_file.cd()   # Make test_file the current directory
test_tree = tree.CloneTree(0) 

for i in test_indices:
    tree.GetEntry(i)
    test_tree.Fill()

test_tree.Write()
test_file.Close()

print(f"Train file saved at: {output_train}")
print(f"Validation file saved at: {output_test}")