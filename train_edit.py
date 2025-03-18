from fast_vertex_quality.tools.config import read_definition, rd

### GPU test ###
import tensorflow as tf
import time

# Check TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# Check if TensorFlow can access GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Number of GPUs detected: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"GPU {i}: {gpu}")
else:
    print("No GPU detected. TensorFlow is running on CPU.")
###


from fast_vertex_quality.training_schemes.track_chi2 import trackchi2_trainer
from fast_vertex_quality.training_schemes.vertex_quality import vertex_quality_trainer
from fast_vertex_quality.testing_schemes.BDT import BDT_tester
import fast_vertex_quality.tools.data_loader as data_loader

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

from matplotlib.colors import LogNorm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score
import os

use_intermediate = False

rd.current_mse_raw = tf.convert_to_tensor(1.0)

### Directory setup to save model ###

# test_tag = 'NewConditions_mini'

# test_loc = f'test_runs_branches/{test_tag}/'
# try:
# 	os.mkdir(f'{test_loc}')
# 	os.mkdir(f'{test_loc}/networks')
# except:
# 	print(f"{test_loc} will be overwritten")
# rd.test_loc = test_loc

# rd.network_option = 'VAE'
# load_state = f"{test_loc}/networks/{test_tag}"

test_tag = 'plot_test'
test_loc = f'model_final_runs/{test_tag}/'

# Ensure all directories exist before proceeding
os.makedirs(test_loc, exist_ok=True)
os.makedirs(f'{test_loc}/networks', exist_ok=True)

rd.test_loc = test_loc
rd.network_option = 'VAE'
load_state = f"{test_loc}/networks/{test_tag}"

print(f"Directories ensured: {test_loc} and {test_loc}/networks")

# Create a dedicated directory for input distributions
input_dist_dir = os.path.join(test_loc, "input_distributions")
os.makedirs(input_dist_dir, exist_ok=True)

print(f"Input distributions will be saved in: {input_dist_dir}")


### Network Configuration ###

rd.latent = 10 # VAE latent dims

# rd.D_architecture=[int(512*1.5),int(1024*1.5),int(1024*1.5),int(512*1.5)]
# rd.G_architecture=[int(512*1.5),int(1024*1.5),int(1024*1.5),int(512*1.5)]

rd.D_architecture = [256]  
rd.G_architecture = [256]

# Experiment with:
# Multiple layers
# rd.D_architecture = [512, 1024, 512]  
# rd.G_architecture = [512, 1024, 512]

# # Different layer sizes
# rd.D_architecture = [256, 512, 512, 256]
# rd.G_architecture = [256, 512, 512, 256]

rd.beta = 1000.
rd.batch_size = 256

# rd.batch_size = 64

rd.conditions = [
	"B_plus_P",
	"B_plus_PT",
 
	"angle_K_plus",
	"angle_e_plus",
	"angle_e_minus",
 
	"K_plus_FLIGHT",
	"e_plus_FLIGHT",
	"e_minus_FLIGHT",
 
	# "B_plus_TRUEID"
	"K_plus_TRUEID",
	"e_plus_TRUEID",
	"e_minus_TRUEID",
 
	# Resampled from removed 0 
 	"B_plus_vtxX_TRUE",
	"B_plus_vtxY_TRUE",
	"B_plus_vtxZ_TRUE",
 
	"K_plus_vtxX_TRUE",
	"K_plus_vtxY_TRUE",
	"K_plus_vtxZ_TRUE",
 
	"e_plus_vtxX_TRUE",
	"e_plus_vtxY_TRUE",
	"e_plus_vtxZ_TRUE",
 
	"e_minus_vtxX_TRUE",
	"e_minus_vtxY_TRUE",
	"e_minus_vtxZ_TRUE",
]

rd.targets = [
	"B_plus_ENDVERTEX_CHI2",
	"B_plus_IPCHI2_OWNPV",
	"B_plus_FDCHI2_OWNPV",
	"B_plus_DIRA_OWNPV",
	"K_plus_IPCHI2_OWNPV",
	"K_plus_TRACK_CHI2NDOF",
	"e_minus_IPCHI2_OWNPV",
	"e_minus_TRACK_CHI2NDOF",
	"e_plus_IPCHI2_OWNPV",
	"e_plus_TRACK_CHI2NDOF",
	# "J_psi_1S_FDCHI2_OWNPV",
	# "J_psi_1S_IPCHI2_OWNPV",
]

rd.conditional_targets = []

rd.daughter_particles = ["K_plus", "e_plus", "e_minus"] # K e e
rd.mother_particle = 'B_plus'
rd.intermediate_particle = 'J_psi_1S'

### Data Loading ###

print(f"Loading data...")
training_data_loader = data_loader.load_data(
	[
		#"/users/zw21147/ResearchProject/datasets_mixed/mixed_Kee_newconditions.root",
		"/users/zw21147/ResearchProject/datasets/combinatorial_select_Kuu_renamed_resampled.root",
		
	],
	convert_to_RK_branch_names=True,
	conversions={'MOTHER':'B_plus', 'DAUGHTER1':'K_plus', 'DAUGHTER2':'e_plus', 'DAUGHTER3':'e_minus', 'INTERMEDIATE':'J_psi_1S'},
	# testing_frac=0.1
	testing_frac=0.1/20. * 2.
)

training_data_loader.add_missing_mass_frac_branch()

transformers = training_data_loader.get_transformers()

print(training_data_loader.shape())

# training_data_loader.reweight_for_training("fully_reco", weight_value=1., plot_variable='B_plus_M')
# training_data_loader.reweight_for_training("fully_reco", weight_value=50., plot_variable='B_plus_M')


# Commented out as no fully_reco available for mixed dataset, is it necessary for a combinatorial approach?
#training_data_loader.reweight_for_training("fully_reco", weight_value=100., plot_variable='B_plus_M')

print(f"Creating vertex_quality_trainer...")

trackchi2_trainer_obj = None

#training_data_loader.print_branches()

print("Plot conditions...")
training_data_loader.plot('conditions.pdf',rd.conditions)
print("Plot targets...")
training_data_loader.plot('targets.pdf',rd.targets)

quit()


### Network creation ###

vertex_quality_trainer_obj = vertex_quality_trainer(
	training_data_loader,
	trackchi2_trainer_obj,
	conditions=rd.conditions,
	targets=rd.targets,
	beta=float(rd.beta),
	latent_dim=rd.latent,
	batch_size=rd.batch_size,
	D_architecture=rd.D_architecture,
	G_architecture=rd.G_architecture,
	network_option=rd.network_option,
)

### DEBUGGING PLOTS ###

#Trained weights loaded
# vertex_quality_trainer_obj.load_state(tag=load_state)

# #Plot (script changed in fast_vertex_quality -> src -> training_schemes -> vertex_quality.py)
# vertex_quality_trainer_obj.make_plots(filename=f'plots.pdf', save_dir=f'{test_loc}/plots' , testing_file=["/users/zw21147/ResearchProject/datasets/combinatorial_select_Kuu_renamed.root"])
# print('Plots made')

# quit()

# ###


def test_with_ROC(training_data_loader_roc, vertex_quality_trainer_obj, it, last_BDT_distributions=None, tag='', weight=True):

	ROC_vars = [tar for tar in rd.targets if tar not in rd.conditional_targets]

	resample = False

	if weight and resample:
		X_test_data_all_pp = training_data_loader_roc.get_branches(
							rd.targets + rd.conditions + ['training_weight'], processed=True, option='testing'
						)
		X_test_data_all_pp = X_test_data_all_pp.sample(frac=1, weights=X_test_data_all_pp['training_weight'],replace=True)
		X_test_data_all_pp = X_test_data_all_pp.drop(columns=['training_weight'])
	else:
		X_test_data_all_pp = training_data_loader_roc.get_branches(
							rd.targets + rd.conditions, processed=True, option='testing'
						)

	print(f'test_with_ROC shape: {X_test_data_all_pp.shape}')

	images_true = np.asarray(X_test_data_all_pp[rd.targets])

	X_test_conditions = X_test_data_all_pp[rd.conditions]
	X_test_conditions = np.asarray(X_test_conditions)
	X_test_targets = X_test_data_all_pp[rd.targets]
	X_test_targets = np.asarray(X_test_targets)

	if rd.network_option == 'VAE':

		z, z_mean, z_log_var = np.asarray(vertex_quality_trainer_obj.encoder([X_test_targets, X_test_conditions]))
		images_cheating = np.asarray(vertex_quality_trainer_obj.decoder([z, X_test_conditions]))

		gen_noise = np.random.normal(0, 1, (np.shape(X_test_conditions)[0], rd.latent))
		images = np.asarray(vertex_quality_trainer_obj.decoder([gen_noise, X_test_conditions]))

	images_dict = {}
	for i in range(len(ROC_vars)):
		images_dict[ROC_vars[i]] = images[:,i]
	images = training_data_loader_roc.post_process(pd.DataFrame(images_dict))

	images_true_dict = {}
	for i in range(len(ROC_vars)):
		images_true_dict[ROC_vars[i]] = images_true[:,i]
	images_true = training_data_loader_roc.post_process(pd.DataFrame(images_true_dict))

	if rd.network_option == 'VAE':
		images_cheating_dict = {}
		for i in range(len(ROC_vars)):
			images_cheating_dict[ROC_vars[i]] = images_cheating[:,i]
		images_cheating = np.squeeze(training_data_loader_roc.post_process(pd.DataFrame(images_cheating_dict)))

	clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4)

	bdt_train_size = int(np.shape(images)[0]/2)

	real_training_data = np.squeeze(images_true[:bdt_train_size])

	real_test_data = np.squeeze(images_true[bdt_train_size:])

	fake_training_data = np.squeeze(images[:bdt_train_size])

	fake_test_data = np.squeeze(images[bdt_train_size:])

	real_training_labels = np.ones(bdt_train_size)

	fake_training_labels = np.zeros(bdt_train_size)

	total_training_data = np.concatenate((real_training_data, fake_training_data))

	total_training_labels = np.concatenate((real_training_labels, fake_training_labels))

	clf.fit(total_training_data, total_training_labels)

	out_real = clf.predict_proba(real_test_data)

	out_fake = clf.predict_proba(fake_test_data)
	
	if rd.network_option == 'VAE':
		out_cheat = clf.predict_proba(images_cheating)
		plt.hist([out_real[:,1],out_fake[:,1], out_cheat[:,1]], bins = 100,label=['real','gen','gen - cheat'], histtype='step', color=['tab:red','tab:blue','tab:green'], density=True)
	else:
		plt.hist([out_real[:,1],out_fake[:,1]], bins = 100,label=['real','gen'], histtype='step', color=['tab:red','tab:blue'], density=True)
	if last_BDT_distributions:
		plt.hist(last_BDT_distributions, bins = 100,alpha=0.5, histtype='step', color=['tab:red','tab:blue'], density=True)
	last_BDT_distributions = [out_real[:,1],out_fake[:,1]]
	plt.xlabel('Output of BDT')
	plt.legend(loc='upper right')
	plt.savefig(f'{test_loc}BDT_out_{it}_{rd.network_option}{tag}.png', bbox_inches='tight')
	plt.close('all')

	importance = clf.feature_importances_
	for idx, target in enumerate(ROC_vars):
		print(f'{target}:\t {importance[idx]/np.amax(importance):.2f}')

	ROC_AUC_SCORE_curr = roc_auc_score(np.append(np.ones(np.shape(out_real[:,1])),np.zeros(np.shape(out_fake[:,1]))),np.append(out_real[:,1],out_fake[:,1]))

	y_true = np.append(np.ones(np.shape(out_real[:, 1])), np.zeros(np.shape(out_fake[:, 1])))
	y_scores = np.append(out_real[:, 1], out_fake[:, 1])
	fpr, tpr, thresholds = roc_curve(y_true, y_scores)

	# Plot ROC curve
	plt.figure()
	plt.plot(fpr, tpr, label=f'ROC curve (AUC = {ROC_AUC_SCORE_curr:.2f})', color='blue')
	plt.plot([0, 1], [0, 1], 'k--', label='Chance level')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver Operating Characteristic (ROC) Curve')
	plt.legend(loc='best')
	plt.savefig(f'{test_loc}ROC_curve_{it}_{rd.network_option}{tag}.png', bbox_inches='tight')
	plt.close()

	print(ROC_AUC_SCORE_curr)
	return ROC_AUC_SCORE_curr, last_BDT_distributions


# def save_input_distributions(training_data_loader, iteration):
#     """
#     Function to save input distributions to the specified directory.
#     """
#     input_data = training_data_loader.get_branches(rd.conditions, processed=True, option='testing')

#     for feature in rd.conditions:
#         plt.figure()
#         plt.hist(input_data[feature], bins=50)
#         plt.xlabel(feature)
#         plt.ylabel("Density")
#         plt.title(f"Input Distribution: {feature}")

#         save_path = os.path.join(input_dist_dir, f"{feature}_iter_{iteration}.png")
#         plt.savefig(save_path, bbox_inches='tight')
#         plt.close()

#     print(f"Saved input distributions for iteration {iteration}")
# save_input_distributions(training_data_loader, iteration=0)

# quit()

### Training / Testing / Saving ###

steps_for_plot = 5000 # number of training iterations between plots/checkpoints

# Initial evaluation
ROC_collect = np.empty((0,2))
ROC_collect_Kee = np.empty((0,2))
ROC_collect = np.append(ROC_collect, [[0, 1.]], axis=0)
ROC_collect_Kee = np.append(ROC_collect_Kee, [[0, 1.]], axis=0)

chi2_collect = np.empty((0,3))
chi2_collect_best = np.empty((0,3))

vertex_quality_trainer_obj.train(steps=steps_for_plot)
vertex_quality_trainer_obj.save_state(tag=load_state)

ROC_AUC_SCORE_curr, last_BDT_distributions = test_with_ROC(training_data_loader, vertex_quality_trainer_obj, 0)
ROC_collect = np.append(ROC_collect, [[0, ROC_AUC_SCORE_curr]], axis=0)

start_time = time.time()  # Record the start time

# Infinite training, creates and outputs progress plots
for i in range(int(1E30)):

	vertex_quality_trainer_obj.train_more_steps(steps=steps_for_plot)

	### Elapsed time ###
	elapsed_time = time.time() - start_time
	hours = int(elapsed_time // 3600)
	minutes = int((elapsed_time % 3600) // 60)
	seconds = int(elapsed_time % 60)
	print(f"Elapsed time: {hours}h {minutes}m {seconds}s")
	###
	
	ROC_AUC_SCORE_curr, last_BDT_distributions = test_with_ROC(training_data_loader, vertex_quality_trainer_obj, i+1, last_BDT_distributions=last_BDT_distributions)
	ROC_collect = np.append(ROC_collect, [[i+1, ROC_AUC_SCORE_curr]], axis=0)

	# Save actual inputs going into the VAE
	# vertex_quality_trainer_obj.save_vae_input_distributions(iteration=i+1, save_dir=os.path.join(test_loc, "input_distributions"))
 
	# Save the state of the network 
	vertex_quality_trainer_obj.save_state(tag=load_state)
 
	# New plotting functions 
	print('Loading and plotting...')
	vertex_quality_trainer_obj.load_state(tag=load_state)
	vertex_quality_trainer_obj.make_plots(filename=f'plots_{i+1}.pdf', save_dir=f'{test_loc}/plots' , testing_file=["/users/zw21147/ResearchProject/datasets/combinatorial_select_Kuu_renamed_resampled.root"])
	# vertex_quality_trainer_obj.make_inverse_plots(filename=f'inverse_plots_{i+1}.pdf', save_dir=f'{test_loc}/plots' , testing_file=["/users/zw21147/ResearchProject/datasets/combinatorial_select_Kuu_renamed_resampled.root"])
	print('Plots made')

	plt.plot(ROC_collect[:,1])
	plt.savefig(f'{test_loc}Progress_ROC_{rd.network_option}')
	plt.close('all')

