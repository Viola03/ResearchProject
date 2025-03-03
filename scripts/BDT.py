#!/usr/bin/env python
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import tensorflow as tf
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score

# Import your configuration and trainer classes
from fast_vertex_quality.tools.config import read_definition, rd
from fast_vertex_quality.training_schemes.vertex_quality import vertex_quality_trainer
import fast_vertex_quality.tools.data_loader as data_loader

# -------------------------------
# Load configuration and transformers
# -------------------------------
config_file = "inference/example/models/vertexing_configs.pkl"
transformers_file = "inference/example/models/vertexing_transfomers.pkl"

with open(config_file, "rb") as f:
    configs = pickle.load(f)
with open(transformers_file, "rb") as f:
    transformers = pickle.load(f)

# For example, assume configs contains:
#   configs["conditions"], configs["targets"], configs["latent_dim"]

# -------------------------------
# Load test data
# -------------------------------

test_file = "path/to/your/test_file.root"  
test_data_loader = data_loader.load_data(
    test_file,
    transformers=transformers,
    convert_to_RK_branch_names=True,
    conversions={'MOTHER':'B_plus', 'DAUGHTER1':'K_plus', 'DAUGHTER2':'e_plus', 'DAUGHTER3':'e_minus', 'INTERMEDIATE':'J_psi_1S'}
)

test_data_loader.add_missing_mass_frac_branch()

#subset of N events for inference
N = 10000
test_data_loader.select_randomly(Nevents=N)

#processed space
conditions_names = configs["conditions"]
targets_names = configs["targets"]
X_test_conditions = test_data_loader.get_branches(conditions_names, processed=True)
X_test_targets = test_data_loader.get_branches(targets_names, processed=True)
X_test_conditions = np.asarray(X_test_conditions[conditions_names])
X_test_targets = np.asarray(X_test_targets[targets_names])

# -------------------------------
# Load the trained model
# -------------------------------

trained_state = "inference/example/models/vertexing_state"  # update as needed
# Initialize the trainer with load_config option
vertex_trainer = vertex_quality_trainer(load_config=trained_state)
vertex_trainer.load_state(tag=trained_state)

# -------------------------------
# Generate predictions using the trained model
# -------------------------------

# For a VAE, generated outputs from random latent noise and conditions.
if rd.network_option == 'VAE':
    # Generate noise for the latent space
    gen_noise = np.random.normal(0, 1, (X_test_conditions.shape[0], rd.latent))
    # Generate "fake" outputs
    generated_images = vertex_trainer.decoder.predict([gen_noise, X_test_conditions])
    # For "real" outputs, use the true targets from the test data
    real_images = X_test_targets
elif rd.network_option in ['GAN', 'WGAN']:
    gen_noise = np.random.normal(0, 1, (X_test_conditions.shape[0], rd.latent))
    generated_images = vertex_trainer.generator.predict([gen_noise, X_test_conditions])
    real_images = X_test_targets
else:
    raise ValueError("Unknown network option")

# Optionally, apply any post-processing that was used during training.
# Here we assume test_data_loader.post_process takes a DataFrame and returns one.
df_generated = pd.DataFrame({name: generated_images[:, i] for i, name in enumerate(targets_names)})
df_real = pd.DataFrame({name: real_images[:, i] for i, name in enumerate(targets_names)})
df_generated = test_data_loader.post_process(df_generated)
df_real = test_data_loader.post_process(df_real)
generated_images = df_generated.to_numpy()
real_images = df_real.to_numpy()

# -------------------------------
# Prepare data for the BDT classifier
# -------------------------------
n_samples = generated_images.shape[0]
split_index = n_samples // 2

# Split into training and testing halves
real_train = real_images[:split_index, :]
real_test = real_images[split_index:, :]
fake_train = generated_images[:split_index, :]
fake_test = generated_images[split_index:, :]

# Combine training data and labels
X_train = np.concatenate((real_train, fake_train), axis=0)
y_train = np.concatenate((np.ones(real_train.shape[0]), np.zeros(fake_train.shape[0])), axis=0)

# Combine test data and labels
X_test = np.concatenate((real_test, fake_test), axis=0)
y_test = np.concatenate((np.ones(real_test.shape[0]), np.zeros(fake_test.shape[0])), axis=0)

# -------------------------------
# Train a BDT and compute the ROC curve
# -------------------------------
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4)
clf.fit(X_train, y_train)

# Get predicted probabilities for the positive class
y_scores = clf.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_scores)
fpr, tpr, thresholds = roc_curve(y_test, y_scores)

# -------------------------------
# Plot and save the ROC curve
# -------------------------------
roc_plot_path = "inference_ROC_curve.png"
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', color='blue')
plt.plot([0, 1], [0, 1], 'k--', label='Chance level')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='best')
plt.savefig(roc_plot_path, bbox_inches='tight')
plt.show()

print("ROC AUC:", roc_auc)
print("ROC curve saved to:", roc_plot_path)
