from training import train_individually_fair_subspace_robust_model
import tensorflow as tf

tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

import pandas as pd
from dummies import one_hot_integer_columns

df = pd.read_parquet("hf://datasets/cestwc/german-credit/data/train-00000-of-00001.parquet")
df = one_hot_integer_columns(df)

X_df = df.drop(['class'], axis=1)
y_df = df[['class']]


config = {
        'dataset_name': 'german',
        'sensitive_features': [x for x in X_df.columns if x.startswith('personal status and sex')],
        'training_goal': 'classification',
        'vanilla_epochs': 35,
        'fair_hyperplane_epochs': 20,
        'sensitive_batch_size': 64,
        'sensitive_reg': 0.02,
        'fair_epochs': 250,
        'fair_batch_size': 300,
        'reg': 0.005,
        'n_units': [64],
        'lr': 0.001,
        'debiased_training': True,
        'training_MILP': True,
        'epsilon': 0.2,
        'delta': 0.1,
        'lambda': 0.5,
        'training_opt_mode': 'milp',
        'training_verif_time_limit': None,
        # 'embedding': 0,
    }



_, _, model = train_individually_fair_subspace_robust_model( [''], [''], X_df, y_df, X_df, y_df, [X_df[config['sensitive_features']]], config, save_directory=None)

print(model.summary())

import os

def get_latest_modified_dir(base_path):
    # Get absolute paths of all subdirectories
    dirs = [
        os.path.join(base_path, d)
        for d in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, d))
    ]
    if not dirs:
        return None
    # Sort by last modified time
    latest_dir = max(dirs, key=os.path.getmtime)
    return latest_dir

latest = get_latest_modified_dir("saved_models")

tf.compat.v1.disable_eager_execution()
sess = tf.compat.v1.keras.backend.get_session()
sess.run(tf.compat.v1.global_variables_initializer())
model.load_weights(latest+'/model')

weights = model.get_weights()

import torch
torch_model = torch.hub.load('cat-claws/nn', 'simplecnn', convs = [], linears = [X_df.shape[1]] + config['n_units'], num_classes = 1)

with torch.no_grad():
    keras_linear_index = 0
    for layer in torch_model.layers:
        if isinstance(layer, torch.nn.Linear):
            W = weights[keras_linear_index]       # e.g. W0
            b = weights[keras_linear_index + 1]   # e.g. b0
            layer.weight.copy_(torch.tensor(W.T))  # Transpose Keras (in, out) → PyTorch (out, in)
            layer.bias.copy_(torch.tensor(b))
            keras_linear_index += 2

from sharpen import generate_serial

torch.save(torch_model.state_dict(), 'saved_models/' + config['dataset_name'] + '_' + '_'.join(str(n) for n in config['n_units']) + f"_{generate_serial()}.pt")
print("✅ Saved .pt")

print(torch_model)
