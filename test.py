from lib.parse_datasets import parse_datasets
import torch
import lib.utils as utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

args = Args(
    n=10000,  # Size of the dataset
    niters=100,
    lr=1e-2,  # Starting learning rate
    batch_size=50,
    viz=False,  # Show plots while training
    save='experiments/',  # Path to save checkpoints
    load=None,  # ID of the experiment to load for evaluation. If None, run a new experiment
    random_seed=1991,  # Random seed
    dataset='hopper',  # Dataset to load
    sample_tp=None,  # Number of time points to sub-sample
    cut_tp=None,  # Cut out the section of the timeline
    quantization=0.1,  # Quantization on the physionet dataset
    latent_ode=True,  # Run Latent ODE seq2seq model
    z0_encoder='odernn',  # Type of encoder for Latent ODE model
    classic_rnn=False,  # Run RNN baseline
    rnn_cell="gru",  # RNN Cell type
    input_decay=False,  # For RNN: use the input that is the weighted average of empirical mean and previous value
    ode_rnn=False,  # Run ODE-RNN baseline
    rnn_vae=False,  # Run RNN baseline: seq2seq model with sampling of the h0 and ELBO loss
    latents=6,  # Size of the latent state
    rec_dims=10,  # Dimensionality of the recognition model
    rec_layers=1,  # Number of layers in ODE func in recognition ODE
    gen_layers=1,  # Number of layers in ODE func in generative ODE
    units=100,  # Number of units per layer in ODE func
    gru_units=100,  # Number of units per layer in each of GRU update networks
    poisson=False,  # Model poisson-process likelihood for the density of events in addition to reconstruction
    classif=False,  # Include binary classification loss
    linear_classif=False,  # Use a linear classifier instead of 1-layer NN
    extrap=False,  # Set extrapolation mode
    timepoints=100,  # Total number of time-points
    max_t=5.0,  # Subsample points in the interval [0, args.max_t]
    noise_weight=0.01  # Noise amplitude for generated trajectories
)

data_obj = parse_datasets(args, device)

num_batches = data_obj["n_train_batches"]
# print(type(num_batches)) #int
print(num_batches)

batch_dict = utils.get_next_batch(data_obj["train_dataloader"])
# print(type(batch_dict))# dict
print(len(batch_dict))

x= data_obj["test_dataloader"]
# print(type(x))#generator
print(x)

y= data_obj["train_dataloader"]
# print(type(y))#generator
print(y)

test_dict = utils.get_next_batch(data_obj["test_dataloader"])
# print(type(test_dict)) #dict

