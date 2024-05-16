from mujoco_physics import HopperPhysics
import torch
import os


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# #generating the trajectories and loading the .pt file
# dataset_obj = HopperPhysics(root='data', download=False, generate=True, device = device)

# root = r"C:\Users\msi\Desktop\ECE-228\Project\latent_ode\data\HopperPhysics\training.pt"

# dataset =torch.load(root)

# print(type(dataset))
# print(dataset.shape)
# print("First sample, first time step:", dataset[0, 0, :])

# output_dir = 'hopper_imgs'
# os.makedirs(output_dir, exist_ok=True)

#to get the frames saved for vizualization inside hopper_imgs
hopper = HopperPhysics(root='data', download=False, generate=False)

dataset = hopper.get_dataset()
traj_index = 23   # Index of the trajectory you want to visualize
trajectory = dataset[traj_index]
hopper.visualize(trajectory, plot_name=f'traj_{traj_index}', dirname=output_dir)


# # the visual environment I showed first during the meeting, with the mouse.
# from dm_control import suite
# from dm_control import viewer
# import numpy as np

# env = suite.load(domain_name="humanoid", task_name="run")
# action_spec = env.action_spec()

# # Define a uniform random policy.
# def random_policy(time_step):
#   del time_step  # Unused.
#   return np.random.uniform(low=action_spec.minimum,
#                            high=action_spec.maximum,
#                            size=action_spec.shape)

# # Launch the viewer application.
# viewer.launch(env, policy=random_policy)