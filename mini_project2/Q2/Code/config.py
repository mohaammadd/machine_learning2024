import torch

# Define activation functions and neurons as options for models
activations = ["ReLU", "ELU", "GELU"]
neurons = [16, 32, 64]

# Store all configurations
configs = []

# Define device based on availability of CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate configurations for each combination of neurons and activation functions
for n in neurons:
    for a in activations:
        configs.append({
            "num_features": 11,  # Update based on your dataset
            "num_classes": 4,  # Update based on your dataset
            "neurons": n,
            "activation": a,
            "optimizer_type": "SGD",  # Could also be Adam, etc.
            "learning_rate": 0.001,
            "model_name": f"model_{len(configs)+1}",
            #"device": device,  # Use the device (GPU or CPU)
        })

# Optional: If you'd like to save this configuration to a file for easy access, you could add:
# import json
# with open("configs.json", "w") as f:
#     json.dump(configs, f, indent=4)

