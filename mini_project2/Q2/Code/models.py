import torch
import torch.nn as nn
import torch.optim as optim

class OneLayerNetwork(nn.Module):
    def __init__(self, num_features, num_classes, neurons, activation, optimizer_type, learning_rate, model_name):
        super(OneLayerNetwork, self).__init__()
        self.config = {
            "num_features": num_features,
            "num_classes": num_classes,
            "neurons": neurons,
            "activation": activation,
            "optimizer_type": optimizer_type,
            "learning_rate": learning_rate,
            "model_name": model_name,
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.build_model().to(self.device)
        self.optimizer = self.get_optimizer()
        self.loss_fn = nn.CrossEntropyLoss()

    def build_model(self):
        # Define the layers using nn.Sequential and the specified activation function
        return nn.Sequential(
            nn.Linear(self.config["num_features"], self.config["neurons"]),
            getattr(nn, self.config["activation"])(),  # Get activation function by name
            nn.Linear(self.config["neurons"], self.config["num_classes"])
        )

    def get_optimizer(self):
        if self.config["optimizer_type"] == "SGD":
            return optim.SGD(self.model.parameters(), lr=self.config["learning_rate"], momentum=0.9)
        elif self.config["optimizer_type"] == "Adam":
            return optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])
        else:
            raise ValueError(f"Unsupported optimizer: {self.config['optimizer_type']}")

    def forward(self, x):
        # Since we're using nn.Sequential, we don't need to manually define forward pass
        return self.model(x)

