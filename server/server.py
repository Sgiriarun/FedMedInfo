import sys
import os
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.server.server import ServerConfig
from flwr.common import parameters_to_ndarrays
import torch
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from flwr.common import EvaluateRes, Scalar
# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Models.model import GenericModel

# Global variables for tracking metrics
FLloss_list = []
FLacc_list = []
trainAcc_list = [[] for _ in range(10)]  # Assuming up to 10 clients
trainLoss_list = [[] for _ in range(10)]
Total_dev = 10  # Total number of devices/clients
Total_rnds = 10  # Total number of rounds

# Custom Strategy
class CustomStrategy(FedAvg):
    def __init__(self, net, save_rnd=2, save_path="./global_models", **kwargs):
        super().__init__(**kwargs)
        self.net = net
        self.save_rnd = save_rnd
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.best_acc = 0.0
        self.best_rnd = 0

    def aggregate_fit(self, rnd, results, failures):
        """Aggregate model parameters and save the checkpoint."""
        aggregated_parameters_tuple = super().aggregate_fit(rnd, results, failures)
        aggregated_parameters, _ = aggregated_parameters_tuple

        if rnd % self.save_rnd == 0 and aggregated_parameters is not None:
            print(f"Saving global model for round {rnd}...")
            aggregated_weights = parameters_to_ndarrays(aggregated_parameters)
            state_dict = OrderedDict(zip(self.net.state_dict().keys(), [torch.tensor(w) for w in aggregated_weights]))
            self.net.load_state_dict(state_dict)
            torch.save(self.net.state_dict(), self.save_path / f"global_model_round_{rnd}.pth")
            print(f"Model saved for round {rnd}.")

        return aggregated_parameters_tuple

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses and track metrics."""
        if not results:
            return None, {}
        # Skip if failures exist and are not acceptable
        if failures and not self.accept_failures:
            return None, {}

        # Collect metrics from clients
        test_num = [r[1].num_examples for r in results]
        test_loss = [r[1].loss for r in results]
        test_acc = [r[1].metrics['test_acc'] for r in results]
        train_loss = [r[1].metrics['train_loss'] for r in results]
        train_acc = [r[1].metrics['train_acc'] for r in results]
        cid_list = [r[1].metrics['cid'] for r in results]

        # Print aggregated metrics
        print(f"Round {rnd:03d} \t FL_loss: {test_loss[0]:.4f} \t FL_acc: {test_acc[0]:.4f} \t Test images: {test_num[0]}")

        # Record metrics
        FLloss_list.append(round(test_loss[0], 4))
        FLacc_list.append(round(test_acc[0], 4))
        for i in range(len(cid_list)):
            trainAcc_list[cid_list[i]-1].append(round(train_acc[i], 4))
            trainLoss_list[cid_list[i]-1].append(round(train_loss[i], 4))

        # Track the best accuracy
        if rnd == 1:
            self.best_acc = 0.0
            self.best_rnd = 0
        if self.best_acc < test_acc[0]:
            self.best_acc = test_acc[0]
            self.best_rnd = rnd
        if rnd == Total_rnds:
            print(f"Best FL accuracy: {self.best_acc:.4f} on round {self.best_rnd}")

        # Call the base class method to finalize aggregation
        return super().aggregate_evaluate(rnd, results, failures)


if __name__ == "__main__":
    input_size = 5
    model = GenericModel(input_size=input_size)

    # Define the custom strategy
    strategy = CustomStrategy(
        net=model,
        save_rnd=2,
        save_path="./global_models",
        min_fit_clients=2,
        min_available_clients=2,
        fraction_fit=1.0,
    )

    # Server configuration
    config = ServerConfig(num_rounds=Total_rnds)

    # Start the Flower server
    print("Starting the server...")
    fl.server.start_server(
        server_address="0.0.0.0:9090",
        config=config,
        strategy=strategy
    )
