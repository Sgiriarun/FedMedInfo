import argparse
import flwr as fl
import torch
from torch.utils.data import DataLoader, TensorDataset
from Models.model import GenericModel, load_subject_data, load_multiple_subjects_data, train_model, evaluate_model
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters



if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument("--mode", type=str, choices=["subject_dependent", "subject_independent"], required=True,
                        help="Mode of training: 'subject_dependent' or 'subject_independent'")
    parser.add_argument("--subject_ids", type=int, nargs="+", required=True,
                        help="List of subject IDs for training. For subject-dependent, provide one ID.")
    parser.add_argument("--server_address", type=str, default="0.0.0.0:9090",
                        help="Server address (default: 0.0.0.0:9090)")
    args = parser.parse_args()

    # Determine client_id
    if args.mode == "subject_dependent":
        if len(args.subject_ids) != 1:
            raise ValueError("For subject-dependent mode, provide exactly one subject ID.")
        client_id = args.subject_ids[0]  # Use the subject number as client_id
        print(f"Client ID: {client_id} (Subject Dependent)")
        print(f"Loading data for subject-dependent training (Subject ID: {client_id})...")
        X, y = load_subject_data(client_id)
        dataset = TensorDataset(X, y)
    elif args.mode == "subject_independent":
        if len(args.subject_ids) < 2:
            raise ValueError("For subject-independent mode, provide at least two subject IDs.")
        # Generate a unique client_id using the list of subject IDs
        client_id = hash(tuple(sorted(args.subject_ids))) % 1_000_000  # Shorten hash for readability
        print(f"Client ID: {client_id} (Subject Independent)")
        print(f"Loading data for subject-independent training (Subject IDs: {args.subject_ids})...")
        dataset = load_multiple_subjects_data(args.subject_ids)
    else:
        raise ValueError("Invalid mode. Choose 'subject_dependent' or 'subject_independent'.")

    # Create data loaders
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Initialize model
    input_size = 5  # Number of features
    model = GenericModel(input_size=input_size)

    # Define Flower Client
    class FlowerClient(fl.client.NumPyClient):
        def __init__(self, client_id, model, train_loader, test_loader):
            self.client_id = client_id
            self.model = model
            self.train_loader = train_loader
            self.test_loader = test_loader

        def get_parameters(self, config):
            """Return model parameters as a list of NumPy arrays."""
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

        def set_parameters(self, parameters):
            """Set model parameters received from the server."""
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            self.model.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            """Train the model locally."""
            self.set_parameters(parameters)
            print(f"Client {self.client_id}: Starting local training...")
            train_model(self.model, self.train_loader, epochs=20)
            return self.get_parameters(config), len(self.train_loader.dataset), {}

        def evaluate(self, parameters, config):
            """Evaluate the model locally."""
            self.set_parameters(parameters)

            # Test set evaluation
            total_correct = 0
            total_samples = 0
            total_loss = 0.0

            criterion = torch.nn.MSELoss()  # Use the same loss as in training
            self.model.eval()
            with torch.no_grad():
                for X_batch, y_batch in self.test_loader:
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    total_loss += loss.item() * X_batch.size(0)

                    # Calculate accuracy for regression within tolerance
                    correct = torch.isclose(outputs, y_batch, atol=0.1).sum().item()
                    total_correct += correct
                    total_samples += y_batch.numel()

            # Calculate test loss and accuracy
            avg_loss = total_loss / total_samples
            test_acc = total_correct / total_samples

            # Train set evaluation (for train_loss and train_acc)
            train_loss = 0.0
            train_total_correct = 0
            train_total_samples = 0
            with torch.no_grad():
                for X_batch, y_batch in self.train_loader:
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    train_loss += loss.item() * X_batch.size(0)

                    correct = torch.isclose(outputs, y_batch, atol=0.1).sum().item()
                    train_total_correct += correct
                    train_total_samples += y_batch.numel()

            avg_train_loss = train_loss / train_total_samples
            train_acc = train_total_correct / train_total_samples

            # Use self.client_id for Client ID
            print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {test_acc:.4f}")
            print(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
            print(f"Client ID: {self.client_id}")

            # Return evaluation metrics
            return float(avg_loss), len(self.test_loader.dataset), {
                "test_acc": test_acc,
                "train_loss": avg_train_loss,
                "train_acc": train_acc,
                "cid": self.client_id  # Return self.client_id
            }



    # Start Flower client
    print(f"Starting Flower client with mode '{args.mode}'...")
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=FlowerClient(client_id, model, train_loader, test_loader)
    )


# """
# For Subject-Dependent:
# python client.py --mode subject_dependent --subject_ids 1

# For Subject-Independent:
# python client.py --mode subject_independent --subject_ids 1 2 3
# """