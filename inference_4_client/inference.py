import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import csv

import torch
import numpy as np
from Models.model import GenericModel, load_subject_data
from torch.utils.data import DataLoader, TensorDataset

# Configuration
MODEL_PATH = "./global_models/global_model_round_10.pth"  # Update to the correct path if necessary
DATA_DIRECTORY = "processed_features"
SUBJECT_ID = 30

def load_model(model_path, input_size=5):
    """Load the stored model from the given path."""
    # Initialize the model with the architecture defined in model.py
    model = GenericModel(input_size=input_size)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found.")
    
    # Load the state dict
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {model_path}")
    return model

def prepare_inference_data(subject_id, data_directory=DATA_DIRECTORY):
    """Load and prepare data for inference."""
    file_path = os.path.join(data_directory, f"subject_{subject_id}.npy")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file {file_path} not found.")
    
    # Load the subject data
    X, y = load_subject_data(subject_id, data_directory)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)  # Use a fixed batch size
    print(f"Data for subject {subject_id} loaded successfully.")
    return dataloader

def run_inference(model, dataloader):
    """Run inference on the provided data and model."""
    criterion = torch.nn.MSELoss()  # Assuming Mean Squared Error as the loss function
    total_loss = 0.0
    total_samples = 0
    all_predictions = []
    all_ground_truths = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item() * X_batch.size(0)  # Accumulate total loss
            total_samples += X_batch.size(0)

            # Store predictions and ground truth
            all_predictions.append(outputs.numpy())
            all_ground_truths.append(y_batch.numpy())

    # Compute average loss
    avg_loss = total_loss / total_samples
    print(f"Average Inference Loss: {avg_loss:.4f}")

    # Concatenate predictions and ground truths
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_ground_truths = np.concatenate(all_ground_truths, axis=0)
    return avg_loss, all_predictions, all_ground_truths

if __name__ == "__main__":
    # Load the model
    model = load_model(MODEL_PATH)

    # Prepare the data for subject 30
    dataloader = prepare_inference_data(SUBJECT_ID)

    # Run inference
    avg_loss, predictions, ground_truths = run_inference(model, dataloader)

    output_folder = "inference_4_client"
    # Save predictions and ground truths to a CSV file
    output_file = os.path.join(output_folder, "inference_results_subject_30.csv")

    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Prediction", "Ground Truth"])  # Write header
        for pred, truth in zip(predictions, ground_truths):
            plain_pred = [float(p) for p in pred]
            plain_truth = [float(t) for t in truth]
            writer.writerow([plain_pred, plain_truth])

    print(f"Inference results saved to {output_file}")
