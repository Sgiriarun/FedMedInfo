import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, ConcatDataset
from sklearn.preprocessing import StandardScaler

# Neural Network Model
class GenericModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=2):
        super(GenericModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)


# Load Data for a Single Subject
def load_subject_data(subject_id, data_directory="processed_features"):
    file_path = os.path.join(data_directory, f"subject_{subject_id}.npy")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data for subject {subject_id} not found.")

    data = np.load(file_path)
    X = data[:, :5]
    y = data[:, 5:]

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# Load Data for Multiple Subjects
def load_multiple_subjects_data(subject_ids, data_directory="processed_features"):
    datasets = [TensorDataset(*load_subject_data(sid, data_directory)) for sid in subject_ids]
    return ConcatDataset(datasets)


# Train Model
def train_model(model, train_loader, epochs=5, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")


# Evaluate Model
def evaluate_model(model, test_loader):
    criterion = nn.MSELoss()
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item()
    print(f"Test Loss: {test_loss / len(test_loader):.4f}")
    return test_loss / len(test_loader)
