import pandas as pd
import matplotlib.pyplot as plt

# Load the metrics from CSV
df = pd.read_csv("./metrics.csv")

# Plot Federated Learning Loss
plt.figure()
df.groupby("Round")["FL_Loss"].mean().plot(marker="o")
plt.title("Federated Learning Loss")
plt.xlabel("Round")
plt.ylabel("Loss")
plt.grid()
plt.savefig("fl_loss.png")
plt.show()

# Plot Federated Learning Accuracy
plt.figure()
df.groupby("Round")["FL_Accuracy"].mean().plot(marker="o")
plt.title("Federated Learning Accuracy")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.grid()
plt.savefig("fl_accuracy.png")
plt.show()

# Plot Train Loss for Each Client
plt.figure()
for cid in df["Client_ID"].unique():
    client_data = df[df["Client_ID"] == cid]
    plt.plot(client_data["Round"], client_data["Train_Loss"], marker="o", label=f"Client {cid}")
plt.title("Train Loss per Client")
plt.xlabel("Round")
plt.ylabel("Train Loss")
plt.legend()
plt.grid()
plt.savefig("train_loss.png")
plt.show()

# Plot Train Accuracy for Each Client
plt.figure()
for cid in df["Client_ID"].unique():
    client_data = df[df["Client_ID"] == cid]
    plt.plot(client_data["Round"], client_data["Train_Accuracy"], marker="o", label=f"Client {cid}")
plt.title("Train Accuracy per Client")
plt.xlabel("Round")
plt.ylabel("Train Accuracy")
plt.legend()
plt.grid()
plt.savefig("train_accuracy.png")
plt.show()
