import joblib
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import torch
from sklearn.feature_extraction.text import TfidfTransformer
from torch.utils.data import DataLoader


# Visualize feature importance (dummy example for visualization) and save as SVG + show the plot
def visualize_feature_importance(TOKENS, FEATURE_IMPORTANCE, FILENAME="Plot.svg"):
    # Limit the number of tokens to visualize
    TOKENS = TOKENS[:1000]
    FEATURE_IMPORTANCE = FEATURE_IMPORTANCE[:1000]

    plt.figure(figsize=(len(TOKENS) * 0.5, 6))
    sns.barplot(x=TOKENS, y=FEATURE_IMPORTANCE, palette="coolwarm", hue=TOKENS, legend=False)
    plt.title("Feature Importance")
    plt.xlabel("Tokens")
    plt.ylabel("Importance")
    plt.xticks(rotation=45)
    plt.savefig(FILENAME, format="svg")
    plt.show()  # Show the plot interactively
    plt.close()  # Close the plot to release memory


# Function to visualize the loss landscape as an interactive 3D object
def plot_loss_landscape_3d(MODEL, DATA_LOADER, CRITERION, GRID_SIZE=200, EPSILON=0.01, FILENAME="Plot.html"):
    MODEL.eval()  # Set model to evaluation mode
    param = next(MODEL.parameters())  # Use the first parameter for landscape perturbations
    param_flat = param.view(-1)

    # Define perturbation directions u and v
    u = torch.randn_like(param_flat).view(param.shape).to(param.device)
    v = torch.randn_like(param_flat).view(param.shape).to(param.device)

    # Normalize perturbations
    u = EPSILON * u / torch.norm(u)
    v = EPSILON * v / torch.norm(v)

    # Create grid
    x = np.linspace(-1, 1, GRID_SIZE)
    y = np.linspace(-1, 1, GRID_SIZE)
    loss_values = np.zeros((GRID_SIZE, GRID_SIZE))

    # Iterate through the grid to compute losses
    for i, dx in enumerate(x):
        for j, dy in enumerate(y):
            param.data += dx * u + dy * v  # Apply perturbation
            loss = 0

            # Compute loss for all batches in data loader
            for batch in DATA_LOADER:
                inputs, targets = batch
                inputs = inputs.to(param.device)
                targets = targets.to(param.device)
                outputs = MODEL(inputs)
                loss += CRITERION(outputs, targets).item()

            loss_values[i, j] = loss  # Store the loss
            param.data -= dx * u + dy * v  # Revert perturbation

    # Create a meshgrid for plotting
    X, Y = np.meshgrid(x, y)

    # Plot the 3D surface using Plotly
    fig = go.Figure(data=[go.Surface(z=loss_values, x=X, y=Y, colorscale="Viridis")])
    fig.update_layout(
        title="Loss Landscape (Interactive 3D)",
        scene=dict(
            xaxis_title="Perturbation in u",
            yaxis_title="Perturbation in v",
            zaxis_title="Loss",
        ),
    )

    # Save as an interactive HTML file
    fig.write_html(FILENAME)
    print(f"3D loss landscape saved as {FILENAME}")


# Example of DataLoader for loss landscape (dummy dataset for visualization)
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        self.data = torch.randn(num_samples, 10000)  # Increased number of features
        self.labels = torch.randint(0, 2, (num_samples,))  # Binary labels

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


if __name__ == "__main__":
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load vectorizer (change the path to your vectorizer .pkl file)
    vectorizer_path = "../Vectorizer .3n3.pkl"
    model_path = "../Model SenseMini .3n3.pth"

    # Load vectorizer
    print(f"Loading vectorizer from: {vectorizer_path}")
    with open(vectorizer_path, "rb") as f:
        vectorizer = joblib.load(f)

    # Load model and move to the appropriate device (GPU/CPU)
    print(f"Loading model from: {model_path}")
    model = torch.load(model_path, weights_only=False)
    model.to(device)  # Move model to GPU or CPU
    print(model)

    # Instantiate dummy data loader
    print("Creating dummy data loader...")
    dummy_data_loader = DataLoader(DummyDataset(), batch_size=32)

    # Define loss criterion
    print("Defining loss criterion...")
    criterion: torch.nn = torch.nn.CrossEntropyLoss()

    # Visualizations
    print("Creating visualizations...")
    tokens: TfidfTransformer = vectorizer.get_feature_names_out()

    # Feature importance (dummy data)
    NUMBER_OF_FEATURES: int = -1  # Number of features to visualize, -1 for all
    # Max number of features to visualize is 3000 due to image constraints
    print(f"Visualizing feature importance - This may take a while for {len(tokens[:NUMBER_OF_FEATURES])+1} tokens...")
    feature_importance = np.random.rand(len(tokens[:NUMBER_OF_FEATURES]))  # Example random importance
    visualize_feature_importance(tokens[:NUMBER_OF_FEATURES], feature_importance, FILENAME="NN features/feature_importance.svg")

    # Loss landscape
    print("Visualizing loss landscape - This may take a while...")
    plot_loss_landscape_3d(model, dummy_data_loader, criterion, FILENAME="NN features/loss_landscape_3d.html")

    print("Completed.")
