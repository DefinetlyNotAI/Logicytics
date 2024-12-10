import os
import os.path
import random
from collections import OrderedDict
from configparser import ConfigParser
from os import mkdir

import joblib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import torch
import torch.nn as nn
from faker import Faker
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset
from torchviz import make_dot
from tqdm import tqdm


# TODO Add docstring, and hint-type
# TODO Do v3.1 plans
#  ZIP the file and attach somewhere (Data)

# Example of DataLoader for loss landscape (dummy dataset for visualization)
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=100, input_dim=10000):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.data = []
        self.labels = []
        faker = Faker()
        for _ in range(num_samples):
            if random.random() < 0.05:  # 5% chance to include sensitive data
                self.data.append(f"Name: {faker.name()}, SSN: {faker.ssn()}, Address: {faker.address()}")
                self.labels.append(1)  # Label as sensitive
            else:
                self.data.append(faker.text(max_nb_chars=100))  # Non-sensitive data
                self.labels.append(0)  # Label as non-sensitive

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        # Convert data to tensor of ASCII values and pad to input_dim
        data_tensor = torch.tensor([ord(c) for c in data], dtype=torch.float32)
        if len(data_tensor) < self.input_dim:
            padding = torch.zeros(self.input_dim - len(data_tensor))
            data_tensor = torch.cat((data_tensor, padding))
        else:
            data_tensor = data_tensor[:self.input_dim]
        label_tensor = torch.tensor(label, dtype=torch.long)
        return data_tensor, label_tensor


def load_data(text_data, vectorizer_to_load):
    # Vectorize the text data
    X = vectorizer_to_load.transform(text_data)
    # Create a dummy label for visualization (replace with real labels if available)
    y = np.zeros(len(text_data))
    # Convert to torch tensors
    X_tensor = torch.tensor(X.toarray(), dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=32, shuffle=True)


def visualize_weight_distribution(model_to_load):
    # Access weights of the first layer
    weights = model_to_load[0].weight.detach().cpu().numpy()  # Move tensor to CPU before conversion to numpy
    plt.hist(weights.flatten(), bins=50)
    plt.title("Weight Distribution - First Layer")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.savefig("NN features/Weight Distribution.png")
    plt.close()


def visualize_activations(model_to_load, input_tensor):
    # Check the device of the model
    device_va = next(model_to_load.parameters()).device

    # Move the input tensor to the same device as the model
    input_tensor = input_tensor.to(device_va)

    activations = []

    # noinspection PyUnusedLocal
    def hook_fn(module, inputx, output):
        # Hook function to extract intermediate layer activations
        activations.append(output)

    model_to_load[0].register_forward_hook(hook_fn)  # Register hook on first layer

    # Perform a forward pass
    _ = model_to_load(input_tensor)
    activation = activations[0].detach().cpu().numpy()  # Move activations to CPU

    # Plot activations as a bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(activation[0])), activation[0])
    plt.title("Activation Values - First Layer")
    plt.xlabel("Neuron Index")
    plt.ylabel("Activation Value")
    plt.savefig("NN features/Visualize Activation.png")
    plt.close()


def visualize_tsne(model_to_load, dataloader):
    # Get the device of the model
    device_va = next(model_to_load.parameters()).device

    model_to_load.eval()  # Set the model to evaluation mode

    features = []
    labels = []

    with torch.no_grad():
        for data, target in dataloader:
            # Move data and target to the same device as the model
            data, target = data.to(device_va), target.to(device_va)

            # Extract features (output of the model)
            output = model_to_load(data)
            features.append(output.cpu().numpy())  # Move output to CPU for concatenation
            labels.append(target.cpu().numpy())  # Move target to CPU for concatenation

    # Stack all batches
    features = np.vstack(features)
    labels = np.hstack(labels)

    # Determine suitable perplexity
    num_samples = features.shape[0]
    perplexity = min(30, num_samples - 1)  # Ensure perplexity < num_samples

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    reduced_features = tsne.fit_transform(features)

    # Plot the t-SNE results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label="Class")
    plt.title("t-SNE Visualization of Features")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.savefig("NN features/Visualize t-SNE.png")
    plt.close()


# Main function to run all visualizations
def plot_many_graphs():
    print("Starting synthetic data generation...")
    # Load data
    faker = Faker()

    # Generate sensitive examples
    sensitive_data = [
        f"Name: {faker.name()}, SSN: {faker.ssn()}, Address: {faker.address()}",
        f"Credit Card: {faker.credit_card_number()}, Expiry: {faker.credit_card_expire()}, CVV: {faker.credit_card_security_code()}",
        f"Patient: {faker.name()}, Condition: {faker.text(max_nb_chars=20)}",
        f"Password: {faker.password()}",
        f"Email: {faker.email()}",
        f"Phone: {faker.phone_number()}",
        f"Medical Record: {faker.md5()}",
        f"Username: {faker.user_name()}",
        f"IP: {faker.ipv4()}",
    ]

    # Generate non-sensitive examples
    non_sensitive_data = [
        faker.text(max_nb_chars=50) for _ in range(50000)
    ]

    data_text = non_sensitive_data + (sensitive_data * 15)
    random.shuffle(data_text)
    print("Loaded data for visualization.")
    dataloader = load_data(data_text, vectorizer)

    # Visualizations
    print("Creating visualizations...")
    visualize_weight_distribution(model)

    # For activations, use a sample from the dataloader
    print("Creating activation visualizations...")
    sample_input = next(iter(dataloader))[0]
    visualize_activations(model, sample_input)

    print("Creating t-SNE visualization - May take a long time...")
    visualize_tsne(model, dataloader)

    print("Completed.")


# Visualize feature importance (dummy example for visualization) and save as SVG
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
        print(f"Computing loss for row {i + 1}/{GRID_SIZE}...")
        for j, dy in enumerate(y):
            print(f"    Computing loss for column {j + 1}/{GRID_SIZE}...")
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


def main_plot():
    # Instantiate data loader
    print("Creating dummy data loader...")
    dummy_data_loader = DataLoader(DummyDataset(), batch_size=32)

    # Define loss criterion
    print("Defining loss criterion...")
    criterion = torch.nn.CrossEntropyLoss()

    # Visualizations
    print("Creating visualizations...")
    tokens = vectorizer.get_feature_names_out()

    # Feature importance
    # Max number of features to visualize is 3000 due to image constraints
    print(
        f"Visualizing feature importance - This may take a while for {len(tokens[:NUMBER_OF_FEATURES]) + 1} tokens...")
    feature_importance = np.random.rand(len(tokens[:NUMBER_OF_FEATURES]))  # Example random importance
    visualize_feature_importance(tokens[:NUMBER_OF_FEATURES], feature_importance,
                                 FILENAME="NN features/feature_importance.svg")

    # Loss landscape
    print("Visualizing loss landscape - This may take a while...")
    plot_loss_landscape_3d(model, dummy_data_loader, criterion, FILENAME="NN features/loss_landscape_3d.html")

    # Set model to evaluation mode, and plot many graphs
    print("Setting model to evaluation mode...")
    model.eval()  # Set the model to evaluation mode
    plot_many_graphs()


def save_data(model_to_use, input_size, batch_size=-1, device_to_use="cuda"):
    def register_hook(module):

        def hook(modules, inputs, output):
            class_name = str(modules.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summaries)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summaries[m_key] = OrderedDict()
            summaries[m_key]["input_shape"] = list(inputs[0].size())
            summaries[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summaries[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summaries[m_key]["output_shape"] = list(output.size())
                summaries[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(modules, "weight") and hasattr(modules.weight, "size"):
                params += torch.prod(torch.LongTensor(list(modules.weight.size())))
                summaries[m_key]["trainable"] = modules.weight.requires_grad
            if hasattr(modules, "bias") and hasattr(modules.bias, "size"):
                params += torch.prod(torch.LongTensor(list(modules.bias.size())))
            summaries[m_key]["nb_params"] = params

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == model_to_use)
        ):
            hooks.append(module.register_forward_hook(hook))

    device_to_use = device_to_use.lower()
    assert device_to_use in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device_to_use == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batch norm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]

    # create properties
    summaries = OrderedDict()
    hooks = []

    # register hook
    model_to_use.apply(register_hook)

    # make a forward pass
    model_to_use(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    # Save the summary
    mode = "a" if os.path.exists("NN features/Model Summary.txt") else "w"
    with open('NN features/Model Summary.txt', mode) as vf_ms:
        vf_ms.write("----------------------------------------------------------------\n")
        line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
        vf_ms.write(f"{line_new}\n")
        vf_ms.write("================================================================\n")
        total_params = 0
        total_output = 0
        trainable_params = 0
        for layer in summaries:
            # input_shape, output_shape, trainable, nb_params
            line_new = "{:>20}  {:>25} {:>15}".format(
                layer,
                str(summaries[layer]["output_shape"]),
                "{0:,}".format(summaries[layer]["nb_params"]),
            )
            total_params += summaries[layer]["nb_params"]
            total_output += np.prod(summaries[layer]["output_shape"])
            if "trainable" in summaries[layer]:
                if summaries[layer]["trainable"]:
                    trainable_params += summaries[layer]["nb_params"]
            vf_ms.write(f"{line_new}\n")

        # assume 4 bytes/number (float on cuda).
        total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
        total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
        total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
        total_size = total_params_size + total_output_size + total_input_size

        vf_ms.write("\n================================================================")
        vf_ms.write("\nTotal params: {0:,}".format(total_params))
        vf_ms.write("\nTrainable params: {0:,}".format(trainable_params))
        vf_ms.write("\nNon-trainable params: {0:,}".format(total_params - trainable_params))
        vf_ms.write("\n----------------------------------------------------------------")
        vf_ms.write("\nInput size (MB): %0.2f" % total_input_size)
        vf_ms.write("\nForward/backward pass size (MB): %0.2f" % total_output_size)
        vf_ms.write("\nParams size (MB): %0.2f" % total_params_size)
        vf_ms.write("\nEstimated Total Size (MB): %0.2f" % total_size)
        vf_ms.write("\n----------------------------------------------------------------\n")


def save_graph():
    # Create a directed graph
    G = nx.DiGraph()

    def add_edges_bulk(layer_names, weight_matrices):
        """Efficiently add edges to the graph with progress tracking."""
        threshold = 0.1  # Adjust this threshold as needed
        significant_weights = np.abs(weight_matrices) > threshold
        rows, cols = np.where(significant_weights)
        weights = weight_matrices[rows, cols]

        # Use tqdm for progress tracking
        edge_count = len(rows)
        with tqdm(total=edge_count, desc=f"Processing {layer_names}", unit="edges") as pbar:
            for row, col, weight in zip(rows, cols, weights):
                in_node = f"{layer_names}_in_{col}"
                out_node = f"{layer_names}_out_{row}"
                G.add_edge(in_node, out_node, weight=weight)
                pbar.update(1)

    # Process parameters
    for name, param in model.named_parameters():
        if 'weight' in name:
            layer_name = name.split('.')[0]
            weight_matrix = param.data.cpu().numpy()

            # Add edges with progress bar
            add_edges_bulk(layer_name, weight_matrix)

    # Draw the graph
    print("Writing the graph to a file...")
    nx.write_gexf(G, "NN features/Neural Network Nodes Graph.gexf")


def setup_environment():
    print("Visualizing the model and vectorizer features...")
    print("This may take a while, please wait.")

    if not os.path.exists('NN features'):
        mkdir('NN features')


def load_vectorizer():
    vectorizer_load = joblib.load(vectorizer_path)
    feature_names = vectorizer_load.get_feature_names_out()
    with open('NN features/Vectorizer features.txt', 'w') as file:
        file.write(f"Number of features: {len(feature_names)}\n\n")
        file.write('\n'.join(feature_names))
    return vectorizer_load


def visualize_top_features(top_n=90):
    feature_names = vectorizer.get_feature_names_out()
    sorted_indices = vectorizer.idf_.argsort()[:top_n]
    top_features = [feature_names[i] for i in sorted_indices]
    top_idf_scores = vectorizer.idf_[sorted_indices]

    plt.figure(figsize=(20, 12))  # Increase the figure size
    sns.barplot(x=top_idf_scores, y=top_features)
    plt.title('Top 90 Features by IDF Score')
    plt.xlabel('IDF Score')
    plt.ylabel('Feature')

    # Save the plot as a vector graphic
    plt.savefig('NN features/Top_90_Features.svg', format='svg')
    plt.close()


def load_model():
    device_load = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_load = torch.load(model_path, weights_only=False)
    model_load.to(device_load)
    return model_load, device_load


def save_model_state_dict():
    with open('NN features/Model state dictionary.txt', 'w') as file:
        file.write("Model's state dictionary:\n\n")
        for param_tensor in model.state_dict():
            file.write(f"\n{param_tensor}\t{model.state_dict()[param_tensor].size()}")


def generate_model_visualization():
    dummy_input = torch.randn(1, vectorizer.vocabulary_.__len__()).to(device)
    model_viz = make_dot(model(dummy_input), params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
    model_viz.format = 'png'
    model_viz.render(filename='NN features/Model Visualization', format='png')


def cleanup_temp_files():
    if os.path.exists("NN features/Model Visualization"):
        os.remove("NN features/Model Visualization")


def model_summary():
    mode = "a" if os.path.exists("NN features/Model Summary.txt") else "w"
    with open("NN features/Model Summary.txt", mode) as file:
        file.write(str(model))


if __name__ == '__main__':
    # Print the welcome message
    print("===========================================================================================")
    print("= This script will visualize the features of the model and vectorizer.                    =")
    print("= Please ensure that the model and vectorizer files are present in the specified paths.   =")
    print("= The visualization will be saved in the 'NN features' directory.                         =")
    print("= This script will take a while to run, please be patient.                                =")
    print("===========================================================================================")

    # Read the config file
    print("\n\nReading config file and setting up...")
    config = ConfigParser()
    config.read('../../config.ini')

    setup_environment()

    # Load the paths from the config file
    vectorizer_path = config.get('VulnScan.study Settings', 'vectorizer_path')
    model_path = config.get('VulnScan.study Settings', 'model_path')
    NUMBER_OF_FEATURES = int(config.get('VulnScan.study Settings', 'number_of_features'))

    # Check if the paths exist
    if not os.path.exists(vectorizer_path):
        print(f"Vectorizer file not found. Please double check the path {vectorizer_path}.")
        exit(1)
    if not os.path.exists(model_path):
        print(f"Model file not found. Please double check the path {model_path}.")
        exit(1)

    # Load the vectorizer and model
    vectorizer = load_vectorizer()
    visualize_top_features()
    model, device = load_model()
    # Save the model summary, state dictionary, and visualization
    save_data(model, input_size=(1, vectorizer.vocabulary_.__len__()))
    save_model_state_dict()
    generate_model_visualization()
    cleanup_temp_files()
    save_graph()
    print("Model visualization and summary have been saved to the 'NN features' directory.")

    # Check if GPU is available
    if not os.path.exists('NN features'):
        os.mkdir('NN features')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load vectorizer (change the path to your vectorizer .pkl file)
    vectorizer_path = "../Vectorizer .3n3.pkl"
    model_path = "../Model SenseMini .3n3.pth"

    # Load vectorizer
    print(f"Reloading vectorizer from: {vectorizer_path}")
    with open(vectorizer_path, "rb") as f:
        vectorizer = joblib.load(f)

    # Load model and move to the appropriate device (GPU/CPU)
    print(f"Reloading model from: {model_path}")
    model = torch.load(model_path, weights_only=False)
    model.to(device)  # Move model to GPU or CPU

    model_summary()
    main_plot()
