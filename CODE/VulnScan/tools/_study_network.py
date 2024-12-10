import os.path
from collections import OrderedDict
from os import mkdir

import joblib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torchviz import make_dot


def save_graph(model_to_use, input_size, batch_size=-1, device_to_use="cuda"):
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
    # print(type(x[0]))

    # create properties
    summaries = OrderedDict()
    hooks = []

    # register hook
    model_to_use.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model_to_use(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    with open('NN features/Model Summary.txt', 'w') as vf_ms:
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


def visualize_model(models, output_dir="NN features"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes and edges to the graph
    for model_i in models:
        for names, param in model_i.named_parameters():
            G.add_node(names, size=param.numel())
            if param.requires_grad:
                G.add_edge(names, f"{names}_grad")

    # Define the output file path
    output_file = os.path.join(output_dir, "model.graphml")

    # Write the graph to a GraphML file
    nx.write_graphml(G, output_file)

    print(f"Model visualization saved as {output_file}")


if __name__ == '__main__':
    print("Visualizing the model and vectorizer features...")
    print("This may take a while, please wait.")

    if not os.path.exists('NN features'):
        mkdir('NN features')

    # Load the vectorizer
    vectorizer_path = '../Vectorizer .3n3.pkl'
    vectorizer = joblib.load(vectorizer_path)

    # Inspect the vectorizer
    feature_names = vectorizer.get_feature_names_out()
    with open('NN features/Vectorizer features.txt', 'w') as f:
        f.write(f"Number of features: {len(feature_names)}\n\n")
        f.write('\n'.join(feature_names))

    # Visualize the top 90 features
    top_n = 90
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

    plt.show()

    # Load the model
    model_path = '../Model SenseMini .3n3.pth'
    model = torch.load(model_path, weights_only=False)

    # Save the model summary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    save_graph(model, input_size=(1, vectorizer.vocabulary_.__len__()))

    # Save the model's state dictionary
    with open('NN features/Model state dictionary.txt', 'w') as f:
        f.write("Model's state dictionary:\n\n")
        for param_tensor in model.state_dict():
            f.write(f"\n{param_tensor}\t{model.state_dict()[param_tensor].size()}")

    # Create a dummy input tensor with the appropriate size
    dummy_input = torch.randn(1, vectorizer.vocabulary_.__len__()).to(device)

    # Generate the visualization
    model_viz = make_dot(model(dummy_input), params=dict(model.named_parameters()), show_attrs=True, show_saved=True)

    # Save the visualization to a file
    model_viz.format = 'png'
    model_viz.render(filename='NN features/Model Visualization', format='png')

    # Removing the temporary files as they are no longer needed, we saved them to the desired location
    if os.path.exists("NN features/Model Visualization"):
        os.remove("NN features/Model Visualization")

    # Visualize the model
    visualize_model(model)

    print("Model visualization and summary have been saved to the 'NN features' directory.")
