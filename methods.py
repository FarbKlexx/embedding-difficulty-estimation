'''
This file handles the machine learning pipelines of the following methods:
- naive_e5_regression: embedding regression training based on true difficulty of items
- best_regressor: find the regressor with the highest accuracy from a collection of regressor candidates based on the average embeddings of multiple prompts
- best_prompt: find the best prompt for a specific regressor
- 

Embedding Model e5 and Base Code from https://huggingface.co/intfloat/multilingual-e5-large-instruct
Plot Templates by Andrey Churkin from https://github.com/AndreyChurkin/BeautifulFigures/
'''

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl

def e5_embed(data: list[str]) -> list:
    """
    Handles the e5 embedding process

    Parameters
    ----------
    data: list[str]
        - A list containing the texts that should be embedded

    Returns
    -------
    list
        - The embeddings
    """
    if (torch.cuda.is_available()):
        print("Cuda is available. Calculate embeddings on gpu...")
        model = SentenceTransformer('intfloat/multilingual-e5-large-instruct', device='cuda')
    else:
        print("Cuda is unavailable. Calculate embeddings on cpu...")
        model = SentenceTransformer('intfloat/multilingual-e5-large-instruct')

    embeddings = model.encode(data, convert_to_tensor=True, normalize_embeddings=True)
    embeddings = embeddings.cpu()

    print(f'Embedded {len(embeddings)} items.')
    print("\n")

    return embeddings

def baseline_regressor(y_train: list, y_test: list) -> float:
    """
    Immitates a regressor that always predicts the mean

    Parameters
    ----------
    y_train: list
        - The y-values of the training data

    y_test: list
        - The y-values of the  test data

    Returns
    -------
    float
        - The baseline rmse
    """
    y_train_mean = np.mean(y_train)
    y_train_mean_list = []
    for i in y_test:
        y_train_mean_list.append(y_train_mean)

    rmse = root_mean_squared_error(y_test, y_train_mean_list)

    print('Baseline RMSE: ' + str(rmse))
    
    return rmse

def test_regressor(y_test: list, y_pred: list) -> tuple:
    """
    Outputs the RMSE and R2-Score for the regressor

    Parameters
    ----------
    data: list
        A list containing tuples of the form (X,y)
        - X: the embedded text
        - y: the corresponding difficulty
        
    Returns
    -------
    tuple of floats
        A tuple with 2 elements:
        rmse: the root mean squared error of the predicted values
        r2: the r2 score of the predicted values
    """
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print('RMSE for Regressor: ' + str(rmse))
    print('R2-Score for Regressor: ' + str(r2))
    print("\n")
    
    # # Defining the fonts before plotting:
    plt.rcParams.update({
        'font.family': 'Courier New',  # monospace font
        'font.size': 20,
        'axes.titlesize': 20,
        'axes.labelsize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20,
        'figure.titlesize': 20
    }) 
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.set_aspect('equal', adjustable='datalim') # Lock the square shape
    
    ax.grid(True, which='major', linestyle='-', linewidth=0.75, alpha=0.25)
    
    ax.minorticks_on()
    ax.grid(True, which='minor', linestyle='-', linewidth=0.25, alpha=0.15)
    
    ax.set_axisbelow(True) # <-- Ensure grid is below data

    regression_color = "#a1a1a1" # <-- neutral grey       
     
    mean_color = "#707070"

    # Plot the Test and Pred Values
    
    # Define color gradient
    cmap = LinearSegmentedColormap.from_list("custom", ['#9671bd', '#77b5b6'])
    cmap_edges = LinearSegmentedColormap.from_list("custom", ['#6a408d', '#378d94'])

    norm = Normalize(vmin=min(y_test), vmax=max(y_test))
    norm_edges = Normalize(vmin=min(y_test), vmax=max(y_test))
    
    colors = cmap(norm(y_test))
    colors_edges = cmap_edges(norm_edges(y_test))
    
    x_values = [min(y_test), max(y_test)]
    y_values = [min(y_test), max(y_test)]
    
    ax.scatter(y_test, y_pred, 
               s = 90,
               color = colors,
               edgecolors = colors_edges,
               linewidths = 1.5,
               zorder = 3
    )
    
    ax.plot(x_values, y_values, 
            label="Optimal",
            linewidth = 2.6,
            color = regression_color,
            linestyle = '--',
            zorder = 2 # use the z-order to force scatter to be displayed over lines
    )
    
    mean = np.mean(y_test)
    
    plt.axhline(y=mean,
            label="Mean",
            linewidth = 2.6,
            color = mean_color,
            zorder = 2 # use the z-order to force scatter to be displayed over lines
    )
    
    handles, labels = ax.get_legend_handles_labels() # get all legend items

    ax.legend(
        handles,
        labels,
        loc = 'upper center',
        bbox_to_anchor = (0.5, 1.10),  # center top, above axes
        ncol = 4,                      # spread horizontally
        frameon = False                # removes legend border
    ) 
    
    plt.title('Comparison of Predicted Difficulty')
    plt.xlabel('Actual Difficulty')
    plt.ylabel('Predicted Difficulty')
    plt.show()
    
    return rmse,r2