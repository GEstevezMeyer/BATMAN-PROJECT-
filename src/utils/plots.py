import matplotlib.pyplot as plt 
import math 
import numpy as np 



def plot_metrics(history: dict):
    
    keys = list(history.keys())
    epochs = [i + 1 for i in range(len(history[keys[0]]))]

    n = len(keys)
    number_of_rows = min(n, 3)
    number_of_columns = math.ceil(n / number_of_rows)

    fig, ax = plt.subplots(number_of_rows, number_of_columns)
    ax = np.array(ax).reshape(number_of_rows, number_of_columns)

    for i, key in enumerate(keys):
        ax_x, ax_y = i % number_of_rows, i // number_of_rows
        ax[ax_x, ax_y].plot(epochs, history[key])
        ax[ax_x, ax_y].set_title(key)

    for i in range(n, number_of_rows * number_of_columns):
        ax[i % number_of_rows, i // number_of_rows].set_visible(False)

    plt.tight_layout()
    plt.show()
        
