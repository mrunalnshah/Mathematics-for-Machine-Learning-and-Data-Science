# -*- coding: utf-8 -*-
# +
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


FONT_SIZE_TICKS = 14
FONT_SIZE_TITLE = 20
FONT_SIZE_AXES = 16


def calculate_feature_importance(features, lr, X_test, y_test):
    if len(features) > 1:
        bunch = permutation_importance(
            lr, X_test, y_test, n_repeats=10, random_state=42
        )
        imp_means = bunch.importances_mean
        ordered_imp_means_args = np.argsort(imp_means)[::-1]

        results = {}
        for i in ordered_imp_means_args:
            name = list(X_test.columns)[i]
            imp_score = imp_means[i]
            results.update({name: [imp_score]})

        most_important = list(X_test.columns)[ordered_imp_means_args[0]]
        results_df = pd.DataFrame.from_dict(results)

        return most_important, results_df
    
    else:
        return features[0], None


def plot_feature_importance(df):
    # Create a plot for feature importance
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel("Importance Score", fontsize=FONT_SIZE_AXES)
    ax.set_ylabel("Feature", fontsize=FONT_SIZE_AXES)
    ax.set_title("Feature Importance", fontsize=FONT_SIZE_TITLE)
    ax.tick_params(labelsize=FONT_SIZE_TICKS)

    sns.barplot(data=df, orient="h", ax=ax, color="deepskyblue")

    plt.show()

    
def plot_happiness(variable, x1, y1, x2, y2):
    """Plots a predictor variable on x-axis and happiness (life ladder) on y axis.

    Args:
        variable: The name of the x axis variable
        x1, y1: The x, y original data to be plotted. Both can be None if not available.
        x2, y2: The x, y data model to be plotted. Both can be None if not available.
    """
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    # Plot the original data
    ax.scatter(
        x1, y1, color="blue", edgecolors="white", s=15, label="Training Data"
    )
    # Plot the model
    ax.scatter(
        x2, y2,
        color="orange", edgecolors="black", s=15, marker="D", label="Predictions on the Test Set"
    )
    
    variable_title = " ".join(variable.split("_")).title()
    ax.set_xlabel(f"{variable_title}", fontsize=FONT_SIZE_AXES)
    ax.set_ylabel("Life Lsadder (1-10)", fontsize=FONT_SIZE_AXES)
    ax.set_title(f"Happiness vs. {variable_title}", fontsize=FONT_SIZE_TITLE)
    ax.tick_params(labelsize=FONT_SIZE_TICKS)
    ax.legend(fontsize=FONT_SIZE_TICKS)
