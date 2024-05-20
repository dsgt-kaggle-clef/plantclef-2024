import io
import math

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def plot_images_from_binary(df, data_col: str, image_col: str, grid_size=(3, 3)):
    """
    Display images in a grid with binomial names as labels.

    :param df: DataFrame with the embeddings data.
    :param data_col: Name of the data column.
    :param image_col: Name of the species being displayed as image labels.
    :param grid_size: Tuple (rows, cols) representing the grid size.
    """
    # Unpack the number of rows and columns for the grid
    rows, cols = grid_size

    # Collect binary image data from DataFrame
    subset_df = df.limit(rows * cols).collect()
    image_data_list = [row[data_col] for row in subset_df]
    image_names = [row[image_col] for row in subset_df]

    # Create a matplotlib subplot with the specified grid size
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12), dpi=80)

    # Flatten the axes array for easy iteration if it's 2D
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for ax, binary_data, name in zip(axes, image_data_list, image_names):
        # Convert binary data to an image and display it
        image = Image.open(io.BytesIO(binary_data))
        ax.imshow(image)
        name = name.replace("_", " ")
        ax.set_xlabel(name)  # Set the binomial name as xlabel
        ax.xaxis.label.set_size(14)  # Set the font size for the xlabel
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()


def plot_images_from_embeddings(df, data_col: str, image_col: str, grid_size=(3, 3)):
    """
    Display images in a grid with species names as labels.

    :param df: DataFrame with the embeddings data.
    :param data_col: Name of the data column.
    :param image_col: Name of the species being displayed as image labels.
    :param grid_size: Tuple (rows, cols) representing the grid size.
    """
    # Unpack the number of rows and columns for the grid
    rows, cols = grid_size

    # Collect binary image data from DataFrame
    subset_df = df.limit(rows * cols).collect()
    embedding_data_list = [row[data_col] for row in subset_df]
    image_names = [row[image_col] for row in subset_df]

    # Create a matplotlib subplot with specified grid size
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12), dpi=80)

    # Flatten the axes array for easy iteration
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for ax, embedding, name in zip(axes, embedding_data_list, image_names):
        # Find the next perfect square size greater than or equal to the embedding length
        next_square = math.ceil(math.sqrt(len(embedding))) ** 2
        padding_size = next_square - len(embedding)

        # Pad the embedding if necessary
        if padding_size > 0:
            embedding = np.pad(
                embedding, (0, padding_size), "constant", constant_values=0
            )

        # Reshape the embedding to a square
        side_length = int(math.sqrt(len(embedding)))
        image_array = np.reshape(embedding, (side_length, side_length))

        # Normalize the embedding to [0, 255] for displaying as an image
        normalized_image = (
            (image_array - np.min(image_array))
            / (np.max(image_array) - np.min(image_array))
            * 255
        )
        image = Image.fromarray(normalized_image).convert("L")

        ax.imshow(image, cmap="gray")
        ax.set_xlabel(name)  # Set the species name as xlabel
        ax.xaxis.label.set_size(14)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()


def plot_species_histogram(df, species_count: int = 100, bar_width: float = 0.8):
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)

    species_df = (
        df.filter(f"n >= {species_count}").orderBy("n", ascending=False).toPandas()
    )

    # Get the top and bottom 5 species
    top5_df = species_df.head(5)

    # Plot all species
    ax.bar(
        species_df["species"], species_df["n"], color="lightslategray", width=bar_width
    )

    # Highlight the top 5 species in different colors
    if species_count >= 600:
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
        for i, row in top5_df.iterrows():
            ax.bar(
                row["species"],
                row["n"],
                color=colors[i],
                label=row["species"],
                width=bar_width,
            )
        ax.legend(title="Top 5 Species")

    ax.set_xlabel("Species")
    ax.set_ylabel("Count")
    ax.set_title(
        f"PlantCLEF 2024 Histogram of Plant Species with Count >= {species_count}",
        weight="bold",
        fontsize=16,
    )
    ax.set_xticks([])
    ax.set_xmargin(0)
    ax.xaxis.label.set_size(14)  # Set the font size for the xlabel
    ax.yaxis.label.set_size(14)  # Set the font size for the xlabel
    ax.grid(color="blue", linestyle="--", linewidth=1, alpha=0.2)
    spines = ["top", "right", "bottom", "left"]
    for s in spines:
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    plt.show()
