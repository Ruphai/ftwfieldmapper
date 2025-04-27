"""
Utility functions for the FTW Project. Mostly for data preparation and visualization. 
Last Edited: 2025-04-21
Author: Rufai Omowunmi Balogun
"""

import os
import sys
import pathlib
import random
import math
import numpy as np
import pandas as pd

import rasterio
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
import pathlib as Path
import random
from scipy.ndimage import label
from skimage.color import label2rgb

def print_information(wina, winb, sem2, sem3, inst):
    """ Function to print detailed information about the images and masks"""

    def print_stats(data, name):
        """Print statistics like shape, dtype, range, mean, etc."""
        print(f"{name} Shape: ", data.shape)
        print(f"{name} Dtype: ", data.dtype)
        print(f"{name} Value Range: ", np.min(data), np.max(data))
        print(f"{name} Mean: ", np.mean(data))
        print(f"{name} Standard Deviation: ", np.std(data))
        print(f"{name} Unique Values: ", np.unique(data))
        print(f"{name} NaN or Invalid Values Present: ", np.isnan(data).any())

    # Window A
    print_stats(wina, "Window A")
    
    # Window B
    print_stats(winb, "Window B")
    
    # Semantic 2-class
    print_stats(sem2, "Semantic 2-class")
    
    # Semantic 3-class
    print_stats(sem3, "Semantic 3-class")
    
    # Instance Class
    print_stats(inst, "Instance Class")


def plot_ftw_data(window_a_file, window_b_file, semantic_2_class_file, semantic_3_class_file, instance_class_file):
    # Load window A and window B
    with rasterio.open(window_a_file) as src:
        window_a = src.read()[0:3, :, :]  # Reading first 3 bands
        window_a = window_a.transpose(1, 2, 0) / 3000  # Normalizing
    
    with rasterio.open(window_b_file) as src:
        window_b = src.read()[0:3, :, :]  # Reading first 3 bands
        window_b = window_b.transpose(1, 2, 0) / 3000  # Normalizing

    # Load semantic and instance data
    with rasterio.open(semantic_2_class_file) as src:
        semantic_2_class = src.read()

    with rasterio.open(semantic_3_class_file) as src:
        semantic_3_class = src.read()

    with rasterio.open(instance_class_file) as src:
        instance_class = src.read()[0]  # Assuming it's single band data for instance labels

        # Generate random colors for each instance class
        unique_labels = np.unique(instance_class)
        colors = [(np.random.rand(), np.random.rand(), np.random.rand()) for _ in unique_labels]
        instance_mask_rgb = label2rgb(instance_class, bg_label=0, bg_color=(0, 0, 0), colors=colors)

    # Print detailed information about the images and masks
    # print_information(window_a, window_b, semantic_2_class, semantic_3_class, instance_class)

    # Create subplots to visualize the data
    fig, axs = plt.subplots(1, 5, figsize=(20, 10))
    
    # Display Window A
    axs[0].imshow(np.clip(window_a, 0, 1))  # Clipping to avoid over-scaling issues
    axs[0].set_title('Window A')
    
    # Display Window B
    axs[1].imshow(np.clip(window_b, 0, 1))  # Clipping to avoid over-scaling issues
    axs[1].set_title('Window B')
    
    # Display Semantic 2-class
    axs[2].imshow(semantic_2_class[0], cmap='viridis', vmin=0, vmax=2)
    axs[2].set_title('Semantic 2-class')
    
    # Display Semantic 3-class
    axs[3].imshow(semantic_3_class[0], cmap='viridis', vmin=0, vmax=2)
    axs[3].set_title('Semantic 3-class')
    
    # Display Instance class with RGB mask
    axs[4].imshow(instance_mask_rgb)
    axs[4].set_title('Instance class')

    for ax in axs:
        ax.axis('off')

    # Display the plot
    plt.show()


def show_image_and_mask(
    row,
    root_dir: str = ".",
    img_col: str = "image",
    mask_col: str = "label",
    rgb_bands: tuple[int, int, int] | None = (1, 2, 3),
    mask_cmap: str = "tab20",
    alpha: float = 1.0,
):
    img_path  = os.path.join(root_dir, str(row[img_col]))
    img_path = img_path[:-4]
    mask_path = os.path.join(root_dir, str(row[mask_col]))

    if not (os.path.exists(img_path) and os.path.exists(mask_path)):
        raise FileNotFoundError(f"Cannot find {img_path} or {mask_path}")
    with rasterio.open(img_path) as src:
        if rgb_bands is None:
            rgb = src.read(1, out_dtype="float32", resampling=Resampling.nearest)
            rgb = np.clip(rgb, np.nanpercentile(rgb, 2), np.nanpercentile(rgb, 98))
            img_rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)
            img_rgb = np.repeat(img_rgb[None, ...], 3, axis=0)  
        else:
            rgb = src.read(rgb_bands, out_dtype="float32", resampling=Resampling.nearest)
            for i in range(rgb.shape[0]):
                p2, p98 = np.nanpercentile(rgb[i], (2, 98))
                rgb[i] = np.clip(rgb[i], p2, p98)
                rgb[i] = (rgb[i] - p2) / (p98 - p2 + 1e-6)
            img_rgb = rgb
    
    img_rgb = np.moveaxis(img_rgb, 0, -1)  # (H,W,C)

    with rasterio.open(mask_path) as msk:
        mask = msk.read(1)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), dpi=120)

    axs[0].imshow(img_rgb)
    axs[0].set_title("Image")
    axs[0].axis("off")
    axs[1].imshow(mask, cmap=mask_cmap, alpha=alpha, interpolation="nearest")
    axs[1].set_title("Mask")
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()


def show_random_samples_by_class(
    df: pd.DataFrame,
    root_dir: str = ".",
    class_col: str = "Class",
    classes: list[int] | None = None,
    samples_per_class: int = 3,
    rgb_bands: tuple[int, int, int] | None = (1, 2, 3),
    mask_cmap: str = "tab20",
    alpha: float = 1.0,
    seed: int | None = None,
):
    if seed is not None:
        random.seed(seed)

    if classes is None:
        classes = sorted(df[class_col].unique())
        
    n_rows = len(classes)
    n_cols = samples_per_class * 2          # image + mask per sample
    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), dpi=110
    )

    if n_rows == 1:
        axs = axs[None, :]

    for r, cls in enumerate(classes):
        subset = df[df[class_col] == cls]
        if subset.empty:
            raise ValueError(f"No rows found for class {cls}")
        
        rows = random.sample(list(subset.index), min(samples_per_class, len(subset)))
        if len(rows) < samples_per_class:
            rows *= math.ceil(samples_per_class / len(rows))
            rows = rows[:samples_per_class]

        for c, idx in enumerate(rows):
            ax_img = axs[r, 2 * c]    
            ax_msk = axs[r, 2 * c + 1]

            _show_single_pair_into_axes(
                df.loc[idx],
                ax_img,
                ax_msk,
                root_dir=root_dir,
                rgb_bands=rgb_bands,
                mask_cmap=mask_cmap,
                alpha=alpha,
            )

            if r == 0:
                ax_img.set_title(f"Img {c+1}", fontsize=12)
                ax_msk.set_title(f"Mask {c+1}", fontsize=12)
                
        axs[r, 0].set_ylabel(f"Class {cls}", fontsize=14, rotation=0, labelpad=50)

    plt.tight_layout()
    plt.show()

def _show_single_pair_into_axes(
    row,
    ax_img,
    ax_msk,
    root_dir,
    rgb_bands,
    mask_cmap,
    alpha,
):
    """
    Internal helper used by show_random_samples_by_class.
    Reads & plots exactly like show_image_and_mask but into the two
    axes we supply instead of creating a new figure each time.
    """
    import os
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling

    img_path = os.path.join(root_dir, str(row["image"]))
    img_path = img_path[:-4]
    msk_path = os.path.join(root_dir, str(row["label"]))

    with rasterio.open(img_path) as src:
        if rgb_bands is None:
            band = src.read(1, out_dtype="float32", resampling=Resampling.nearest)
            p2, p98 = np.nanpercentile(band, (2, 98))
            band = np.clip(band, p2, p98)
            img_rgb = (band - p2) / (p98 - p2 + 1e-6)
            img_rgb = np.repeat(img_rgb[None, ...], 3, axis=0)
        else:
            rgb = src.read(rgb_bands, out_dtype="float32", resampling=Resampling.nearest)
            for i in range(rgb.shape[0]):
                p2, p98 = np.nanpercentile(rgb[i], (2, 98))
                rgb[i] = np.clip(rgb[i], p2, p98)
                rgb[i] = (rgb[i] - p2) / (p98 - p2 + 1e-6)
            img_rgb = rgb
    img_rgb = np.moveaxis(img_rgb, 0, -1)

    with rasterio.open(msk_path) as msk:
        mask = msk.read(1)

    ax_img.imshow(img_rgb)
    ax_img.axis("off")

    ax_msk.imshow(mask, cmap=mask_cmap, alpha=alpha, interpolation="nearest")
    ax_msk.axis("off")

import math, pathlib, numpy as np, pandas as pd

def split_by_rscore(
    csv_path: str | pathlib.Path,
    out_dir: str | pathlib.Path = ".",
    class_col: str = "Class",
    score_col: str = "Rscore",
    train_frac: float = 0.60,
    val_frac: float   = 0.20,
    test_frac: float  = 0.20,
    seed: int | None  = 42,
    na_goes_last: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    if not math.isclose(train_frac + val_frac + test_frac, 1.0, abs_tol=1e-6):
        raise ValueError("train_frac + val_frac + test_frac must equal 1.0")

    rng = np.random.default_rng(seed)
    csv_path, out_dir = map(pathlib.Path, (csv_path, out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = df.drop(columns=["assignment_id", "rscore", "rscore2", "x", "y", "farea", "nflds"], errors="ignore")

    # NEW: remove rows with any NaNs left
    df = df.dropna().reset_index(drop=True)

    # Treat NaNs in the score_col specifically (though now unlikely after dropna)
    fill_val = np.inf if not na_goes_last else -np.inf
    df[score_col] = df[score_col].fillna(fill_val)

    train_parts, val_parts, test_parts = [], [], []

    for cls, grp in df.groupby(class_col, sort=False):
        grp = grp.sample(frac=1.0, random_state=rng.integers(0, 1e9))
        grp = grp.sort_values(score_col, ascending=False)
        n = len(grp)
        n_train = int(round(n * train_frac))
        n_val   = int(round(n * val_frac))

        train_parts.append(grp.iloc[:n_train])
        val_parts  .append(grp.iloc[n_train : n_train + n_val])
        test_parts .append(grp.iloc[n_train + n_val :])

    def _finalise(split_parts: list[pd.DataFrame], tag: str) -> pd.DataFrame:
        df_split = (
            pd.concat(split_parts, ignore_index=True)
              .sample(frac=1.0, random_state=rng.integers(0, 1e9))  # shuffle rows
              .assign(usage=tag)
        )
        return df_split

    train_df = _finalise(train_parts, "train")
    val_df   = _finalise(val_parts,   "validate")
    test_df  = _finalise(test_parts,  "test")

    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "validate.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    def _desc(d: pd.DataFrame) -> str:
        m, md = d[score_col].replace([-np.inf, np.inf], np.nan).agg(["mean", "median"])
        return f"n={len(d):6,d} • mean={m:6.3f} • median={md:6.3f}"

    print(
        f"Split complete → files in {out_dir}\n"
        f"  ▸ train    { _desc(train_df) }\n"
        f"  ▸ validate { _desc(val_df)   }\n"
        f"  ▸ test     { _desc(test_df)  }"
    )

    return train_df, val_df, test_df
