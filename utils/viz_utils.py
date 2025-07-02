# viz_utils.py — processing script

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import os

def load_image_as_np(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))

def plot_slice(image_np, title="", cmap="gray"):
    middle = image_np.shape[0] // 2
    plt.imshow(image_np[middle, :, :], cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.show()

def plot_overlay(image_np, mask_np, alpha=0.4):
    middle = image_np.shape[0] // 2
    plt.imshow(image_np[middle, :, :], cmap="gray")
    plt.imshow(mask_np[middle, :, :], cmap="jet", alpha=alpha)
    plt.title("Overlay")
    plt.axis("off")
    plt.show()

def visualize_subject(t1_path, seg_path=None):
    img_np = load_image_as_np(t1_path)
    plot_slice(img_np, title="T2 Image")
    if seg_path:
        seg_np = load_image_as_np(seg_path)
        plot_overlay(img_np, seg_np)

def visualize_subject(t1_path, t1ce_path, t2f_path, t2w_path, seg_path=None):
    # Load all images
    t1_np = load_image_as_np(t1_path)
    t1ce_np = load_image_as_np(t1ce_path)
    t2f_np = load_image_as_np(t2f_path)
    t2w_np = load_image_as_np(t2w_path)
    seg_np = load_image_as_np(seg_path) if seg_path else None

    middle = t1_np.shape[0] // 2

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    # First row: images and segmentation mask
    images = [t1_np, t1ce_np, t2f_np, t2w_np, seg_np]
    titles = ["T1", "T1C", "T2F", "T2W", "Segmentation"]
    cmaps = ["gray", "gray", "gray", "gray", "jet"]

    for i, (img, title, cmap) in enumerate(zip(images, titles, cmaps)):
        if img is not None:
            axes[0, i].imshow(img[middle, :, :], cmap=cmap)
        axes[0, i].set_title(title)
        axes[0, i].axis("off")

    # Second row: overlays with segmentation mask labels in distinct colors
    if seg_np is not None:
        # Define a color for each label (extend as needed)
        label_colors = {
            0: [0, 0, 0],       # background (black)
            1: [1, 0, 0],       # label 1: red
            2: [0, 1, 0],       # label 2: green
            3: [0, 0, 1],       # label 3: blue
            4: [1, 1, 0],       # label 4: yellow
            5: [1, 0, 1],       # label 5: magenta
            6: [0, 1, 1],       # label 6: cyan
            # Add more if needed
        }
        seg_slice = seg_np[middle]
        color_mask = np.zeros(seg_slice.shape + (3,), dtype=np.float32)
        for label, color in label_colors.items():
            color_mask[seg_slice == label] = color

    for i, img in enumerate([t1_np, t1ce_np, t2f_np, t2w_np]):
        axes[1, i].imshow(img[middle, :, :], cmap="gray")
        if seg_np is not None:
            axes[1, i].imshow(color_mask, alpha=0.5)
            axes[1, i].set_title(f"{titles[i]} + Seg", color="black")
        else:
            axes[1, i].set_title(titles[i], color="black")
        axes[1, i].axis("off")

    # Last cell in second row: just segmentation mask in RGB
    if seg_np is not None:
        axes[1, 4].imshow(color_mask)
        axes[1, 4].set_title("Segmentation", color="black")
    else:
        axes[1, 4].set_title("Segmentation", color="black")
    axes[1, 4].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    base_dir = "/home/ajay/Documents/auto-rapno/data/raw/BraTS-PEDs2024_Training/BraTS-PED-00001-000"
    # Find all relevant files in the directory
    files = os.listdir(base_dir)
    t1_path = os.path.join(base_dir, next(f for f in files if "-t1n.nii.gz" in f))
    t1c_path = os.path.join(base_dir, next(f for f in files if "-t1c.nii.gz" in f))
    t2f_path = os.path.join(base_dir, next(f for f in files if "-t2f.nii.gz" in f))
    t2w_path = os.path.join(base_dir, next(f for f in files if "-t2w.nii.gz" in f))
    seg_files = [f for f in files if "-seg.nii.gz" in f]
    seg_path = os.path.join(base_dir, seg_files[0]) if seg_files else None

    # Print shapes
    t1_np = load_image_as_np(t1_path)
    t1c_np = load_image_as_np(t1c_path)
    t2f_np = load_image_as_np(t2f_path)
    t2w_np = load_image_as_np(t2w_path)
    print("T1 shape:", t1_np.shape)
    print("T1C shape:", t1c_np.shape)
    print("T2F shape:", t2f_np.shape)
    print("T2W shape:", t2w_np.shape)
    if seg_path:
        seg_np = load_image_as_np(seg_path)
        print("Segmentation shape:", seg_np.shape)
    else:
        seg_np = None

    visualize_subject(t1_path, t1c_path, t2f_path, t2w_path, seg_path)