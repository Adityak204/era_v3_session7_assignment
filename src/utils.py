import torch
import random
import matplotlib.pyplot as plt


def plot_random_mnist_images(test_loader, num_images=49):
    # Gather all images and labels from the test loader
    all_images, all_labels = [], []
    for images, labels in test_loader:
        all_images.append(images)
        all_labels.append(labels)

    # Combine all batches into single tensors
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Select random indices
    indices = random.sample(range(len(all_images)), num_images)

    # Plot the selected images
    plt.figure(figsize=(10, 10))
    for i, idx in enumerate(indices):
        plt.subplot(7, 7, i + 1)  # Adjust grid size to fit images
        plt.imshow(all_images[idx].squeeze(), cmap="gray")
        plt.axis("off")
        plt.title(str(all_labels[idx].item()))

    plt.tight_layout()
    plt.show()
