import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# Create compact plots from existing individual plots

def create_k_comparison_plot():
    """Combine k=1, k=5, k=10 final plots into one figure"""
    fig = plt.figure(figsize=(14, 18))

    plot_files = [
        ('plots/GMM_k1_final_random_init.png', 'k=1'),
        ('plots/GMM_k5_final_random_init.png', 'k=5'),
        ('plots/GMM_k10_final_random_init.png', 'k=10')
    ]

    for idx, (filename, title) in enumerate(plot_files, 1):
        ax = fig.add_subplot(3, 1, idx)
        img = mpimg.imread(filename)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold')

    fig.suptitle('GMM Models Comparison - Different Number of Components', fontsize=18, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig('plots/GMM_k_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: plots/GMM_k_comparison.png")


def create_random_init_epochs_plot():
    """Combine k=33 random init epoch plots into one figure"""
    fig = plt.figure(figsize=(20, 12))

    epochs = [1, 11, 21, 31, 41, 50]

    for idx, epoch in enumerate(epochs, 1):
        filename = f'plots/GMM_k33_epoch{epoch}_random_init.png'
        ax = fig.add_subplot(2, 3, idx)
        img = mpimg.imread(filename)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'Epoch {epoch}', fontsize=14, fontweight='bold')

    fig.suptitle('GMM k=33 Training Progress - Random Initialization', fontsize=18, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig('plots/GMM_k33_random_init_epochs.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: plots/GMM_k33_random_init_epochs.png")


def create_country_init_epochs_plot():
    """Combine k=33 country init epoch plots into one figure"""
    fig = plt.figure(figsize=(20, 12))

    epochs = [1, 11, 21, 31, 41, 50]

    for idx, epoch in enumerate(epochs, 1):
        filename = f'plots/GMM_k33_epoch{epoch}_country_init.png'
        ax = fig.add_subplot(2, 3, idx)
        img = mpimg.imread(filename)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'Epoch {epoch}', fontsize=14, fontweight='bold')

    fig.suptitle('GMM k=33 Training Progress - Country Initialization', fontsize=18, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig('plots/GMM_k33_country_init_epochs.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: plots/GMM_k33_country_init_epochs.png")


if __name__ == "__main__":
    print("Creating compact plots...")
    create_k_comparison_plot()
    create_random_init_epochs_plot()
    create_country_init_epochs_plot()
    print("\nDone! Created 3 compact plots.")
