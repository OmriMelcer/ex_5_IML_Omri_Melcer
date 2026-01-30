import torch
import torch.nn as nn
from dataset import EuropeDataset
import matplotlib.pyplot as plt
import numpy as np

def normalize_tensor(tensor, d):
    """
    Normalize the input tensor along the specified axis to have a mean of 0 and a std of 1.
    
    Parameters:
        tensor (torch.Tensor): Input tensor to normalize.
        d (int): Axis along which to normalize.
    
        torch.Tensor: Normalized tensor.
    Returns:
    """
    mean = torch.mean(tensor, dim=d, keepdim=True)
    std = torch.std(tensor, dim=d, keepdim=True)
    normalized = (tensor - mean) / std
    return normalized


class GMM(nn.Module):
    def __init__(self, n_components):
        """
        Gaussian Mixture Model in 2D using PyTorch.

        Args:
            n_components (int): Number of Gaussian components.
        """
        super().__init__()        
        self.n_components = n_components

        # Mixture weights (logits to be softmaxed)
        self.weights = nn.Parameter(torch.randn(n_components))

        # Means of the Gaussian components (n_components x 2 for 2D data)
        self.means = nn.Parameter(torch.randn(n_components, 2))

        # Log of the variance of the Gaussian components (n_components x 2 for 2D data)
        self.log_variances = nn.Parameter(torch.zeros(n_components, 2))  # Log-variances (diagonal covariance)




    def forward(self, X):
        """
        Compute the log-likelihood of the data.
        Args:
            X (torch.Tensor): Input data of shape (n_samples, 2).

        Returns:
            torch.Tensor: Log-likelihood of shape (n_samples,).
        """
        # log softmax for to to get log probabilities. 
        log_weights = nn.functional.log_softmax(self.weights, dim=0)  # (n_components,)
        # Expand dimensions for broadcasting
        X_expanded = X.unsqueeze(1)  # (n_samples, 1, 2)
        means_expanded = self.means.unsqueeze(0) # (1, n_components, 2)
        variances = torch.exp(self.log_variances).unsqueeze(0)  # (1, n_components, 2)
        diff_sq = ((X_expanded - means_expanded) ** 2 / variances).sum(dim=2)  # (n_samples, n_components)
        log_det = self.log_variances.sum(dim=1)  # (n_components,)
        log_2_pi = 2 * torch.log(torch.tensor(2 * torch.pi)) #scalar
        log_gaussian = -0.5 * ( log_2_pi+ log_det + diff_sq)
        log_likelihood = torch.logsumexp(log_weights + log_gaussian, dim=1)  # (n_samples,)
        return log_likelihood

    def loss_function(self, log_likelihood):
        """
        Compute the negative log-likelihood loss.
        Args:
            log_likelihood (torch.Tensor): Log-likelihood of shape (n_samples,).

        Returns:
            torch.Tensor: Negative log-likelihood.
        """
        #### YOUR CODE GOES HERE ####
        return -torch.mean(log_likelihood)


    def sample(self, n_samples):
        """
        Generate samples from the GMM model.
        Args:
            n_samples (int): Number of samples to generate.

        Returns:
            torch.Tensor: Generated samples of shape (n_samples, 2).
        """
        #### YOUR CODE GOES HERE ####
        K_for_sample_i = torch.multinomial(nn.functional.softmax(self.weights, dim=0), n_samples, replacement=True) # (n_samples,)
        means_selected = self.means[K_for_sample_i]  # (n_samples, 2)
        variances_selected = torch.exp(self.log_variances)[K_for_sample_i]  # (n_samples, 2)
        samples = torch.randn(n_samples, 2) * torch.sqrt(variances_selected) + means_selected
        return samples
    
    def conditional_sample(self, n_samples, label):
        """
        Generate samples from a specific uniform component.
        Args:
            n_samples (int): Number of samples to generate.
            label (int): Component index.

        Returns:
            torch.Tensor: Generated samples of shape (n_samples, 2).
        """
        means_selected = self.means[label]  # (1, 2)
        variances_selected = torch.exp(self.log_variances)[label]  # (1, 2)
        samples = torch.randn(n_samples, 2) * torch.sqrt(variances_selected) + means_selected
        return samples



class UMM(nn.Module):
    def __init__(self, n_components):
        """
        Uniform Mixture Model in 2D using PyTorch.

        Args:
            n_components (int): Number of uniform components.
        """
        super().__init__()        
        self.n_components = n_components

        # Mixture weights (logits to be softmaxed)
        self.weights = nn.Parameter(torch.randn(n_components))

        # Center value of the uniform components (n_components x 2 for 2D data)
        self.centers = nn.Parameter(torch.randn(n_components, 2))

        # Log of size of the uniform components (n_components x 2 for 2D data)
        self.log_sizes = nn.Parameter(torch.log(torch.ones(n_components, 2) + torch.rand(n_components, 2)*0.2))


    def forward(self, X):
        """
        Compute the log-likelihood of the data.
        Args:
            X (torch.Tensor): Input data of shape (n_samples, 2).

        Returns:
            torch.Tensor: Log-likelihood of shape (n_samples,).
        """
        #### YOUR CODE GOES HERE ####
        
    
    
    def loss_function(self, log_likelihood):
        """
        Compute the negative log-likelihood loss.
        Args:
            log_likelihood (torch.Tensor): Log-likelihood of shape (n_samples,).

        Returns:
            torch.Tensor: Negative log-likelihood.
        """
        #### YOUR CODE GOES HERE ####


    def sample(self, n_samples):
        """
        Generate samples from the UMM model.
        Args:
            n_samples (int): Number of samples to generate.

        Returns:
            torch.Tensor: Generated samples of shape (n_samples, 2).
        """
        #### YOUR CODE GOES HERE ####

    def conditional_sample(self, n_samples, label):
        """
        Generate samples from a specific uniform component.
        Args:
            n_samples (int): Number of samples to generate.
            label (int): Component index.

        Returns:
            torch.Tensor: Generated samples of shape (n_samples, 2).
        """
        #### YOUR CODE GOES HERE ####


def train_and_plot_gmm_umm (train_dataset, test_dataset, num_epochs=50, learning_rate = 0.01, Model_type = type(GMM)) :
    """
    Train GMM and UMM models and plot the results.

    Args:
        train_dataset (EuropeDataset): Training dataset.
        test_dataset (EuropeDataset): Testing dataset.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizers.
    """
    #TODO: think of lr for each model - what theory says about it.

    import os
    os.makedirs('plots', exist_ok=True)

    def plot_1000_samples(model, ax, title):
        """Plot 1000 samples from the GMM model"""
        model.eval()
        with torch.no_grad():
            samples = model.sample(1000).detach().numpy()
        ax.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=5)
        ax.set_title(title)
        ax.set_xlabel('Longitude (normalized)')
        ax.set_ylabel('Latitude (normalized)')
        ax.grid(True, alpha=0.3)

    def plot_100_of_each_conditional_samples(model, ax, title):
        """Plot 100 samples from each component, colored by component"""
        model.eval()
        with torch.no_grad():
            all_samples = []
            colors = []
            for k in range(model.n_components):
                samples = model.conditional_sample(100, k).detach().numpy()
                all_samples.append(samples)
                colors.extend([k] * 100)
            all_samples = np.vstack(all_samples)

        scatter = ax.scatter(all_samples[:, 0], all_samples[:, 1],
                            c=colors, cmap='tab10', alpha=0.6, s=5)
        ax.set_title(title)
        ax.set_xlabel('Longitude (normalized)')
        ax.set_ylabel('Latitude (normalized)')
        ax.grid(True, alpha=0.3)
        if model.n_components <= 10:
            plt.colorbar(scatter, ax=ax, label='Component')

    def plot_mean_log_likelihoods_over_epochs(epochs_log_likelihoods_train, epochs_log_likelihoods_test, model_name, init_type):
        """Plot mean log-likelihoods vs epoch for train and test"""
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = sorted(epochs_log_likelihoods_train.keys())
        train_vals = [epochs_log_likelihoods_train[e] for e in epochs]
        test_vals = [epochs_log_likelihoods_test[e] for e in epochs]

        ax.plot([e+1 for e in epochs], train_vals, 'o-', label='Train', linewidth=2, markersize=6)
        ax.plot([e+1 for e in epochs], test_vals, 's-', label='Test', linewidth=2, markersize=6)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean Log-Likelihood')
        ax.set_title(f'{model_name} - Mean Log-Likelihood vs Epoch ({init_type})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'plots/{model_name}_log_likelihood_{init_type}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: plots/{model_name}_log_likelihood_{init_type}.png")

    def get_means_of_countries_from_train_dataset(train_dataset):
        """Returns the means of each country from the train dataset"""
        classes = train_dataset.labels.unique()
        means = torch.stack([train_dataset.features[train_dataset.labels==c].mean(dim=0) for c in classes], dim=0)
        return means

    model_name = Model_type.__name__
    classes = train_dataset.labels.unique().size(0)
    models_n_components = [1,5,10,classes,classes]

    epochs_to_save = [0,10,20,30,40,49]
    epochs_mean_log_liklihood_train = {epoch: 0 for epoch in epochs_to_save}
    epochs_mean_log_liklihood_test = {epoch: 0 for epoch in epochs_to_save}
    dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4096, shuffle=True, num_workers=0)

    for i,n_components in enumerate(models_n_components):
        is_last_model = (i == len(models_n_components) - 1)
        init_type = "country_init" if is_last_model else "random_init"

        print(f"\n{'='*60}")
        print(f"Training {model_name} with {n_components} components ({init_type})")
        print(f"{'='*60}")

        model = Model_type(n_components=n_components)
        if is_last_model:
            # Initialize the means to the means of each country from the train dataset
            with torch.no_grad():
                model.means.copy_(get_means_of_countries_from_train_dataset(train_dataset))

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            num_batches = 0
            for batch_features, _ in dataset_loader:
                optimizer.zero_grad()
                log_likelihood = model(batch_features)
                loss = model.loss_function(log_likelihood)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

            # Track log-likelihoods for n_components == classes
            if n_components == classes and epoch in epochs_to_save:
                model.eval()
                with torch.no_grad():
                    train_log_likelihood = model(train_dataset.features)
                    test_log_likelihood = model(test_dataset.features)
                    epochs_mean_log_liklihood_test[epoch] = torch.mean(test_log_likelihood).item()
                    epochs_mean_log_liklihood_train[epoch] = torch.mean(train_log_likelihood).item()

                # Plot samples at specific epochs for n_components == classes
                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                plot_1000_samples(model, axes[0], f'1000 Samples - Epoch {epoch+1}')
                plot_100_of_each_conditional_samples(model, axes[1], f'100 Samples per Component - Epoch {epoch+1}')
                fig.suptitle(f'{model_name} with {n_components} components - Epoch {epoch+1} ({init_type})', fontsize=14)
                plt.tight_layout()
                plt.savefig(f'plots/{model_name}_k{n_components}_epoch{epoch+1}_{init_type}.png', dpi=150, bbox_inches='tight')
                plt.close()
                print(f"Saved plot: plots/{model_name}_k{n_components}_epoch{epoch+1}_{init_type}.png")

        # After training: plot final samples for all models
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        plot_1000_samples(model, axes[0], '1000 Samples')
        plot_100_of_each_conditional_samples(model, axes[1], '100 Samples per Component')
        fig.suptitle(f'{model_name} with {n_components} components - Final ({init_type})', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'plots/{model_name}_k{n_components}_final_{init_type}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: plots/{model_name}_k{n_components}_final_{init_type}.png")

        # Evaluation on test set
        model.eval()
        with torch.no_grad():
            test_log_likelihood = model(test_dataset.features)
            test_loss = model.loss_function(test_log_likelihood)
            print(f"Test Loss for {n_components} components: {test_loss.item():.4f}")

    # Plot log-likelihood curves for the two n_classes models
    plot_mean_log_likelihoods_over_epochs(epochs_mean_log_liklihood_train, epochs_mean_log_liklihood_test,
                                         model_name, "random_vs_country_init")
    
# Example Usage
if __name__ == "__main__":

    torch.manual_seed(42)
    train_dataset = EuropeDataset('train.csv')
    test_dataset = EuropeDataset('test.csv')

    num_epochs = 50
    learning_rate_for_GMM = 0.01
    learning_rate_for_UMM = 0.001

    # Normalize the datasets
    train_dataset.features = normalize_tensor(train_dataset.features, d=0)
    test_dataset.features = normalize_tensor(test_dataset.features, d=0)

    # Train and plot GMM
    print("\n" + "="*70)
    print("TRAINING GMM MODELS")
    print("="*70)
    train_and_plot_gmm_umm(train_dataset, test_dataset,
                          num_epochs=num_epochs,
                          learning_rate=learning_rate_for_GMM,
                          Model_type=GMM)




