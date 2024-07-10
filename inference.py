import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from train import UNETLightning


def sample_ddpm(
    model,
    noise_scheduler,
    n_inference_steps=50,
    batch_size=8,
    image_size=(1, 28, 28),
    device="cuda",
    class_labels=None,
):
    """
    Sample images using the DDPM model with class conditioning.

    Args:
    model (nn.Module): The trained UNet model
    noise_scheduler (DDPMScheduler): The noise scheduler
    n_inference_steps (int): Number of inference steps
    batch_size (int): Number of images to generate
    image_size (tuple): Size of the image (channels, height, width)
    device (str): Device to run the model on ('cuda' or 'cpu')
    class_labels (torch.Tensor): Tensor of class labels for conditioning

    Returns:
    tuple: (step_history, pred_output_history)
    """
    noise_scheduler.set_timesteps(n_inference_steps)

    # Start from random noise
    x = torch.randn(batch_size, *image_size).to(device)
    step_history = [x.detach().cpu()]
    pred_output_history = []

    if class_labels is None:
        class_labels = torch.randint(0, 10, (batch_size,)).to(device)
    else:
        class_labels = class_labels.to(device)

    model.eval()
    with torch.no_grad():
        for t in noise_scheduler.timesteps:
            # Expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([x] * 2)
            latent_model_input = noise_scheduler.scale_model_input(
                latent_model_input, t
            )

            timesteps = torch.full(
                (batch_size * 2,), t, device=device, dtype=torch.long
            )
            class_labels_input = class_labels.repeat(2)

            # Predict the noise residual
            noise_pred = model(latent_model_input, timesteps, class_labels_input)
            noise_pred = noise_pred[:batch_size]  # Take only the unconditional output

            # Compute the previous noisy sample x_t -> x_t-1
            x = noise_scheduler.step(noise_pred, t, x).prev_sample

            step_history.append(x.detach().cpu())
            pred_output_history.append(noise_pred.detach().cpu())

    return step_history, pred_output_history


def visualize_sampling_process(
    step_history,
    pred_output_history,
    class_labels,
    save_path="ddpm_sampling_visualization.png",
):
    """
    Visualize the sampling process in a single image.

    Args:
    step_history (list): List of tensors representing the sampling steps
    pred_output_history (list): List of tensors representing the model predictions
    class_labels (torch.Tensor): Tensor of class labels used for conditioning
    save_path (str): Path to save the visualization
    """
    n_steps = len(pred_output_history)
    n_samples = step_history[0].shape[0]

    fig, axs = plt.subplots(n_steps, 2, figsize=(12, 2 * n_steps))
    plt.subplots_adjust(hspace=0.3)

    axs[0, 0].set_title("Denoised Image")
    axs[0, 1].set_title("Predicted Noise")

    for i in range(n_steps):
        # Plot denoised image
        axs[i, 0].imshow(make_grid(step_history[i]), cmap="gray")
        axs[i, 0].axis("off")

        # Plot predicted noise
        axs[i, 1].imshow(make_grid(pred_output_history[i]), cmap="viridis")
        axs[i, 1].axis("off")

        axs[i, 0].set_ylabel(f"Step {n_steps - i}")

    # Add class labels to the plot
    fig.suptitle(f"Generated Digits: {class_labels.tolist()}", fontsize=16)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")


def make_grid(tensor, nrow=8, padding=2):
    """
    Make a grid of images.

    Args:
    tensor (Tensor): 4D mini-batch Tensor of shape (B x C x H x W)
    nrow (int): Number of images displayed in each row of the grid
    padding (int): Amount of padding

    Returns:
    numpy.ndarray: Grid of images
    """
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(np.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[2] + padding), int(tensor.shape[3] + padding)
    grid = np.zeros(
        (height * ymaps + padding, width * xmaps + padding), dtype=np.float32
    )
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            image = tensor[k].squeeze().numpy()
            height_start = y * height + padding
            height_end = height_start + height - padding
            width_start = x * width + padding
            width_end = width_start + width - padding
            grid[height_start:height_end, width_start:width_end] = image
            k += 1
    return grid


if __name__ == "__main__":
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model from checkpoint
    checkpoint_path = "diffusion-policy/xh3xputh/checkpoints/epoch=5-step=2814.ckpt"  # Replace with your checkpoint path
    model = UNETLightning.load_from_checkpoint(checkpoint_path)
    model = model.to(device)
    print("Model loaded successfully")

    # Create the noise scheduler
    noise_scheduler = model.noise_scheduler

    # Set inference parameters
    n_inference_steps = 15
    batch_size = 8
    image_size = (1, 28, 28)  # Adjust based on your model's expected input size

    # Set class labels for conditioning (generate digits 0-7)
    class_labels = torch.arange(8).to(device)

    # Perform sampling
    step_history, pred_output_history = sample_ddpm(
        model.net,
        noise_scheduler,
        n_inference_steps,
        batch_size,
        image_size,
        device,
        class_labels,
    )
    print("Sampling completed")

    # Visualize and save the results
    visualize_sampling_process(
        step_history,
        pred_output_history,
        class_labels,
        "ddpm_class_conditioned_sampling_visualization.png",
    )
    print("Process completed")
