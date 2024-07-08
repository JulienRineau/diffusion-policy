import torch
import torchvision
import cv2
import os
import numpy as np
from pytorch_lightning import LightningModule


class UNETLightning(LightningModule):
    # Your UNET model definition here
    # This is a placeholder - replace with your actual model implementation
    pass


def sample_unet(model, n_steps=5, batch_size=8, image_size=(1, 28, 28), device="cuda"):
    """
    Sample images using the UNet model with a specified sampling strategy.

    Args:
    model (nn.Module): The trained UNet model
    n_steps (int): Number of sampling steps
    batch_size (int): Number of images to generate
    image_size (tuple): Size of the image (channels, height, width)
    device (str): Device to run the model on ('cuda' or 'cpu')

    Returns:
    tuple: (step_history, pred_output_history)
    """
    x = torch.rand(batch_size, *image_size).to(device)  # Start from random noise
    step_history = [x.detach().cpu()]
    pred_output_history = []

    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # No need to track gradients during inference
        for i in range(n_steps):
            pred = model(x)  # Predict the denoised x0
            pred_output_history.append(
                pred.detach().cpu()
            )  # Store model output for saving
            mix_factor = 1 / (n_steps - i)  # How much we move towards the prediction
            x = x * (1 - mix_factor) + pred * mix_factor  # Move part of the way there
            step_history.append(x.detach().cpu())  # Store step for saving

    return step_history, pred_output_history


def save_sampling_process(
    step_history, pred_output_history, save_dir="sampling_results"
):
    """
    Save the sampling process as images using OpenCV.

    Args:
    step_history (list): List of tensors representing the sampling steps
    pred_output_history (list): List of tensors representing the model predictions
    save_dir (str): Directory to save the images
    """
    os.makedirs(save_dir, exist_ok=True)
    n_steps = len(pred_output_history)

    for i in range(n_steps):
        # Process and save model input (x)
        x_grid = torchvision.utils.make_grid(step_history[i])[0].numpy()
        x_image = (x_grid.clip(0, 1) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, f"step_{i}_input.png"), x_image)

        # Process and save model prediction
        pred_grid = torchvision.utils.make_grid(pred_output_history[i])[0].numpy()
        pred_image = (pred_grid.clip(0, 1) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, f"step_{i}_prediction.png"), pred_image)

    print(f"Images saved in {save_dir}")


if __name__ == "__main__":
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model from checkpoint
    checkpoint_path = (
        "path/to/your/checkpoint.ckpt"  # Replace with your checkpoint path
    )
    model = UNETLightning.load_from_checkpoint(checkpoint_path)
    model = model.to(device)
    print("Model loaded successfully")

    # Set inference parameters
    n_steps = 5
    batch_size = 8
    image_size = (1, 28, 28)  # Adjust based on your model's expected input size

    # Perform sampling
    step_history, pred_output_history = sample_unet(
        model, n_steps, batch_size, image_size, device
    )
    print("Sampling completed")

    # Save the results
    save_dir = "unet_sampling_results"
    save_sampling_process(step_history, pred_output_history, save_dir)
    print("Process completed")
