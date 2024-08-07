import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from train import DiTLightning, TrainerConfig
from diffusers import DDPMScheduler
from torchvision.utils import make_grid, save_image
from dataset import ShardedLeRobotDataset


def load_model(checkpoint_path):
    trainer_config = TrainerConfig(
        batch_size=64,
        lr=3e-4,
        obs_horizon=2,
        pred_horizon=16,
        image_size=96,
        patch_size=8,
        action_dim=2,
        n_dit_layer=8,
        n_dit_head=8,
        n_dit_embd=512,
        n_ocot_embd=384,
    )
    model = DiTLightning.load_from_checkpoint(
        checkpoint_path, trainer_config=trainer_config
    )
    model.eval()
    return model


def sample_dit(
    model,
    noise_scheduler,
    observation_images,
    observation_states,
    n_inference_steps=50,
    device="cuda",
):
    noise_scheduler.set_timesteps(n_inference_steps)
    batch_size = observation_images.shape[0]

    # Start from random noise
    x = torch.randn(
        (batch_size, model.trainer_config.pred_horizon, model.trainer_config.action_dim)
    ).to(device)

    step_history = [x.detach().cpu()]

    model.to(device)
    observation_images = observation_images.to(device)
    observation_states = observation_states.to(device) / 512

    with torch.no_grad():
        for t in tqdm(noise_scheduler.timesteps, desc="Sampling"):
            model_input = noise_scheduler.scale_model_input(x, t)
            timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # Predict the noise residual
            noise_pred = model.net(
                model_input, observation_images, observation_states, timesteps
            )

            # Compute the previous noisy sample x_t -> x_t-1
            x = noise_scheduler.step(noise_pred, t, x).prev_sample
            step_history.append(x.detach().cpu())

    return step_history


def scale_state(state, original_range=(0, 512), target_range=(0, 96)):
    return (state - original_range[0]) / (original_range[1] - original_range[0]) * (
        target_range[1] - target_range[0]
    ) + target_range[0]


def overlay_prediction(
    image, current_state, observation_states, baseline_actions, predicted_actions
):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    # Scale and plot current state
    scaled_current_state = scale_state(current_state)
    plt.scatter(
        scaled_current_state[0],
        scaled_current_state[1],
        c="red",
        s=50,
        label="Current State",
    )

    # Scale and plot observed states in grey
    scaled_observation_states = scale_state(np.array(observation_states))
    for i, state in enumerate(scaled_observation_states):
        plt.scatter(state[0], state[1], c="grey", s=30, alpha=0.7)
        if i > 0:
            plt.plot(
                [scaled_observation_states[i - 1][0], state[0]],
                [scaled_observation_states[i - 1][1], state[1]],
                c="grey",
                linestyle=":",
                alpha=0.5,
            )

    # Scale and plot baseline actions in grey
    scaled_baseline_actions = scale_state(np.array(baseline_actions))
    for i, state in enumerate(scaled_baseline_actions):
        plt.scatter(state[0], state[1], c="grey", s=30, alpha=0.7)
        if i > 0:
            plt.plot(
                [scaled_baseline_actions[i - 1][0], state[0]],
                [scaled_baseline_actions[i - 1][1], state[1]],
                c="grey",
                linestyle="-",
                alpha=0.5,
            )

    # Scale and plot predicted actions with color gradient
    colors = plt.cm.rainbow(np.linspace(0, 1, len(predicted_actions)))
    scaled_predicted_actions = scale_state(np.array(predicted_actions))
    for i, (state, color) in enumerate(zip(scaled_predicted_actions, colors)):
        plt.scatter(state[0], state[1], c=[color], s=30, alpha=0.7)
        if i > 0:
            plt.plot(
                [scaled_predicted_actions[i - 1][0], state[0]],
                [scaled_predicted_actions[i - 1][1], state[1]],
                c=color,
                linestyle="-",
                alpha=0.5,
            )

    plt.title("State Prediction Overlay")
    plt.legend()
    plt.axis("off")
    plt.tight_layout()

    # Convert plot to image
    fig = plt.gcf()
    fig.canvas.draw()
    plot_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()

    return plot_image


def create_image_grid(images):
    """Create a grid of images."""
    # images shape: (batch_size, num_frames, channels, height, width)
    b, f, c, h, w = images.shape
    images = images.view(b * f, c, h, w)
    grid = make_grid(images, nrow=f)
    return grid.permute(1, 2, 0).cpu().numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = load_model(
        "checkpoints_dp_scaled/dp-epoch=14-step=4815-train_loss=0.05-val_loss=0.05.ckpt"
    )
    model.to(device)

    # Initialize the noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",
    )

    # Load a sample from your dataset
    dataset = ShardedLeRobotDataset(
        "lerobot/pusht", prediction_horizon=16, observation_horizon=2
    )
    for idx in range(len([1000] * 5)):
        sample = dataset[idx]

        observation_images = sample["observation_images"].unsqueeze(0).to(device)
        observation_states = sample["observation_states"].unsqueeze(0).to(device)
        current_state = observation_states[0, -1].cpu().numpy()
        baseline_actions = sample["prediction_actions"].to(device)

        # Sample from the model
        sampled_actions = sample_dit(
            model,
            noise_scheduler,
            observation_images,
            observation_states,
            n_inference_steps=50,
            device=device,
        )

        # Get the final predicted actions
        predicted_actions = sampled_actions[-1].squeeze().cpu().numpy() * 512

        # Use only the last observation image for visualization
        last_image = observation_images[0, -1].permute(1, 2, 0).cpu().numpy()

        # Visualize the results
        image = overlay_prediction(
            last_image,
            current_state,
            observation_states.squeeze().cpu().numpy(),
            baseline_actions.cpu().numpy(),
            predicted_actions,
        )

        # Save the image
        plt.imsave(f"action_prediction_comparison_{idx}.png", image)


if __name__ == "__main__":
    main()
