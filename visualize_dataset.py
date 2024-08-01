import logging
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


def normalize_image(image):
    """Normalize float image to uint8."""
    if image.dtype != np.uint8:
        if np.max(image) <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = (
                (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
            ).astype(np.uint8)
    return image


def scale_state(state, original_range=(0, 512), target_range=(0, 96)):
    """Scale the state values from the original range to the target range."""
    return (state - original_range[0]) / (original_range[1] - original_range[0]) * (
        target_range[1] - target_range[0]
    ) + target_range[0]


def overlay_prediction(image, current_state, observation_states, prediction_actions):
    """Overlay prediction on the image."""
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

    # Scale and plot future actions with color gradient
    colors = plt.cm.rainbow(np.linspace(0, 1, len(prediction_actions)))
    scaled_prediction_actions = scale_state(np.array(prediction_actions))
    for i, (state, color) in enumerate(zip(scaled_prediction_actions, colors)):
        plt.scatter(state[0], state[1], c=[color], s=30, alpha=0.7)
        if i > 0:
            plt.plot(
                [scaled_prediction_actions[i - 1][0], state[0]],
                [scaled_prediction_actions[i - 1][1], state[1]],
                c=color,
                linestyle=":",
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


def generate_frames(dataset, start_index=0, end_index=None):
    """Generate frames for the video."""
    if end_index is None:
        end_index = len(dataset)

    frames = []
    for i in tqdm(
        range(start_index, min(end_index, len(dataset))), desc="Generating frames"
    ):
        sample = dataset[i]
        image = sample["observation.image"]
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).numpy()
        image = normalize_image(image)

        current_state = sample["observation.state"]
        observation_states = sample["observation_states"]
        prediction_actions = sample["prediction_actions"]

        overlay_image = overlay_prediction(
            image, current_state, observation_states, prediction_actions
        )
        frames.append(overlay_image)

    return frames


def create_video_from_frames(frames, output_filename, fps=30.0):
    """Create a video from a list of frames."""
    if not frames:
        logging.error("No frames to create video.")
        return

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    for frame in tqdm(frames, desc="Writing video"):
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    video.release()
    logging.info(f"Video saved as {output_filename}")


def test_visualization_with_custom_dataset(
    dataset,
    start_index=0,
    end_index=None,
    output_filename="custom_dataset_visualization.mp4",
):
    """
    Test visualization functions with a custom dataset.
    """
    frames = generate_frames(dataset, start_index, end_index)
    create_video_from_frames(frames, output_filename)


if __name__ == "__main__":
    import torch

    from dataset import CustomLeRobotDataset

    # Create the custom dataset
    custom_dataset = CustomLeRobotDataset(
        "lerobot/pusht", prediction_horizon=24, observation_horizon=4
    )

    # Test visualization with custom dataset
    test_visualization_with_custom_dataset(custom_dataset, end_index=1000)
