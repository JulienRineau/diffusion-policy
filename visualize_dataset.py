import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

from dataset import CustomLeRobotDataset


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


def draw_arrow(image, start, end, color, thickness=2, tip_length=0.2):
    """Draw an arrow on the image."""
    cv2.arrowedLine(image, start, end, color, thickness, tipLength=tip_length)


def create_video_from_dataset(
    dataset,
    start_index=0,
    end_index=None,
    output_filename="output_video.mp4",
    frame_size=(672, 224),  # Increased width to accommodate three plots
    fps=30.0,
):
    """
    Create a video from a dataset, including real-time plots of state and action evolution and visual cues.
    """
    if end_index is None:
        end_index = len(dataset)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_filename, fourcc, fps, frame_size)

    # Initialize state and action history
    state_history = {"x": [], "y": []}
    action_history = {"x": [], "y": []}

    # Define colors (in BGR for OpenCV)
    x_color = (255, 127, 14)  # Orange in BGR
    y_color = (31, 119, 180)  # Blue in BGR

    for i in tqdm(
        range(start_index, min(end_index, len(dataset))), desc="Creating video"
    ):
        # Get the image, state, and action from the dataset
        sample = dataset[i]
        image = sample["observation.image"]
        state = sample["observation.state"]
        action = sample["action"]

        # Check the type of image and convert accordingly
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).numpy()  # Assuming CHW format, change to HWC
        elif not isinstance(image, np.ndarray):
            print(f"Unexpected image type at index {i}: {type(image)}")
            continue

        # Normalize the image
        image = normalize_image(image)

        # Convert the image to BGR format (OpenCV uses BGR)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Resize the image to one-third of the frame width
        image_bgr = cv2.resize(image_bgr, (frame_size[0] // 3, frame_size[1]))

        # Draw arrows representing x and y axes
        origin = (20, 20)  # Top-left corner
        draw_arrow(
            image_bgr, origin, (origin[0] + 30, origin[1]), x_color, 2
        )  # x-axis (orange)
        draw_arrow(
            image_bgr, origin, (origin[0], origin[1] + 30), y_color, 2
        )  # y-axis (blue)

        # Add text labels for X and Y
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            image_bgr,
            "x",
            (origin[0] + 35, origin[1] + 5),
            font,
            0.7,
            x_color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image_bgr,
            "y",
            (origin[0] - 15, origin[1] + 35),
            font,
            0.7,
            y_color,
            2,
            cv2.LINE_AA,
        )

        # Update state and action history
        state_history["x"].append(state[0])
        state_history["y"].append(state[1])
        action_history["x"].append(action[0])
        action_history["y"].append(action[1])

        # Create plots of the state and action evolution
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

        # State plot
        ax1.plot(state_history["x"], color="#1f77b4", label="state x")
        ax1.plot(state_history["y"], color="#ff7f0e", label="state y")
        ax1.set_xlim(max(0, i - 100), i + 1)  # Show last 100 steps
        ax1.set_ylim(
            min(min(state_history["x"]), min(state_history["y"])) - 0.1,
            max(max(state_history["x"]), max(state_history["y"])) + 0.1,
        )
        ax1.legend()
        ax1.set_title("State Evolution")
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("State Value")

        # Action plot
        ax2.plot(action_history["x"], color="#2ca02c", label="action x")
        ax2.plot(action_history["y"], color="#d62728", label="action y")
        ax2.set_xlim(max(0, i - 100), i + 1)  # Show last 100 steps
        ax2.set_ylim(
            min(min(action_history["x"]), min(action_history["y"])) - 0.1,
            max(max(action_history["x"]), max(action_history["y"])) + 0.1,
        )
        ax2.legend()
        ax2.set_title("Action Evolution")
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Action Value")

        # Adjust layout and convert plot to an image
        plt.tight_layout()
        fig.canvas.draw()
        plot_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plot_image = cv2.cvtColor(plot_image, cv2.COLOR_RGB2BGR)
        plot_image = cv2.resize(plot_image, (frame_size[0] * 2 // 3, frame_size[1]))

        # Combine the observation image and the plots side by side
        combined_image = np.hstack((image_bgr, plot_image))

        # Write the frame
        out.write(combined_image)

        plt.close(fig)

    # Release the video writer
    out.release()
    print(f"Video saved as {output_filename}")


def scale_state(state, original_range=(0, 512), target_range=(0, 96)):
    """Scale the state values from the original range to the target range."""
    return (state - original_range[0]) / (original_range[1] - original_range[0]) * (
        target_range[1] - target_range[0]
    ) + target_range[0]


def visualize_prediction(dataset, idx, save_path):
    sample = dataset[idx]
    image = sample["observation_images"][-1]
    observation_states = sample["observation_states"]
    prediction_actions = sample["prediction_actions"]

    # Convert image to numpy array if it's a tensor
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()

    # Normalize image if necessary
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    # Scale and plot current state
    scaled_current_state = scale_state(observation_states[-1])
    plt.scatter(
        scaled_current_state[0],
        scaled_current_state[1],
        c="red",
        s=50,
        label="Current State",
    )

    # Scale and plot observed states in grey
    scaled_observation_states = scale_state(observation_states)
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
    scaled_prediction_actions = scale_state(prediction_actions)
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

    plt.title(f"State Prediction Overlay (Sample {idx})")
    plt.legend()
    plt.axis("off")  # Turn off axis
    plt.tight_layout()  # Adjust the plot to remove any extra white space
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # dataset = LeRobotDataset("lerobot/pusht")
    # create_video_from_dataset(dataset, end_index=1000)

    custom_dataset = CustomLeRobotDataset(
        "lerobot/pusht", prediction_horizon=24, observation_horizon=12
    )
    indices_to_visualize = [0, 100, 500, 1000]
    for idx in indices_to_visualize:
        visualize_prediction(custom_dataset, idx, f"prediction_overlay_{idx}.png")
        print(f"Saved visualization for sample {idx}")
