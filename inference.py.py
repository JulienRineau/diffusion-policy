import cv2
import gym_pusht
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from visualize_dataset import scale_state


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


def overlay_prediction(image, current_state, past_states, future_actions, env):
    """Overlay prediction on the image."""
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    # Get the image dimensions for scaling
    img_height, img_width = image.shape[:2]

    # Scale current state
    scaled_current_state = scale_state(
        current_state, original_range=(0, 512), target_range=(0, img_width)
    )
    plt.scatter(
        scaled_current_state[0],
        scaled_current_state[1],
        c="red",
        s=50,
        label="Current State",
    )

    # Scale and plot past states in grey
    for i, state in enumerate(past_states):
        scaled_state = scale_state(
            state, original_range=(0, 512), target_range=(0, img_width)
        )
        plt.scatter(scaled_state[0], scaled_state[1], c="grey", s=30, alpha=0.7)
        if i > 0:
            scaled_prev_state = scale_state(
                past_states[i - 1], original_range=(0, 512), target_range=(0, img_width)
            )
            plt.plot(
                [scaled_prev_state[0], scaled_state[0]],
                [scaled_prev_state[1], scaled_state[1]],
                c="grey",
                linestyle=":",
                alpha=0.5,
            )

    # Scale and plot future actions with color gradient
    colors = plt.cm.rainbow(np.linspace(0, 1, len(future_actions)))
    for i, (action, color) in enumerate(zip(future_actions, colors)):
        scaled_action = scale_state(
            action, original_range=env.action_space.low, target_range=(0, img_width)
        )
        plt.scatter(scaled_action[0], scaled_action[1], c=[color], s=30, alpha=0.7)
        if i > 0:
            scaled_prev_action = scale_state(
                future_actions[i - 1],
                original_range=env.action_space.low,
                target_range=(0, img_width),
            )
            plt.plot(
                [scaled_prev_action[0], scaled_action[0]],
                [scaled_prev_action[1], scaled_action[1]],
                c=color,
                linestyle=":",
                alpha=0.5,
            )

    plt.title("State and Action Overlay")
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


def simple_action_function(observation, step_count, env):
    """
    Generate actions to move the agent in a slow diagonal pattern.
    """
    action_space_low = env.action_space.low
    action_space_high = env.action_space.high

    # Increase the period to slow down the movement
    period = 400
    progress = (step_count % period) / period

    # Create a diagonal movement pattern
    if progress < 0.25:
        # Move towards top-right
        target_x = action_space_low[0] + (
            action_space_high[0] - action_space_low[0]
        ) * (progress * 4)
        target_y = action_space_high[1]
    elif progress < 0.5:
        # Move towards bottom-left
        target_x = action_space_high[0] - (
            action_space_high[0] - action_space_low[0]
        ) * ((progress - 0.5) * 4)
        target_y = action_space_low[1]
    elif progress < 0.75:
        # Move towards top-right
        target_x = action_space_low[0] + (
            action_space_high[0] - action_space_low[0]
        ) * (progress * 4)
        target_y = action_space_high[1]

    else:
        # Move towards bottom-left
        target_x = action_space_high[0] - (
            action_space_high[0] - action_space_low[0]
        ) * ((progress - 0.5) * 4)
        target_y = action_space_low[1]

    return np.array([target_x, target_y])


def run_environment_with_visualization(
    episodes=1, steps_per_episode=800, output_filename="gym_pusht_visualization.mp4"
):
    env = gym.make(
        "gym_pusht/PushT-v0", render_mode="rgb_array", obs_type="pixels_agent_pos"
    )
    frames = []

    for episode in tqdm(range(episodes), desc="Episodes"):
        observation, info = env.reset()
        past_states = []

        for step in tqdm(
            range(steps_per_episode), desc=f"Steps (Episode {episode+1})", leave=False
        ):
            action = simple_action_function(observation, step, env)
            observation, reward, terminated, truncated, info = env.step(action)

            image = observation["pixels"]
            image = normalize_image(image)

            past_states.append(observation["agent_pos"])
            if len(past_states) > 20:  # Increased to show more of the past trajectory
                past_states.pop(0)

            future_actions = [
                simple_action_function(observation, step + i, env) for i in range(10)
            ]  # Increased to show more future actions

            overlay_image = overlay_prediction(
                image, observation["agent_pos"], past_states, future_actions, env
            )
            frames.append(overlay_image)

            if terminated or truncated:
                break

    env.close()

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_filename, fourcc, 30.0, (width, height))

    for frame in tqdm(frames, desc="Creating video"):
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    video.release()
    print(f"Video saved as {output_filename}")


if __name__ == "__main__":
    run_environment_with_visualization()
