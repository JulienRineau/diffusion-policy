from typing import Any, Dict

import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import Dataset


class CustomLeRobotDataset(Dataset):
    """
    Custom dataset for LeRobot tasks with observation and prediction horizons.

    Collects past observations and future actions based on specified horizons.
    """

    def __init__(
        self,
        dataset_name: str,
        prediction_horizon: int = 16,
        observation_horizon: int = 2,
    ):
        self.base_dataset = LeRobotDataset(dataset_name)
        self.prediction_horizon = prediction_horizon
        self.observation_horizon = observation_horizon

        assert prediction_horizon > 0, "Prediction horizon must be greater than 0"
        assert observation_horizon > 0, "Observation horizon must be greater than 0"

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.base_dataset[idx]
        current_episode = sample["episode_index"]

        # Initialize observation data
        observation_states = []
        observation_actions = []
        observation_images = []

        # Collect observation data by looking backwards
        for i in range(self.observation_horizon):
            if (
                idx - i >= 0
                and self.base_dataset[idx - i]["episode_index"] == current_episode
            ):
                observation_sample = self.base_dataset[idx - i]
                observation_states.insert(0, observation_sample["observation.state"])
                observation_actions.insert(0, observation_sample["action"])
                observation_images.insert(0, observation_sample["observation.image"])
            else:
                # If we've reached the beginning of the episode or dataset, repeat the first valid data
                if observation_states:
                    observation_states.insert(0, observation_states[0])
                    observation_actions.insert(0, observation_actions[0])
                    observation_images.insert(0, observation_images[0])
                else:
                    # If no valid observation was found, use the current state, a zero action, and a blank image
                    observation_states.insert(0, sample["observation.state"])
                    observation_actions.insert(0, torch.zeros_like(sample["action"]))
                    observation_images.insert(
                        0, torch.zeros_like(sample["observation.image"])
                    )

        # Initialize prediction actions
        prediction_actions = []

        # Collect prediction actions by looking forward
        for i in range(self.prediction_horizon):
            if (
                idx + i < len(self.base_dataset)
                and self.base_dataset[idx + i]["episode_index"] == current_episode
            ):
                future_sample = self.base_dataset[idx + i]
                prediction_actions.append(future_sample["action"])
            else:
                # If we've reached the end of the episode or dataset, repeat the last valid action
                if prediction_actions:
                    prediction_actions.append(prediction_actions[-1])
                else:
                    # If no valid future action was found, use a zero action
                    prediction_actions.append(torch.zeros_like(sample["action"]))

        # Convert lists to tensors
        sample["observation_states"] = torch.stack(observation_states)
        sample["observation_actions"] = torch.stack(observation_actions)
        sample["observation_images"] = torch.stack(observation_images)
        sample["prediction_actions"] = torch.stack(prediction_actions)

        return sample


# Example usage
if __name__ == "__main__":
    dataset = CustomLeRobotDataset(
        "lerobot/pusht", prediction_horizon=16, observation_horizon=2
    )

    print("Sample keys:", dataset[0].keys())
    print("Observation states shape:", dataset[0]["observation_states"].shape)
    print("Observation actions shape:", dataset[0]["observation_actions"].shape)
    print("Observation images shape:", dataset[0]["observation_images"].shape)
    print("Prediction actions shape:", dataset[0]["prediction_actions"].shape)
