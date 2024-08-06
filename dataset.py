import os
import json
import base64
from typing import Any, Dict, List
import torch
from torch.utils.data import Dataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from tqdm import tqdm


class ShardedLeRobotDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        prediction_horizon: int = 16,
        observation_horizon: int = 2,
        shard_size: int = 1000,
        cache_dir: str = "./dataset_cache",
    ):
        self.base_dataset = LeRobotDataset(dataset_name)
        self.prediction_horizon = prediction_horizon
        self.observation_horizon = observation_horizon
        self.shard_size = shard_size
        self.cache_dir = cache_dir

        assert prediction_horizon > 0, "Prediction horizon must be greater than 0"
        assert observation_horizon > 0, "Observation horizon must be greater than 0"

        self.num_shards = (len(self.base_dataset) + shard_size - 1) // shard_size
        self.shards = [[] for _ in range(self.num_shards)]

        os.makedirs(self.cache_dir, exist_ok=True)
        self._create_or_load_shards()

    def _create_or_load_shards(self):
        print("Initializing dataset shards...")
        for shard_idx in tqdm(range(self.num_shards), desc="Processing shards"):
            shard_path = os.path.join(self.cache_dir, f"shard_{shard_idx}.json")
            if os.path.exists(shard_path):
                with open(shard_path, "r") as f:
                    self.shards[shard_idx] = json.load(f)
            else:
                shard_data = self._create_shard(shard_idx)
                self.shards[shard_idx] = shard_data
                with open(shard_path, "w") as f:
                    json.dump(shard_data, f)
        print("Shard initialization complete.")

    def _create_shard(self, shard_idx: int) -> List[Dict[str, Any]]:
        shard_data = []
        start_idx = shard_idx * self.shard_size
        end_idx = min((shard_idx + 1) * self.shard_size, len(self.base_dataset))

        for idx in tqdm(
            range(start_idx, end_idx), desc=f"Creating shard {shard_idx}", leave=False
        ):
            sample = self._process_sample(idx)
            shard_data.append(sample)

        return shard_data

    def _process_sample(self, idx: int) -> Dict[str, Any]:
        sample = self.base_dataset[idx]
        current_episode = sample["episode_index"]

        observation_states = []
        observation_actions = []
        observation_images = []
        prediction_actions = []

        # Collect observation data
        for i in range(self.observation_horizon):
            if (
                idx - i >= 0
                and self.base_dataset[idx - i]["episode_index"] == current_episode
            ):
                obs_sample = self.base_dataset[idx - i]
                observation_states.insert(0, obs_sample["observation.state"].tolist())
                observation_actions.insert(0, obs_sample["action"].tolist())
                observation_images.insert(
                    0, self._encode_image(obs_sample["observation.image"])
                )
            else:
                if observation_states:
                    observation_states.insert(0, observation_states[0])
                    observation_actions.insert(0, observation_actions[0])
                    observation_images.insert(0, observation_images[0])
                else:
                    observation_states.insert(0, sample["observation.state"].tolist())
                    observation_actions.insert(0, [0.0] * len(sample["action"]))
                    observation_images.insert(
                        0,
                        self._encode_image(
                            torch.zeros_like(sample["observation.image"])
                        ),
                    )

        # Collect prediction actions
        for i in range(self.prediction_horizon):
            if (
                idx + i < len(self.base_dataset)
                and self.base_dataset[idx + i]["episode_index"] == current_episode
            ):
                future_sample = self.base_dataset[idx + i]
                prediction_actions.append(future_sample["action"].tolist())
            else:
                if prediction_actions:
                    prediction_actions.append(prediction_actions[-1])
                else:
                    prediction_actions.append([0.0] * len(sample["action"]))

        return {
            "observation_states": observation_states,
            "observation_actions": observation_actions,
            "observation_images": observation_images,
            "prediction_actions": prediction_actions,
        }

    def _encode_image(self, image: torch.Tensor) -> str:
        image_np = (image.numpy() * 255).astype(np.uint8)
        return base64.b64encode(image_np.tobytes()).decode("utf-8")

    def _decode_image(self, image_str: str) -> torch.Tensor:
        image_bytes = base64.b64decode(image_str)
        image_np = np.frombuffer(image_bytes, dtype=np.uint8).reshape(3, 96, 96)
        return torch.from_numpy(image_np).float() / 255.0

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        shard_idx = idx // self.shard_size
        local_idx = idx % self.shard_size
        sample = self.shards[shard_idx][local_idx]

        return {
            "observation_states": torch.tensor(sample["observation_states"]),
            "observation_actions": torch.tensor(sample["observation_actions"]),
            "observation_images": torch.stack(
                [self._decode_image(img) for img in sample["observation_images"]]
            ),
            "prediction_actions": torch.tensor(sample["prediction_actions"]),
        }


# Example usage
if __name__ == "__main__":
    dataset = ShardedLeRobotDataset(
        "lerobot/pusht", prediction_horizon=16, observation_horizon=2, shard_size=1000
    )

    print("Sample keys:", dataset[0].keys())
    print("Observation states shape:", dataset[0]["observation_states"].shape)
    print("Observation actions shape:", dataset[0]["observation_actions"].shape)
    print("Observation images shape:", dataset[0]["observation_images"].shape)
    print("Prediction actions shape:", dataset[0]["prediction_actions"].shape)
