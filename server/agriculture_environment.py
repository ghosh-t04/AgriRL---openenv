# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Agriculture Environment Implementation.

An agriculture decision environment where the agent selects crops based on
soil, nutrient, climate, and groundwater conditions to maximize reward.
"""

import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import AgricultureAction, AgricultureObservation
except ImportError:
    from models import AgricultureAction, AgricultureObservation


class AgricultureEnvironment(Environment):
    """
    Agriculture crop-selection environment.

    The agent observes farm conditions and selects a crop.
    Reward is based on suitability, expected yield, and sustainability.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the agriculture environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0

        self.crops = [
            "rice",
            "wheat",
            "maize",
            "cotton",
            "groundnut",
            "pulses",
            "millet",
            "sugarcane"
        ]

        self.soil_types = ["sandy", "loamy", "clayey", "black", "red"]
        self.seasons = ["kharif", "rabi", "summer"]

        self.crop_rules = {
            "rice": {
                "preferred_soils": ["loamy", "clayey"],
                "n_range": [70, 120],
                "p_range": [30, 60],
                "k_range": [30, 60],
                "ph_range": [5.5, 7.5],
                "rainfall_range": [150, 300],
                "temp_range": [20, 35],
                "humidity_range": [60, 95],
                "groundwater_need": 70,
                "yield_base": 90
            },
            "wheat": {
                "preferred_soils": ["loamy", "black"],
                "n_range": [50, 100],
                "p_range": [20, 50],
                "k_range": [20, 50],
                "ph_range": [6.0, 7.5],
                "rainfall_range": [50, 120],
                "temp_range": [15, 25],
                "humidity_range": [40, 70],
                "groundwater_need": 50,
                "yield_base": 80
            },
            "maize": {
                "preferred_soils": ["loamy", "red"],
                "n_range": [50, 100],
                "p_range": [20, 50],
                "k_range": [20, 50],
                "ph_range": [5.5, 7.5],
                "rainfall_range": [60, 150],
                "temp_range": [18, 32],
                "humidity_range": [40, 80],
                "groundwater_need": 45,
                "yield_base": 78
            },
            "cotton": {
                "preferred_soils": ["black", "red"],
                "n_range": [40, 90],
                "p_range": [20, 50],
                "k_range": [30, 60],
                "ph_range": [5.8, 8.0],
                "rainfall_range": [50, 120],
                "temp_range": [21, 35],
                "humidity_range": [30, 70],
                "groundwater_need": 55,
                "yield_base": 75
            },
            "groundnut": {
                "preferred_soils": ["sandy", "red"],
                "n_range": [30, 70],
                "p_range": [20, 50],
                "k_range": [20, 50],
                "ph_range": [6.0, 7.5],
                "rainfall_range": [50, 120],
                "temp_range": [20, 30],
                "humidity_range": [30, 70],
                "groundwater_need": 35,
                "yield_base": 72
            },
            "pulses": {
                "preferred_soils": ["loamy", "red"],
                "n_range": [20, 60],
                "p_range": [20, 50],
                "k_range": [20, 50],
                "ph_range": [6.0, 8.0],
                "rainfall_range": [40, 100],
                "temp_range": [18, 32],
                "humidity_range": [30, 70],
                "groundwater_need": 25,
                "yield_base": 70
            },
            "millet": {
                "preferred_soils": ["sandy", "black"],
                "n_range": [20, 60],
                "p_range": [15, 40],
                "k_range": [15, 40],
                "ph_range": [5.5, 8.0],
                "rainfall_range": [30, 100],
                "temp_range": [22, 38],
                "humidity_range": [20, 60],
                "groundwater_need": 20,
                "yield_base": 68
            },
            "sugarcane": {
                "preferred_soils": ["loamy", "clayey", "black"],
                "n_range": [80, 130],
                "p_range": [30, 60],
                "k_range": [40, 80],
                "ph_range": [6.0, 7.8],
                "rainfall_range": [100, 250],
                "temp_range": [22, 35],
                "humidity_range": [50, 90],
                "groundwater_need": 85,
                "yield_base": 95
            }
        }

        self.max_steps = 3
        self.current_farm_state = {}

    def reset(self) -> AgricultureObservation:
        """Reset the environment and generate a fresh farm state."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1

        self.current_farm_state = {
            "soil_type": random.choice(self.soil_types),
            "nitrogen": random.randint(20, 140),
            "phosphorus": random.randint(5, 145),
            "potassium": random.randint(5, 205),
            "ph": round(random.uniform(4.5, 8.5), 1),
            "rainfall": random.randint(30, 250),
            "temperature": random.randint(15, 40),
            "humidity": random.randint(20, 95),
            "groundwater": random.randint(10, 100),
            "season": random.choice(self.seasons),
        }

        return AgricultureObservation(
            **self.current_farm_state,
            chosen_crop="",
            available_crops=self.crops,
            done=False,
            reward=0.0,
        )

    def step(self, action: AgricultureAction) -> AgricultureObservation:  # type: ignore[override]
        """Execute one crop-selection step."""
        self._state.step_count += 1

        crop = action.crop_name.strip().lower()

        if crop not in self.crops:
            return AgricultureObservation(
                **self.current_farm_state,
                chosen_crop=crop,
                available_crops=self.crops,
                done=False,
                reward=-10.0,
                metadata={
                    "error": f"Invalid crop '{crop}'. Choose from {self.crops}",
                    "step": self._state.step_count,
                },
            )

        reward = self._calculate_reward(crop)
        self._apply_crop_effects(crop)

        done = self._state.step_count >= self.max_steps

        return AgricultureObservation(
            **self.current_farm_state,
            chosen_crop=crop,
            available_crops=self.crops,
            done=done,
            reward=reward,
            metadata={
                "step": self._state.step_count,
                "episode_id": self._state.episode_id,
            },
        )

    def _calculate_reward(self, crop: str) -> float:
        rules = self.crop_rules[crop]
        s = self.current_farm_state
        score = 0.0

        if s["soil_type"] in rules["preferred_soils"]:
            score += 10
        else:
            score -= 10

        if rules["n_range"][0] <= s["nitrogen"] <= rules["n_range"][1]:
            score += 5
        if rules["p_range"][0] <= s["phosphorus"] <= rules["p_range"][1]:
            score += 5
        if rules["k_range"][0] <= s["potassium"] <= rules["k_range"][1]:
            score += 5

        if rules["ph_range"][0] <= s["ph"] <= rules["ph_range"][1]:
            score += 5
        else:
            score -= 5

        if rules["rainfall_range"][0] <= s["rainfall"] <= rules["rainfall_range"][1]:
            score += 5
        if rules["temp_range"][0] <= s["temperature"] <= rules["temp_range"][1]:
            score += 5
        if rules["humidity_range"][0] <= s["humidity"] <= rules["humidity_range"][1]:
            score += 5

        if s["groundwater"] < rules["groundwater_need"]:
            score -= 15
        else:
            score += 5

        score += rules["yield_base"] / 10.0
        return round(score, 2)

    def _apply_crop_effects(self, crop: str):
        rules = self.crop_rules[crop]

        self.current_farm_state["nitrogen"] = max(
            0, self.current_farm_state["nitrogen"] - random.randint(5, 15)
        )
        self.current_farm_state["phosphorus"] = max(
            0, self.current_farm_state["phosphorus"] - random.randint(3, 10)
        )
        self.current_farm_state["potassium"] = max(
            0, self.current_farm_state["potassium"] - random.randint(3, 10)
        )
        self.current_farm_state["groundwater"] = max(
            0, self.current_farm_state["groundwater"] - int(rules["groundwater_need"] / 10)
        )

        if crop == "pulses":
            self.current_farm_state["nitrogen"] = min(
                140, self.current_farm_state["nitrogen"] + 10
            )

        current_idx = self.seasons.index(self.current_farm_state["season"])
        self.current_farm_state["season"] = self.seasons[(current_idx + 1) % len(self.seasons)]

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state