# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Agriculture Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import AgricultureAction, AgricultureObservation


class AgricultureEnv(
    EnvClient[AgricultureAction, AgricultureObservation, State]
):
    """
    Client for the Agriculture Environment.
    """

    def _step_payload(self, action: AgricultureAction) -> Dict:
        """
        Convert AgricultureAction to JSON payload for step message.
        """
        return {
            "crop_name": action.crop_name,
        }

    def _parse_result(self, payload: Dict) -> StepResult[AgricultureObservation]:
        """
        Parse server response into StepResult[AgricultureObservation].
        """
        obs_data = payload.get("observation", {})

        observation = AgricultureObservation(
            soil_type=obs_data.get("soil_type", ""),
            nitrogen=obs_data.get("nitrogen", 0),
            phosphorus=obs_data.get("phosphorus", 0),
            potassium=obs_data.get("potassium", 0),
            ph=obs_data.get("ph", 7.0),
            rainfall=obs_data.get("rainfall", 0),
            temperature=obs_data.get("temperature", 0),
            humidity=obs_data.get("humidity", 0),
            groundwater=obs_data.get("groundwater", 0),
            season=obs_data.get("season", ""),
            chosen_crop=obs_data.get("chosen_crop", ""),
            available_crops=obs_data.get("available_crops", []),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )