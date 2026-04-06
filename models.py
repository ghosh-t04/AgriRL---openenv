# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Agriculture Environment.

The agriculture environment is a simple test environment that echoes back messages.
"""

# from openenv.core.env_server.types import Action, Observation
# from pydantic import Field


# class AgricultureAction(Action):
#     """Action for the Agriculture environment - just a message to echo."""

#     message: str = Field(..., description="Message to echo back")


# class AgricultureObservation(Observation):
#     """Observation from the Agriculture environment - the echoed message."""

#     echoed_message: str = Field(default="", description="The echoed message")
#     message_length: int = Field(default=0, description="Length of the echoed message")


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Agriculture Environment.

This environment simulates crop selection under soil, nutrient, climate,
and groundwater constraints.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class AgricultureAction(Action):
    """Action for the Agriculture environment - select a crop."""

    crop_name: str = Field(..., description="Crop selected by the agent")


class AgricultureObservation(Observation):
    """Observation from the Agriculture environment - current farm state."""

    soil_type: str = Field(default="", description="Type of soil")
    nitrogen: int = Field(default=0, description="Nitrogen level")
    phosphorus: int = Field(default=0, description="Phosphorus level")
    potassium: int = Field(default=0, description="Potassium level")
    ph: float = Field(default=7.0, description="Soil pH")
    rainfall: int = Field(default=0, description="Rainfall level")
    temperature: int = Field(default=0, description="Temperature")
    humidity: int = Field(default=0, description="Humidity percentage")
    groundwater: int = Field(default=0, description="Groundwater level")
    season: str = Field(default="", description="Current season")

    chosen_crop: str = Field(default="", description="Crop chosen in the last step")
    available_crops: list[str] = Field(default_factory=list, description="Available crop choices")