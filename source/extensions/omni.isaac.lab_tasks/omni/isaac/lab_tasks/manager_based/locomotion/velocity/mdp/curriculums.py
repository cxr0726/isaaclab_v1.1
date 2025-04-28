# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`omni.isaac.lab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.terrains import TerrainImporter

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv
from collections import deque
import statistics
def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`omni.isaac.lab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
   # print(move_up,distance,"moooo",terrain.cfg.terrain_generator.size)
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5#*0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


# lenbuffer = deque(maxlen=100)
# reward_scale=0.2
# def reward_penalty_curriculum(
#     env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     asset: Articulation = env.scene[asset_cfg.name]
#     lenbuffer.extend(env.episode_length_buf[env_ids].cpu().numpy().tolist())
#     mean_episode_length=statistics.mean(lenbuffer)
#     if mean_episode_length>env.max_episode_length:
#         reward_scale*=(1+0.002)
#     env.reward_manager._term_cfgs[env.reward_manager._term_names.index("action_rate_l2")].weight=
#     print(mean_episode_length,"eee",lenbuffer)
#     return mean_episode_length
from omni.isaac.lab.managers.manager_base import ManagerTermBase
from omni.isaac.lab.managers.manager_term_cfg import RewardTermCfg
import numpy as np
class reward_penalty_curriculum(ManagerTermBase):
    """
    A reward term for penalizing large instantaneous changes in the network action output.
    This penalty encourages smoother actions over time.
    """

    def __init__(self,cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward term.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)

        self.lenbuffer = deque(maxlen=100)
        self.action_rate_reward_scale = 0.2#+0.1
        self.root_acc_reward_scale=0.
        self.v_limit_reward_scale=0.
        self.v_limit_reward_scale_copy=0.2


        self.direction_reward_scale=1.
        self.origin_scale=env.reward_manager._term_cfgs[env.reward_manager._term_names.index("action_rate_l2")].weight
        self.root_acc_origin_scale=env.reward_manager._term_cfgs[env.reward_manager._term_names.index("root_acc_l2")].weight
        self.v_limit_origin_scale=env.reward_manager._term_cfgs[env.reward_manager._term_names.index("exceed_v_limit")].weight
        self.direction_origin_scale=env.reward_manager._term_cfgs[env.reward_manager._term_names.index("correct_direction")].weight
        self.__name__ = "reward_penalty_curriculum"

    def __call__(self, env: ManagerBasedRLEnv,env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        self.lenbuffer.extend(env.episode_length_buf[env_ids].cpu().numpy().tolist())
        mean_episode_length = statistics.mean(self.lenbuffer)
        if mean_episode_length > env.max_episode_length*0.9:
            if self.action_rate_reward_scale <1.0:
                self.action_rate_reward_scale *= (1 + 0.00005)
                self.v_limit_reward_scale_copy *= (1 + 0.00005)
            self.root_acc_reward_scale=1.0
            self.v_limit_reward_scale=self.v_limit_reward_scale_copy#1.0

            if self.direction_reward_scale>5e-2:#1e-3:
                self.direction_reward_scale *= (1 - 0.0001*0.5)#0.5##this changed *0.6
            else:
                self.direction_reward_scale=0.

        #print(self.lenbuffer,mean_episode_length,env.max_episode_length,"eeepppp")

        env.reward_manager._term_cfgs[env.reward_manager._term_names.index("action_rate_l2")].weight =np.clip(self.origin_scale*self.action_rate_reward_scale,self.origin_scale,0)
        env.reward_manager._term_cfgs[env.reward_manager._term_names.index("root_acc_l2")].weight = self.root_acc_origin_scale* self.root_acc_reward_scale
        env.reward_manager._term_cfgs[env.reward_manager._term_names.index("exceed_v_limit")].weight=self.v_limit_origin_scale*self.v_limit_reward_scale
        env.reward_manager._term_cfgs[env.reward_manager._term_names.index("correct_direction")].weight=self.direction_origin_scale*self.direction_reward_scale
        #print(env.max_episode_length,mean_episode_length, "eee", self.lenbuffer)
        #print(env.reward_manager._term_cfgs[env.reward_manager._term_names.index("exceed_v_limit")].weight)
        return env.reward_manager._term_cfgs[env.reward_manager._term_names.index("exceed_v_limit")].weight