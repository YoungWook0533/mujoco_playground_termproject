# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Open a door."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
import mujoco  # pylint: disable=unused-import
from mujoco.mjx._src import math

from mujoco_playground._src import collision
from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.husky_fr3 import base
from mujoco_playground._src.mjx_env import State  # pylint: disable=g-importing-member
from mujoco_playground._src.manipulation.husky_fr3 import husky_kinematics


def default_config() -> config_dict.ConfigDict:
  """Returns the default configuration for the environment."""
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.005,
      episode_length=300,
      action_repeat=1,
      action_scale=0.02,
      mobile_action_scale=0.4,
      reward_config=config_dict.create(
          scales=config_dict.create(
              # Gripper goes to the box.
              gripper_box=5.0,
              # Door hinge angle to 2.0[rad].
              door_open=10.0,
          # Do not collide the barrier.
          no_barrier_collision=0.25,
          # Do not self-collide.
          no_self_collision=0.1,
          # Arm stays close to target pose.
          robot_target_qpos=0.1,
        #   # Keep joints away from straight/limit configurations.
        #   joint_margin=1.0,
          # Encourage a slight elbow bend to avoid singularity.
          elbow_bend=0.5,
          # Avoid over-extended hand relative to base (singularity surrogate).
          manipulability=0.5,
          # Penalize the end-effector from overextending relative to the base.
        #   hand_base_reach=1.0,
      )
  ),
  )


class HuskyFR3OpenDoor(base.HuskyFR3Base):
  """Environment for training the Franka husky_fr3 robot to bring an object to a
  target."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
      sample_orientation: bool = False,  # pylint: disable=unused-argument
  ):
    xml_path = (
        mjx_env.ROOT_PATH
        / "manipulation"
        / "husky_fr3"
        / "xmls"
        / "mjx_door.xml"
    )
    super().__init__(xml_path, config, config_overrides)

    # Enable hand base collision to shape learning
    self.mj_model.geom("hand_capsule").conaffinity = 3
    self.mj_model.geom("wrist_capsule").conaffinity = 3
    self.mj_model.geom("link5_capsule").conaffinity = 3
    self.mj_model.geom("base_box_lower").conaffinity = 3
    self.mj_model.geom("base_box_upper").conaffinity = 3
    self._mjx_model = mjx.put_model(self.mj_model)

    self._post_init(obj_name="handle", keyframe="home")
    self._barrier_geom = self._mj_model.geom("panel").id

    handle_bid = self.mj_model.body("handle").id
    jnt_adr    = self.mj_model.body_jntadr[handle_bid]
    door_jid   = int(jnt_adr)
    self._door_qposadr = int(self.mj_model.jnt_qposadr[door_jid])
    self._door_home = float(self._init_q[self._door_qposadr])

    self._wheel_act_ids = jp.array([0, 1])
    self._arm_act_ids   = jp.arange(2, 9)

  def reset(self, rng: jax.Array) -> State:
    """Resets the environment to an initial state."""
    rng, rng_target = jax.random.split(rng)

    rng, rng_arm, rng_base = jax.random.split(rng, 3)

    # Set initial Arm qpos
    eps = jp.deg2rad(30)
    perturb_mins = jp.array([-eps, -eps, -eps, -2 * eps, -eps, 0, -eps])
    perturb_maxs = jp.array([eps, eps, eps, 0, eps, 2 * eps, eps])

    perturb_arm = jax.random.uniform(
        rng_arm, (7,), minval=perturb_mins, maxval=perturb_maxs
    )

    init_q = jp.array(self._init_q).at[self._robot_arm_qposadr].set(self._init_q[self._robot_arm_qposadr] + perturb_arm)
    init_ctrl = (
        jp.array(self._init_ctrl).at[self._arm_act_ids].set(self._init_ctrl[self._arm_act_ids] + perturb_arm)
    )

    # Set initial base qpos
    base_xy_jitter = jax.random.uniform(rng_base, (2,), minval=-0.10, maxval=0.10)
    init_q = jp.array(self._init_q).at[:2].set(self._init_q[:2] + base_xy_jitter)

    data: mjx.Data = mjx_env.init(
        self._mjx_model, init_q, jp.zeros(self._mjx_model.nv), ctrl=init_ctrl
    )

    info = {
        "rng": rng,
        "reached_box": 0.0,
        "target_mat": data.xmat[self._obj_body],  # For visualisation only
        "previously_gripped": 0.0,
    }
    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)
    metrics = {
        "out_of_bounds": jp.array(0.0),
        **{k: jp.array(0.0) for k in self._config.reward_config.scales.keys()},
    }
    state = State(data, obs, reward, done, metrics, info)
    return state

  def step(self, state: State, action: jax.Array) -> State:
    """Advances the environment by one timestep."""
    twist = action[self._wheel_act_ids] * self._config.mobile_action_scale
    # twist = action[self._wheel_act_ids]
    wheel_vel_curr  = state.data.qvel[self._robot_wheel_dofadr]
    wheel_vel = husky_kinematics.IK(twist, wheel_vel_curr, self._config.sim_dt)  # returns [w_left, w_right]

    # --- 2.  arm increment ---------------------------------------------
    arm_delta = action[self._arm_act_ids] * self._config.action_scale

    ctrl = state.data.ctrl
    # wheels are velocity actuators (indices 0,1)
    ctrl = ctrl.at[0].set(jp.clip(wheel_vel[0], self._lowers[0], self._uppers[0]))
    ctrl = ctrl.at[1].set(jp.clip(wheel_vel[1], self._lowers[1], self._uppers[1]))

    new_arm = ctrl[self._arm_act_ids] + arm_delta
    new_arm = jp.clip(
        new_arm,
        self._lowers[self._arm_act_ids],
        self._uppers[self._arm_act_ids],
    )
    ctrl = ctrl.at[self._arm_act_ids].set(new_arm)

    data: mjx.Data = mjx_env.step(
        self._mjx_model, state.data, ctrl, self.n_substeps
    )

    raw_rewards = self._get_rewards(data, state.info)
    rewards = {
        k: v * self._config.reward_config.scales[k]
        for k, v in raw_rewards.items()
    }
    reward = jp.clip(sum(rewards.values()), -1e4, 1e4)

    box_pos = data.xpos[self._obj_body]
    out_of_bounds = jp.any(jp.abs(box_pos) > 2.5)
    out_of_bounds |= box_pos[2] < 0
    done = out_of_bounds | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    done = done.astype(float)

    state.metrics.update(
        **raw_rewards, out_of_bounds=out_of_bounds.astype(float)
    )

    obs = self._get_obs(data, state.info)
    state = State(data, obs, reward, done, state.metrics, state.info)

    return state

  def _get_rewards(self, data: mjx.Data, info: dict):
    # Compute reward terms
    box_pos = data.xpos[self._obj_body]
    gripper_pos = data.site_xpos[self._gripper_site]
    fr3_base_pos = data.site_xpos[self._fr3_base_site]
    q_arm = data.qpos[self._robot_arm_qposadr]

    # # Encourage staying away from joint limits / straight-line singularities.
    # lower = jp.array(self._lowers[self._arm_act_ids])
    # upper = jp.array(self._uppers[self._arm_act_ids])
    # margin = jp.minimum(q_arm - lower, upper - q_arm)
    # # Reward increases when the tightest margin is comfortably above 0.
    # joint_margin = jp.tanh(jp.clip(jp.min(margin) - 0.12, -0.2, 0.2) * 15.0)

    # Keep the elbow mildly bent (joint3 for FR3).
    elbow_target = self._init_q[self._robot_arm_qposadr[2]]
    elbow_bend = 1 - jp.tanh(jp.abs(q_arm[2] - elbow_target) * 4.0)

    # Check for collisions with the barrier
    hand_barrier_collision = [
        collision.geoms_colliding(data, self._barrier_geom, g)
        for g in [
            self._left_finger_geom,
            self._right_finger_geom,
            self._link5_geom,
            self._wrist_geom,
            self._hand_geom,
            self._base_lower_geom,
        ]
    ]
    barrier_collision = sum(hand_barrier_collision) > 0
    no_barrier_collision = 1 - barrier_collision

    hand_base_collision = [
        collision.geoms_colliding(data, self._base_upper_geom, g)
        for g in [
            self._left_finger_geom,
            self._right_finger_geom,
            self._link5_geom,
            self._wrist_geom,
            self._hand_geom,
        ]
    ]
    self_collision = sum(hand_base_collision) > 0
    no_self_collision = 1 - self_collision

    info["reached_box"] = 1.0 * jp.maximum(
        info["reached_box"],
        (jp.linalg.norm(box_pos - gripper_pos) < 1.0 * 0.012),
    )

    hinge = data.qpos[self._door_qposadr] - self._door_home
    door_open = jp.clip(hinge / 2.0, 0.0, 1.0)
    gripper_box = 1 - jp.tanh(5 * jp.linalg.norm(box_pos - gripper_pos))
    # Encourage grasping early; once the door is open enough, relax the proximity.
    # Linearly decay between 0.3–0.7 open, then drop to zero emphasis.
    gripper_decay = 1.0 - jp.clip((door_open - 0.3) / 0.4, 0.0, 1.0)
    gripper_box *= gripper_decay

    # Simple manipulability surrogate: avoid over-extending hand relative to base.
    hand_base_dist = jp.linalg.norm(gripper_pos[:2] - fr3_base_pos[:2])
    # hand_base_clipped = jp.clip(hand_base_dist, 0.3, 0.8)
    # manipulability = 1.0 - jp.tanh((hand_base_clipped - 0.5) * 4.0)
    under = jp.maximum(0.3 - hand_base_dist, 0.0)
    over = jp.maximum(hand_base_dist - 0.8, 0.0)
    manipulability = 1.0 - jp.tanh((under + over) * 8.0)

    robot_target_qpos = 1 - jp.tanh(
        jp.linalg.norm(
            data.qpos[self._robot_arm_qposadr]
            - self._init_q[self._robot_arm_qposadr]
        )
    )
    # hand_base_dist = jp.linalg.norm(gripper_pos[:2] - fr3_base_pos[:2])
    # dist_clipped = jp.clip(hand_base_dist, 0.2, 0.5)
    # hand_base_reach = jp.maximum(0.0, (0.7 - dist_clipped)/0.5)
    # # hand_base_reach = -jp.maximum(0.0, (hand_base_dist - 0.7) / 0.3)

    return {
        "door_open": door_open * info["reached_box"],
        "gripper_box": gripper_box,
        "no_barrier_collision": no_barrier_collision.astype(float),
        "no_self_collision": no_self_collision.astype(float),
        "robot_target_qpos": robot_target_qpos,
        # "joint_margin": joint_margin,
        "elbow_bend": elbow_bend,
        "manipulability": manipulability,
        # "hand_base_reach": hand_base_reach,
    }

  def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    gripper_pos = data.site_xpos[self._gripper_site]
    gripper_mat = data.site_xmat[self._gripper_site].ravel()
    target_mat = math.quat_to_mat(data.mocap_quat[self._mocap_target])
    obs = jp.concatenate([
        data.qpos,
        data.qvel,
        gripper_pos,
        gripper_mat[3:],
        data.xmat[self._obj_body].ravel()[3:],
        data.xpos[self._obj_body] - data.site_xpos[self._gripper_site],
        target_mat.ravel()[:6] - data.xmat[self._obj_body].ravel()[:6],
        data.ctrl - data.qpos[self._robot_qposadr[:-1]],
    ])

    return obs
