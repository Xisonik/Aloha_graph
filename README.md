Хорошо 🙌 Ниже — перевод твоего README на английский язык в аккуратном формате. Я сохранил структуру, блоки кода и заголовки.

---

# Video:

```
https://drive.google.com/file/d/1xD-vlL10g1_9S54mAHXR4sFoxZ3-DLWM/view?usp=drive_link
```

# Installation:

```
alias PYTHON_PATH=~/.local/share/ov/pkg/isaac_sim-*/python.sh
```

1. Download the folders `assets`, `model`, and `scene` from the link [https://disk.yandex.com/d/HlCCxiESonoKzQ](https://disk.yandex.com/d/HlCCxiESonoKzQ) into the repository.
2. Install the required modules:

   ```
   ./python.sh -m pip install ftfy regex tqdmip
   ./python.sh -m pip install git+https://github.com/openai/CLIP.git
   ./python.sh -m pip install ultralytics
   ```
3. Change the general project path in the variable `general_path` located in `configs/main_config.py`.

# Running the pipeline:

## Training:

1. Set the variable `eval` in `configs/main_config.py` to `False`.

```
PYTHON_PATH train.py
```

## Inference:

In the file `configs/main_config.py`:

1. Select the model to test in the variable `load_policy`;
2. Set `eval = True`;
3. Define the radius and angle for initial deviation: `eval_radius`, `eval_angle`.

```
PYTHON_PATH train.py
```

# Working with the pipeline:

In `configs/main_config.py`, define the path to the project directory in the variable `general_path`.

# Modify for working with the control module:

Edit the following files inside your Isaac Sim installation:

`/home/kit/.local/share/ov/pkg/isaac-sim-4.1.0/kit/python/lib/python3.10/site-packages/stable_baselines3/common/off_policy_algorithm.py`
line 559:

```python
# Rescale and perform action
new_obs, rewards, dones, infos, message_to_collback = env.step(actions)

if message_to_collback[0]:
    buffer_actions = message_to_collback[1]
```

`/home/kit/.local/share/ov/pkg/isaac-sim-4.1.0/kit/python/lib/python3.10/site-packages/stable_baselines3/common/vec_env/dummy_vec_env.py`
line 56:

```python
def step_wait(self) -> VecEnvStepReturn:
    # Avoid circular imports
    for env_idx in range(self.num_envs):
        obs, self.buf_rews[env_idx], terminated, truncated, self.buf_infos[env_idx], message_to_collback = self.envs[env_idx].step(  # type: ignore[assignment]
            self.actions[env_idx]
        )
        # convert to SB3 VecEnv api
        self.buf_dones[env_idx] = terminated or truncated
        # See https://github.com/openai/gym/issues/3102
        # Gym 0.26 introduces a breaking change
        self.buf_infos[env_idx]["TimeLimit.truncated"] = truncated and not terminated

        if self.buf_dones[env_idx]:
            # save final observation where user can get it, then reset
            self.buf_infos[env_idx]["terminal_observation"] = obs
            obs, self.reset_infos[env_idx] = self.envs[env_idx].reset()
        self._save_obs(env_idx, obs)
    return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos), np.copy(message_to_collback))
```

`/home/kit/.local/share/ov/pkg/isaac-sim-4.1.0/kit/python/lib/python3.10/site-packages/stable_baselines3/common/monitor.py`
line 85:

```python
def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
    """
    Step the environment with the given action

    :param action: the action
    :return: observation, reward, terminated, truncated, information
    """
    if self.needs_reset:
        raise RuntimeError("Tried to step environment that needs reset")
    observation, reward, terminated, truncated, info, message_to_collback = self.env.step(action)
    self.rewards.append(float(reward))
    if terminated or truncated:
        self.needs_reset = True
        ep_rew = sum(self.rewards)
        ep_len = len(self.rewards)
        ep_info = {"r": round(ep_rew, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6)}
        for key in self.info_keywords:
            ep_info[key] = info[key]
        self.episode_returns.append(ep_rew)
        self.episode_lengths.append(ep_len)
        self.episode_times.append(time.time() - self.t_start)
        ep_info.update(self.current_reset_info)
        if self.results_writer:
            self.results_writer.write_row(ep_info)
        info["episode"] = ep_info
    self.total_steps += 1
    return observation, reward, terminated, truncated, info, message_to_collback
```

`/home/kit/.local/share/ov/pkg/isaac-sim-4.1.0/kit/python/lib/python3.10/site-packages/gymnasium/wrappers/common.py`
line 112:

```python
def step(
    self, action: ActType
) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
    """Steps through the environment and if the number of steps elapsed exceeds ``max_episode_steps`` then truncate.

    Args:
        action: The environment step action

    Returns:
        The environment step ``(observation, reward, terminated, truncated, info)`` with `truncated=True`
        if the number of steps elapsed >= max episode steps

    """
    observation, reward, terminated, truncated, info, message_to_collback = self.env.step(action)
    self._elapsed_steps += 1

    if self._elapsed_steps >= self._max_episode_steps:
        truncated = True

    return observation, reward, terminated, truncated, info, message_to_collback
```

line 199:

```python
def step(
    self, action: ActType
) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
    """Steps through the environment with action and resets the environment if a terminated or truncated signal is encountered.

    Args:
        action: The action to take

    Returns:
        The autoreset environment :meth:`step`
    """
    if self.autoreset:
        obs, info = self.env.reset()
        reward, terminated, truncated = 0.0, False, False
    else:
        obs, reward, terminated, truncated, info, message_to_collback = self.env.step(action)

    self.autoreset = terminated or truncated
    return obs, reward, terminated, truncated, info, message_to_collback
```

`/home/kit/.local/share/ov/pkg/isaac-sim-4.1.0/kit/python/lib/python3.10/site-packages/gymnasium/core.py`
line 546:

```python
def step(
    self, action: ActType
) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
    """Modifies the :attr:`env` after calling :meth:`step` using :meth:`self.observation` on the returned observations."""
    observation, reward, terminated, truncated, info, message_to_collback = self.env.step(action)
    return self.observation(observation), reward, terminated, truncated, info, message_to_collback
```

`/home/kit/.local/share/ov/pkg/isaac-sim-4.1.0/kit/python/lib/python3.10/site-packages/gymnasium/utils/passive_env_checker.py`
insert at line 234:

```python
elif len(result) == 6:
    obs, reward, terminated, truncated, info, _ = result

    # np.bool is actual python bool not np boolean type, therefore bool_ or bool8
    if not isinstance(terminated, (bool, np.bool_)):
        logger.warn(
            f"Expects `terminated` signal to be a boolean, actual type: {type(terminated)}"
        )
    if not isinstance(truncated, (bool, np.bool_)):
        logger.warn(
            f"Expects `truncated` signal to be a boolean, actual type: {type(truncated)}"
        )
```

Assets link:

```
https://drive.google.com/drive/folders/1waC17p9AvrMe-7hTv8rpWKMKoUm4Mwyl?usp=drive_link
```

---

### General Pipeline Overview

The pipeline addresses the task of mobile robot (JetBot) navigation in an indoor environment with obstacles, where the goal is to reach a target position (e.g., a bowl). Reinforcement Learning (RL) is used, with an optional demonstration mode based on the Pure Pursuit controller for imitation learning.

Main stages:

1. **Environment Initialization**: setup of Isaac Sim world, robot, cameras, obstacles, and goals.
2. **Environment Reset**: randomizes robot and target positions, obstacles; optionally builds a demonstration path.
3. **Step Function**: applies actions, updates the environment, computes rewards, checks terminations.
4. **Observation Generation**: combines RGB/depth, kinematics, and scene graph embeddings.
5. **Scene Management**: handles obstacle placement and target assignment.
6. **Motion Control**: uses RL actions or Pure Pursuit demonstration trajectories.
7. **Rewards & Episode Termination**: shaped by distance, orientation, collisions, or timeout.
8. **Logging & Evaluation**: tracks Success Rate and saves trajectories.

---

### Detailed Component Description

#### 1. **CLGRCENV (env(1).py)** — main RL environment

*Handles simulation, robot control, observations, rewards, and logging.*

* Initialization: sets up Isaac Sim, robot, controller, cameras, action/observation spaces.
* Reset: re-generates obstacles, targets, robot start positions, updates scene.
* Step: executes actions, updates world, computes rewards, checks for collisions/timeouts.
* Observations: combines CLIP embeddings (images & text), robot velocities, and optional scene graph features.
* Rewards: distance-based, with penalties for time and collisions, success bonus for reaching goal.

#### 2. **Control\_module (control\_manager.py)** — motion controller

*Provides Pure Pursuit demonstration trajectories.*

* Builds paths using Dijkstra or A\*.
* Simplifies and saves paths.
* Pure Pursuit generates linear/angular velocities to follow paths.

#### 3. **Scene\_controller (scene\_manager.py)** — scene management

*Generates obstacles, target positions, robot start positions.*

* Creates and deletes scene objects.
* Provides collision checking.
* Returns valid robot and goal positions.

#### 4. **Graph\_manager** — scene graph handling

*Encodes scene graph into embeddings.*

* Extracts object attributes from JSON.
* Generates embeddings using SentenceTransformer and combines with geometric features.

#### 5. **LocalizationModule** — robot localization

(Not currently integrated.) Uses YOLO detections, depth, and graph alignment to estimate robot pose.

---

### Pipeline Workflow

1. Initialize environment, robot, scene, and controllers.
2. Reset: generate obstacles, assign goals, place robot.
3. Rollout loop: agent or controller produces action → environment executes → returns observation, reward, done.
4. Termination: success, collision, or timeout triggers reset.

---

### Key Method Inputs/Outputs

| Method                                   | Input                        | Output                                       |
| ---------------------------------------- | ---------------------------- | -------------------------------------------- |
| `CLGRCENV.__init__`                      | Config, simulation params    | Environment instance                         |
| `CLGRCENV.reset`                         | Seed, options                | Observations, info                           |
| `CLGRCENV.step`                          | Action `[forward, angular]`  | Obs, reward, done, truncated, info, callback |
| `Control_module.get_path`                | Positions, algorithm         | Path points                                  |
| `Control_module.pure_pursuit_controller` | Pose, params                 | Linear/angular velocities                    |
| `Scene_controller.generate_obstacles`    | Key, positions               | Obstacles                                    |
| `Scene_controller.get_target_position`   | Flag                         | Target, list, index                          |
| `Scene_controller.get_robot_position`    | Target coords, radius, angle | Pose, orientation, success flag              |

---

### Conclusion

The pipeline integrates simulation, RL, and demonstration control for robot navigation. `CLGRCENV` coordinates environment dynamics, `Scene_controller` manages scenes, `Control_module` provides demonstration control. Observations fuse visual, kinematic, and graph-based features, supporting both RL and imitation learning. The localization module is available for future integration.

---

Хочешь, я ещё сделаю тебе компактную **английскую версию README.md в GitHub-стиле** (без длинного теоретического описания, только установка + запуск + краткое объяснение)?
