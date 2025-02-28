# Aloha_graph
 
изменить:
/home/kit/.local/share/ov/pkg/isaac-sim-4.1.0/kit/python/lib/python3.10/site-packages/stable_baselines3/common/off_policy_algorithm.py

line 559
```
# Rescale and perform action
            new_obs, rewards, dones, infos, message_to_collback = env.step(actions)
            
            if message_to_collback[0]:
                buffer_actions = message_to_collback[1]
```

/home/kit/.local/share/ov/pkg/isaac-sim-4.1.0/kit/python/lib/python3.10/site-packages/stable_baselines3/common/vec_env/dummy_vec_env.py
line 56
```
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

/home/kit/.local/share/ov/pkg/isaac-sim-4.1.0/kit/python/lib/python3.10/site-packages/stable_baselines3/common/monitor.py
line 85
```
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

/home/kit/.local/share/ov/pkg/isaac-sim-4.1.0/kit/python/lib/python3.10/site-packages/gymnasium/wrappers/common.py
line 112
```
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

line 199
```
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
/home/kit/.local/share/ov/pkg/isaac-sim-4.1.0/kit/python/lib/python3.10/site-packages/gymnasium/core.py
line 546
```
    def step(
        self, action: ActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Modifies the :attr:`env` after calling :meth:`step` using :meth:`self.observation` on the returned observations."""
        observation, reward, terminated, truncated, info, message_to_collback = self.env.step(action)
        return self.observation(observation), reward, terminated, truncated, info, message_to_collback
```
/home/kit/.local/share/ov/pkg/isaac-sim-4.1.0/kit/python/lib/python3.10/site-packages/gymnasium/utils/passive_env_checker.py
insert in line 234 elif 
```
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
assets link:
```
https://drive.google.com/drive/folders/1waC17p9AvrMe-7hTv8rpWKMKoUm4Mwyl?usp=drive_link
```
