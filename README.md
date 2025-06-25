# Видео:
```
https://drive.google.com/file/d/1xD-vlL10g1_9S54mAHXR4sFoxZ3-DLWM/view?usp=drive_link
```
# Установка:
```
alias PYTHON_PATH=~/.local/share/ov/pkg/isaac_sim-*/python.sh
```
1. Скачать папки assets, model и scene по ссылке https://disk.yandex.com/d/HlCCxiESonoKzQ в репозиторий.
2. Установить необходимые модули:
   ```
   ./python.sh -m pip install ftfy regex tqdmip
   ./python.sh -m pip install git+https://github.com/openai/CLIP.git
   ./python.sh -m pip install ultralytics
   ```
3. Изменить общий путь до проекта в переменной general_path расположенной в файле configs/main_config.py
# Запуск пайплайна:
## обучение:
1. В переменной расположенной eval в файле configs/main_config.py установить значение False
```
PYTHON_PATH train.py
```
## инференс:
В в файле configs/main_config.py
1. выбрать в переменной load_policy модель, которую необходимо протестировать;
2. eval = True
3. задать радиус и угол начального отклонения eval_radius, eval_angle
```
PYTHON_PATH train.py
```
# Работа с пайплайном:
в файле configs/main_cnfig.py в переменной general_path оределить путь до директории проекта
 
# изменить для работы с модулем контроля:
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

---

### Общая схема работы пайплайна

Пайплайн решает задачу навигации мобильного робота (JetBot) в помещении с препятствиями, где цель — достичь заданной позиции (например, чаши, обозначенной как "bowl"). Используется обучение с подкреплением (Reinforcement Learning, RL), но также есть возможность демонстрационного режима с использованием Pure Pursuit контроллера для имитационного обучения. Основные этапы работы:

1. **Инициализация среды**:
   - Создается симуляционная среда с использованием Isaac Sim.
   - Инициализируются сцена, робот, камеры, препятствия и целевая позиция.
   - Настраиваются параметры RL: пространство действий, пространство наблюдений, награды.

2. **Сброс среды (reset)**:
   - Устанавливается новая начальная позиция робота и целевая позиция.
   - Генерируются препятствия с использованием случайного ключа (key).
   - Если включен демонстрационный режим, обновляется путь для Pure Pursuit контроллера.

3. **Шаг среды (step)**:
   - На вход подается действие (action) от агента RL или Pure Pursuit контроллера.
   - Робот выполняет движение, обновляется состояние среды.
   - Генерируются наблюдения (observations), включающие визуальные и кинематические данные.
   - Вычисляется награда (reward) и определяется, завершился ли эпизод (terminated/truncated).
   - В демонстрационном режиме действия записываются для имитационного обучения.

4. **Генерация наблюдений**:
   - Собираются данные с камеры (RGB и глубина), кинематические параметры робота (скорость, ориентация).
   - Если включен графовый режим, добавляется эмбеддинг сцены на основе графа объектов.

5. **Управление сценой**:
   - Генерируются препятствия и целевые позиции.
   - Проверяется столкновение робота с препятствиями или стенами.

6. **Контроллер движения**:
   - В демонстрационном режиме используется Pure Pursuit для генерации управляющих сигналов (линейная и угловая скорость).
   - Путь строится с помощью алгоритма Dijkstra или A* на основе графа сцены.

7. **Обработка наград и завершение эпизода**:
   - Награда зависит от расстояния до цели и ориентации робота.
   - Эпизод завершается при достижении цели, столкновении или превышении времени.

8. **Логирование и оценка**:
   - Сохраняются метрики успешности (Success Rate, SR).
   - Логируются действия и результаты для анализа.

---

### Подробное описание компонентов

#### 1. **CLGRCENV (env(1).py)** — основная среда RL
**Роль**: Управляет симуляцией, взаимодействием с роботом, генерацией наблюдений, вычислением наград и логированием.

- **Инициализация (`__init__`)**:
  - **Вход**: Конфигурация (`MainConfig`), параметры симуляции (skip_frame, physics_dt, rendering_dt, max_episode_length, seed, reward_mode).
  - **Выход**: Настроенная среда с роботом, камерами, сценой и RL-параметрами.
  - **Функции**:
    - Создает симуляцию в Isaac Sim (`SimulationApp`).
    - Инициализирует мир (`World`), робота (`WheeledRobot`), контроллер (`DifferentialController`).
    - Настраивает камеры для получения RGB и глубины.
    - Определяет пространство действий: `Box(low=-1, high=1, shape=(2,))` (линейная и угловая скорость).
    - Определяет пространство наблюдений: `Box(shape=(2060,))` (включает кинематику, визуальные эмбеддинги, графовые эмбеддинги).
    - Инициализирует модули: `Scene_controller`, `Control_module`, `Graph_manager`, `LocalizationModule` (если включены).
    - Загружает CLIP-модель для обработки изображений и текста.

- **Сброс среды (`reset`)**:
  - **Вход**: Seed (опционально), options (опционально).
  - **Выход**: Начальные наблюдения, пустой словарь info.
  - **Функции**:
    - Генерирует случайный ключ (`generate_random_key`) для конфигурации сцены (число препятствий).
    - Вызывает `Scene_controller.generate_obstacles` для создания препятствий.
    - Получает целевую позицию (`Scene_controller.get_target_position`).
    - Устанавливает начальную позицию робота (`Scene_controller.get_robot_position`) с учетом радиуса и угла.
    - Вызывает `set_env` для настройки сцены (удаление старых объектов, создание новых).
    - Если `demonstrate=True`, обновляет `Control_module` для Pure Pursuit.
    - Возвращает наблюдения через `get_observations`.

- **Шаг среды (`step`)**:
  - **Вход**: Действие (`action`) — массив `[forward, angular]` в диапазоне `[-1, 1]`.
  - **Выход**: Наблюдения, награда, terminated, truncated, info, message_to_callback.
  - **Функции**:
    - Получает текущие наблюдения (`get_observations`).
    - Выполняет движение робота через `move`:
      - Если `demonstrate=False`, преобразует действие в скорости и применяет их (`DifferentialController.forward`).
      - Если `demonstrate=True`, использует `Control_module.pure_pursuit_controller` для получения скоростей.
    - Обновляет состояние мира (`World.step`).
    - Получает истинные наблюдения (`get_gt_observations`): позиция, ориентация, расстояние до цели, ошибка ориентации.
    - Вычисляет награду (`get_reward`) на основе расстояния до цели и ориентации.
    - Проверяет завершение эпизода (`_get_terminated`, `_is_timeout`, `_is_collision`).
    - Логирует успехи (`get_success_rate`) и обновляет режим обучения (`change_reward_mode`).
    - Возвращает данные для следующего шага.

- **Генерация наблюдений (`get_observations`)**:
  - **Вход**: Текущее состояние мира.
  - **Выход**: Массив наблюдений (2060 элементов).
  - **Функции**:
    - Получает RGB и глубину с камеры через `rgb_annotator` и `depth_annotator`.
    - Преобразует RGB в эмбеддинг с помощью CLIP (`clip_model.encode_image`).
    - Генерирует текстовый эмбеддинг для задачи ("go to the bowl wall with 1/2 color") через CLIP.
    - Добавляет линейную и угловую скорость робота.
    - Если `use_graph=True`, добавляет эмбеддинг графа сцены (`get_graph_embedding`).
    - Возвращает конкатенированный тензор, преобразованный в numpy-массив.

- **Управление сценой (`set_env`)**:
  - **Вход**: Конфигурация, ключ, свойства препятствий, целевая позиция.
  - **Выход**: Обновленная сцена в симуляции.
  - **Функции**:
    - Удаляет старые препятствия и цель.
    - Создает новые препятствия (table, chair, trashcan, vase) с семантическими тегами.
    - Добавляет цель (чашу) в заданной позиции.
    - Обновляет граф сцены в кэше (`scene_graphs_cache`), если `use_graph=True`.

- **Обработка столкновений (`_on_contact_report_event`)**:
  - **Вход**: Данные о контактах от физического движка.
  - **Выход**: Устанавливает флаг `collision=True` при столкновении.
  - **Функции**:
    - Проверяет количество контактов (игнорирует контакт с полом).
    - Устанавливает флаг, если контактов больше одного.

- **Награды (`get_reward`, `_get_terminated`)**:
  - **Вход**: Наблюдения, режим наград.
  - **Выход**: Награда, флаги terminated/truncated.
  - **Функции**:
    - Если расстояние до цели < 1.3, достигается цель "move".
    - Если ошибка ориентации < 15°, достигается цель "rotation".
    - Награда:
      - -0.2 за каждый шаг (штраф за время).
      - +5 при достижении обеих целей (terminated=True).
      - -5 при превышении времени.
      - -6 при столкновении.
    - Проверяет условия завершения эпизода.

#### 2. **Control_module (control_manager.py)** — контроллер движения
**Роль**: Управляет движением робота в демонстрационном режиме, строит путь к цели и генерирует управляющие сигналы.

- **Инициализация (`__init__`)**:
  - **Вход**: Параметры PID (kp, ki, kd для линейного и углового управления).
  - **Выход**: Объект с начальными параметрами и пустым путем.
  - **Функции**:
    - Устанавливает коэффициенты PID.
    - Инициализирует переменные для пути, позиций, флагов.

- **Обновление (`update`)**:
  - **Вход**: Текущая позиция, целевая позиция, позиции всех целей, контроллер сцены, ключ.
  - **Выход**: Обновляет внутренние параметры и путь.
  - **Функции**:
    - Сохраняет входные данные.
    - Вызывает `get_path` для построения пути.
    - Сбрасывает флаги (`start`, `end`, `ferst_ep`).

- **Построение пути (`get_path`)**:
  - **Вход**: Текущая позиция, целевая позиция, алгоритм (0=A*, 1=Dijkstra), контроллер сцены, флаг сохранения.
  - **Выход**: Список точек пути в реальных координатах.
  - **Функции**:
    - Выбирает алгоритм (по умолчанию Dijkstra).
    - Вызывает `get_path_dijkstra` или `get_path_A_star`.
    - Упрощает путь (`remove_zigzags`, `remove_straight_segments`).
    - Сохраняет путь на диск (`save_to_disk`), если требуется.
    - Преобразует путь в реальные координаты с учетом сдвига и масштаба.

- **Dijkstra (`get_path_dijkstra`)**:
  - **Вход**: Текущая позиция, целевая позиция, контроллер сцены, ключ.
  - **Выход**: Список узлов пути в сетке.
  - **Функции**:
    - Получает граф сцены (`get_scene_grid`).
    - Находит ближайшие достижимые узлы для старта и цели (`find_nearest_reachable_node`).
    - Загружает или генерирует кэш путей (`all_paths`) из JSON.
    - Извлекает путь из кэша для текущей конфигурации.
    - Если путь отсутствует, возвращает пустой список.

- **A* (`get_path_A_star`)**:
  - **Вход**: Текущая позиция, целевая позиция, контроллер сцены, ключ.
  - **Выход**: Список узлов пути в сетке.
  - **Функции**:
    - Аналогично Dijkstra, но использует `nx.astar_path` с эвристикой (евклидово расстояние).

- **Создание графа сцены (`get_scene_grid`)**:
  - **Вход**: Контроллер сцены.
  - **Выход**: Граф `networkx.Graph` с весами ребер.
  - **Функции**:
    - Создает сетку с диагональными связями (`_create_grid_with_diagonals`).
    - Удаляет ребра, пересекающие препятствия или стены (`Scene_controller.intersect_with_obstacles`, `intersect_with_walls`).
    - Устанавливает веса ребер (`assign_edge_weights`) с учетом границ.
    - Кэширует граф в `scene_graphs_cache` и сохраняет в JSON.

- **Pure Pursuit (`pure_pursuit_controller`)**:
  - **Вход**: Текущая позиция, ориентация (в радианах), линейная скорость, дистанция lookahead.
  - **Выход**: Линейная и угловая скорость.
  - **Функции**:
    - Проверяет близость к цели или концу пути (`end=True`).
    - Если `start=True`, поворачивает робота к началу пути.
    - Находит точку lookahead на пути (`get_lookahead_point`).
    - Вычисляет угол к точке и кривизну траектории.
    - Генерирует скорости с ограничениями.

- **Упрощение пути (`remove_zigzags`, `remove_straight_segments`)**:
  - **Вход**: Список узлов пути.
  - **Выход**: Упрощенный путь.
  - **Функции**:
    - Удаляет зигзаги и прямые сегменты для сглаживания траектории.

- **Сохранение на диск (`save_to_disk`)**:
  - **Вход**: Упрощенный и полный путь, контроллер сцены, ключ.
  - **Выход**: Сохраняет визуализацию графа в PNG.
  - **Функции**:
    - Рисует граф с узлами (зеленые — упрощенный путь, красные — полный, голубые — остальные).
    - Сохраняет изображение в лог-директорию.

#### 3. **Scene_controller (scene_manager.py)** — управление сценой
**Роль**: Генерирует конфигурацию сцены (препятствия, цели, начальные позиции робота), проверяет столкновения.

- **Инициализация (`__init__`)**:
  - **Вход**: Конфигурация (`MainConfig`).
  - **Выход**: Объект с параметрами сцены.
  - **Функции**:
    - Инициализирует списки препятствий, целей, возможных позиций.
    - Устанавливает радиус робота (0.34 м).
    - Вызывает `generate_positions_for_contole_module`.

- **Генерация препятствий (`generate_obstacles`)**:
  - **Вход**: Ключ (число препятствий), позиции препятствий (опционально).
  - **Выход**: Список препятствий.
  - **Функции**:
    - Вызывает `generate_positions_for_contole_module`.
    - Создает препятствия типа "chair" в заданных позициях.

- **Получение препятствий (`get_obstacles`)**:
  - **Вход**: Ключ, позиции препятствий.
  - **Выход**: Список препятствий, возможные позиции, ID препятствий.
  - **Функции**:
    - Возвращает текущую конфигурацию сцены.

- **Создание препятствия (`_set_obstacle`)**:
  - **Вход**: Тип, позиция, параметры (радиус, размеры).
  - **Выход**: Словарь с параметрами препятствия.
  - **Функции**:
    - Формирует словарь с типом, позицией и размерами.

- **Проверка столкновений (`intersect_with_obstacles`, `intersect_with_walls`)**:
  - **Вход**: Позиция робота, дополнительный радиус.
  - **Выход**: Булев флаг столкновения.
  - **Функции**:
    - Проверяет пересечение с препятствиями (радиус 0.35 м).
    - Проверяет выход за стены комнаты.

- **Получение целевой позиции (`get_target_position`)**:
  - **Вход**: Флаг `not_change` (сохранить текущую цель).
  - **Выход**: Позиция цели, список всех целей, индекс цели.
  - **Функции**:
    - Выбирает случайную цель из предопределенного списка.
    - Возвращает текущую цель, если `not_change=True`.

- **Получение начальной позиции робота (`get_robot_position`)**:
  - **Вход**: Координаты цели, радиус, угол, флаг настройки.
  - **Выход**: Позиция робота, угол ориентации, флаг успеха.
  - **Функции**:
    - Генерирует случайную позицию вокруг цели с заданным радиусом.
    - Проверяет отсутствие столкновений и допустимую область.
    - Вычисляет ориентацию к цели с учетом квадранта.

#### 4. **Graph_manager (control_manager.py, env(1).py)** — управление графом сцены
**Роль**: Генерирует эмбеддинги графа сцены для наблюдений.

- **Инициализация (`Graph_manager.__init__`)**:
  - **Вход**: Нет.
  - **Выход**: Пустой объект.
  - **Функции**:
    - Подготавливает место для нейросети (не реализовано).

- **Получение эмбеддинга (`get_graph_embedding` в `env(1).py`)**:
  - **Вход**: Ключ сцены.
  - **Выход**: Тензор эмбеддинга (5, 390).
  - **Функции**:
    - Загружает JSON-описание сцены из кэша.
    - Вызывает `get_simple_embedding_from_json` для генерации эмбеддинга.
    - Кэширует результат для повторного использования.

- **Генерация эмбеддинга (`get_simple_embedding_from_json`)**:
  - **Вход**: JSON-данные, модель, максимум объектов, устройство.
  - **Выход**: Тензор (max_objects, 390).
  - **Функции**:
    - Использует `SentenceTransformer` для кодирования текстовых описаний.
    - Комбинирует текстовые (384) и числовые (bbox_center, bbox_extent, 6) признаки.
    - Формирует тензор фиксированного размера.

#### 5. **LocalizationModule (env(1).py)** — локализация робота
**Роль**: Определяет позицию робота на основе RGB, глубины и графа сцены (не используется в текущем пайплайне).

- **Локализация (`localize`)**:
  - **Вход**: RGB, глубина, граф сцены, текущая позиция и ориентация робота.
  - **Выход**: Позиция робота [x, y, theta].
  - **Функции**:
    - Детектирует объекты с помощью YOLO.
    - Строит локальный граф на основе глубины и углов.
    - Выравнивает локальный и глобальный графы с помощью ICP.
    - Возвращает позицию робота.

---

### Последовательность работы пайплайна

1. **Инициализация**:
   - Создается `CLGRCENV`, настраиваются симуляция, робот, камеры, модули.
   - Инициализируется `Scene_controller` с возможными позициями препятствий.
   - Создается `Control_module` с параметрами Pure Pursuit.

2. **Сброс среды**:
   - Генерируется ключ (`CLGRCENV.generate_random_key`).
   - Создаются препятствия (`Scene_controller.generate_obstacles`).
   - Выбирается цель (`Scene_controller.get_target_position`).
   - Устанавливается начальная позиция робота (`Scene_controller.get_robot_position`).
   - Сцена обновляется (`CLGRCENV.set_env`).
   - Если `demonstrate=True`, вызывается `Control_module.update` для построения пути (`get_path_dijkstra` → `get_scene_grid`).

3. **Цикл шагов**:
   - **Получение действия**:
     - Если `demonstrate=False`, агент RL предоставляет действие `[forward, angular]`.
     - Если `demonstrate=True`, вызывается `Control_module.pure_pursuit_controller`.
   - **Выполнение движения** (`CLGRCENV.move`):
     - Действие преобразуется в скорости и применяется через `DifferentialController`.
     - Мир обновляется (`World.step`).
   - **Генерация наблюдений** (`CLGRCENV.get_observations`):
     - Получаются RGB и глубина.
     - Генерируются эмбеддинги CLIP и графа (если `use_graph=True`).
   - **Вычисление наград** (`CLGRCENV.get_reward`):
     - Оценивается расстояние и ориентация (`get_gt_observations`).
     - Проверяются столкновения и время.
   - **Логирование**:
     - Обновляется Success Rate (`get_success_rate`).
     - Сохраняются действия для имитационного обучения.

4. **Завершение эпизода**:
   - При достижении цели, столкновении или таймауте вызывается `reset`.

---

### Входы и выходы ключевых методов

| Метод | Вход | Выход |
|-------|------|-------|
| `CLGRCENV.__init__` | Конфигурация, параметры симуляции | Настроенная среда |
| `CLGRCENV.reset` | Seed, options | Наблюдения, info |
| `CLGRCENV.step` | Действие `[forward, angular]` | Наблюдения, награда, terminated, truncated, info, callback |
| `CLGRCENV.get_observations` | Состояние мира | Массив (2060,) |
| `CLGRCENV.get_reward` | Наблюдения | Награда, terminated, truncated |
| `Control_module.update` | Позиции, контроллер сцены, ключ | Обновленный путь |
| `Control_module.get_path` | Позиции, алгоритм, контроллер сцены | Список точек пути |
| `Control_module.pure_pursuit_controller` | Позиция, ориентация, параметры | Скорости (линейная, угловая) |
| `Scene_controller.generate_obstacles` | Ключ, позиции | Список препятствий |
| `Scene_controller.get_target_position` | Флаг `not_change` | Цель, список целей, индекс |
| `Scene_controller.get_robot_position` | Координаты цели, радиус, угол | Позиция, угол, флаг |

---

### Заключение

Пайплайн объединяет симуляцию, RL и демонстрационный контроль для навигации робота. `CLGRCENV` координирует работу, `Scene_controller` управляет сценой, а `Control_module` обеспечивает движение в демонстрационном режиме. Наблюдения включают визуальные и графовые данные, что делает систему гибкой для RL и имитационного обучения. Локализация (`LocalizationModule`) пока не интегрирована, но может улучшить точность позиционирования в будущем.
