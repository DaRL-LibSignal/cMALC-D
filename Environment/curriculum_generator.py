import torch
import json
import random
import re
import numpy as np
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Union, Any, Set
from abc import ABC, abstractmethod
from Environment.utils.utils import *
from collections import deque
import datetime
import torch
import random
import json
import re
from typing import List, Dict
import copy


from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


class CarTaskGenerator:
    def __init__(self, model_name: str, device: str, run_type: str = "llm", seed=None):
        self.model_name = model_name
        self.device = device
        self.run_type = run_type
        self.model = None
        self.tokenizer = None
        self.seed = seed
        self.rng = random.Random(seed) if seed is not None else None

        self.car_parameters = [
            "length",
            "width",
            "maxPosAcc",
            "maxNegAcc",
            "usualPosAcc",
            "usualNegAcc",
            "minGap",
            "maxSpeed",
            "headwayTime",
        ]
        self.car_regex = build_car_regex(self.car_parameters)

        if self.run_type == "llm":
            self._load_model()

        self.car_param_ranges = [
            (1, 10, float),
            (1, 5, float),
            (0.5, 5, float),
            (0.5, 5, float),
            (1, 5, float),
            (1, 5, float),
            (1, 10, float),
            (3, 15, float),
            (1, 5, int),
        ]

        self.previous_config_output = None
        self.num_previous_results = 0

    def _load_model(self):
        if self.model is None:
            # Load vLLM engine
            self.model = LLM(
                model=self.model_name,
                trust_remote_code=True,
                dtype="float16",
                gpu_memory_utilization=0.90,
                tensor_parallel_size=2,
            )

        if self.tokenizer is None:
            # Load tokenizer separately
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True, use_fast=False
            )
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.padding_side = "left"

    def convert_previous_results(self, previous_results):
        previous_results = [
            {
                "task": result["task"],
                "metrics": {
                    k: format(v, ".3f") if isinstance(v, float) else v
                    for k, v in result["metrics"].items()
                },
            }
            for result in previous_results
        ]
        return previous_results

    def _generate_with_llm(self, previous_results: List = []) -> List[Dict]:
        try:
            prompt = get_car_evolutionary_prompt(previous_results)
            conversation = [
                {"role": "system", "content": prompt},
            ]
            if previous_results:
                previous_results = self.convert_previous_results(previous_results)
                # print(previous_results)
                context = "\n".join(
                    [
                        f"Epoch {i+1}: {json.dumps(result)}"
                        for i, result in enumerate(previous_results)
                    ]
                )
                conversation.append(
                    {
                        "role": "user",
                        "content": f"The results from each epoch are as follows: \n{context}\nPlease generate your insights and new parameters.",
                    }
                )
            sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=400)

            outputs = self.model.chat([conversation], sampling_params)

            output_text = outputs[0].outputs[0].text
            matches = list(re.finditer(self.car_regex, output_text))
            matches = [match.group(0).replace("'", '"') for match in matches]

            if not matches:
                raise ValueError("No config generated")

            configs = json.loads(matches[0])
            self.previous_config_output = configs
            return configs

        except Exception as e:
            print(f"Error during LLM generation: {str(e)}")
            raise

    def _generate_param_value(self, min_val, max_val, val_type, rng):
        return (
            rng.randint(min_val, max_val)
            if val_type == int
            else rng.uniform(min_val, max_val)
        )

    def _generate_random_params(self) -> List[Dict]:
        param_set = {}
        rng = self.rng if self.rng is not None else random
        if self.rng is None:
            rng.seed(int(datetime.datetime.now().strftime("%Y%m%d%H%M%S")))

        for param, (min_val, max_val, val_type) in zip(
            self.car_parameters, self.car_param_ranges
        ):
            param_set[param] = self._generate_param_value(
                min_val, max_val, val_type, rng
            )

        return param_set

    def generate_task(self, previous_results: List = []) -> List[Dict]:
        if self.run_type == "llm":
            for _ in range(5):
                try:
                    result = self._generate_with_llm(previous_results)
                    return result
                except (ValueError, json.JSONDecodeError):
                    continue
            raise ValueError("Failed to generate valid parameters after 5 attempts")
        elif self.run_type == "random":
            return self._generate_random_params()
        else:
            raise ValueError(f"Invalid run_type: {self.run_type}")


class BaseCurriculum(ABC):
    def __init__(self, task_generator, config: Dict[str, Any]):
        self.task_generator = task_generator
        self.config = config
        self.current_task_index = 0
        self.current_task = None
        self.task_history = []

    @abstractmethod
    def update_curriculum(self, metrics: Dict[str, Any]) -> bool:
        pass

    @abstractmethod
    def generate_new_task(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_next_task(self) -> Dict[str, Any]:
        pass


class NoneCurriculum(BaseCurriculum):
    def __init__(self, task_generator, config: Dict[str, Any]):
        super().__init__(task_generator, config)

    def update_curriculum(self, metrics: Dict[str, Any]) -> bool:
        pass

    def generate_new_task(self) -> Dict[str, Any]:
        return {}

    def get_next_task(self) -> Dict[str, Any]:
        return {}


class DomainRandomization(BaseCurriculum):
    def __init__(self, task_generator, config: Dict[str, Any]):
        super().__init__(task_generator, config)

    def generate_new_task(self) -> Dict[str, Any]:
        self.current_task_index += 1
        self.current_task = self.task_generator.generate_task()
        self.task_history.append((self.current_task_index, self.current_task))
        return self.current_task

    def get_next_task(self) -> Dict[str, Any]:
        return self.generate_new_task()

    def update_curriculum(self, metrics: Dict[str, Any]) -> bool:
        pass


class SPACE(BaseCurriculum):
    def __init__(self, task_generator, config):
        super().__init__(task_generator, config)
        self.eta = config.get("eta", 0.05)
        self.kappa = config.get("kappa", 1)
        self.buffer_size = config.get("buffer_size", 1000)

        self.V_prev = {}  # Stores V_{t-1}(context)
        self.V_curr = {}  # Stores V_t(context)
        self.I_curr = []
        self.task_history = []
        self.replay_buffer = deque(maxlen=self.buffer_size)
        self.episode_count = 0
        self.S = 1

        self._initialize_curriculum()

    def _initialize_curriculum(self):
        task = self.task_generator.generate_task()
        context_id = self._hash_context(task)
        self.I_curr = [context_id]
        self.current_task = task
        self.task_history.append((context_id, task))
        self.V_prev[context_id] = 0.0
        self.V_curr[context_id] = 0.0

    def _hash_context(self, context):
        return json.dumps(context, sort_keys=True)

    def update_curriculum(self, metrics):
        context_id = self._hash_context(self.current_task)
        V_value = metrics.get("value_estimate", 0.0)
        self.V_prev[context_id] = self.V_curr.get(context_id, 0.0)
        self.V_curr[context_id] = V_value

        # Add to replay buffer
        self.replay_buffer.append((context_id, metrics.get("episode_reward", 0)))
        self.episode_count += 1

        # Check if value has plateaued across I_curr
        V_prev_mean = np.mean([self.V_prev.get(c, 0.0) for c in self.I_curr])
        V_curr_mean = np.mean([self.V_curr.get(c, 0.0) for c in self.I_curr])

        if (1 - self.eta) * V_prev_mean <= V_curr_mean <= (1 + self.eta) * V_prev_mean:
            self.S += self.kappa
            self._expand_curriculum()

        return True

    def _expand_curriculum(self):
        # Compute learning progress: dt(i) = V_t(i) - V_{t-1}(i)
        all_known_contexts = {c for c, _ in self.task_history}
        for _ in range(self.kappa * 3):  # Try enough times to get unseen contexts
            task = self.task_generator.generate_task()
            context_id = self._hash_context(task)
            if context_id not in all_known_contexts:
                self.task_history.append((context_id, task))
                self.V_prev[context_id] = 0.0
                self.V_curr[context_id] = 0.0
                all_known_contexts.add(context_id)

        dt = {
            c: self.V_curr.get(c, 0.0) - self.V_prev.get(c, 0.0)
            for c in all_known_contexts
        }

        # Select top-S contexts by dt(i)
        sorted_contexts = sorted(dt.keys(), key=lambda c: -dt[c])
        self.I_curr = sorted_contexts[: self.S]

    def generate_new_task(self):
        if not self.I_curr:
            return self.task_generator.generate_task()

        context_id = random.choice(self.I_curr)
        for ctx_id, task in self.task_history:
            if ctx_id == context_id:
                self.current_task = task
                return task

        # fallback if context not found
        new_task = self.task_generator.generate_task()
        new_id = self._hash_context(new_task)
        self.task_history.append((new_id, new_task))
        self.V_prev[new_id] = 0.0
        self.V_curr[new_id] = 0.0
        self.current_task = new_task
        return new_task

    def get_next_task(self):
        return self.current_task


class PLR(BaseCurriculum):

    def __init__(self, task_generator, config: Dict[str, Any]):
        super().__init__(task_generator, config)

        self.task_scores = defaultdict(float)
        self.task_timestamps = defaultdict(int)
        self.visited_tasks: Set[int] = set()
        self.episode_counter = 0
        self.current_task_id = None

        self.replay_prob = config.get("replay_prob", 0.1)
        self.new_task_prob = config.get("new_task_prob", 0.1)
        self.score_window = config.get("score_window", 10)
        self.score_history = defaultdict(list)
        self.rng = np.random.default_rng()

    def get_task_by_id(self, task_id: int) -> Dict[str, Any]:
        for i, task in self.task_history:
            if i == task_id:
                return task
        return None

    def generate_new_task(self) -> Dict[str, Any]:
        new_task = self.task_generator.generate_task()
        self.current_task_id = len(self.task_scores) + 1

        self.visited_tasks.add(self.current_task_id)
        self.task_scores[self.current_task_id] = 0.0
        self.task_timestamps[self.current_task_id] = self.episode_counter
        self.current_task = new_task
        self.task_history.append((self.current_task_id, new_task))
        return new_task

    def get_next_task(self) -> Dict[str, Any]:
        self.episode_counter += 1
        if self.rng.random() < self.new_task_prob:
            return self.generate_new_task()
        if not self.visited_tasks:
            return self.generate_new_task()

        tasks = list(self.visited_tasks)
        scores = np.array([self.task_scores[l] for l in tasks])
        ps = scores / (scores.sum() + 1e-6)

        recency = np.array(
            [1 / (self.episode_counter - self.task_timestamps[l] + 1e-6) for l in tasks]
        )
        pc = recency / (recency.sum() + 1e-6)

        weights = (1 - self.replay_prob) * ps + self.replay_prob * pc
        weights = weights / weights.sum()
        self.current_task_id = self.rng.choice(tasks, p=weights)
        self.current_task = self.get_task_by_id(self.current_task_id)
        return self.current_task

    def update_curriculum(self, metrics: Dict[str, Any]) -> bool:
        if self.current_task_id is None:
            return False

        raw_score = metrics["td_error_abs"]

        self.score_history[self.current_task_id].append(raw_score)
        if len(self.score_history[self.current_task_id]) > self.score_window:
            self.score_history[self.current_task_id].pop(0)
        self.task_scores[self.current_task_id] = raw_score
        self.task_timestamps[self.current_task_id] = self.episode_counter
        return True


class LLMCurriculum:
    def __init__(self, task_generator: Any, config: Optional[Dict[str, Any]] = None):
        self.task_generator = task_generator
        self.config = config or {}
        self.window_size: int = self.config.get("performance_window", 3)
        self.llm_retry_attempts: int = self.config.get("llm_retry_attempts", 3)

        self.task_history: List[Dict[str, Any]] = []
        self.current_task: Optional[List[Dict[str, Any]]] = None

        self.get_next_task()

    def get_next_task(self) -> List[Dict[str, Any]]:
        previous_results = (
            self.task_history[-self.window_size :] if self.task_history else []
        )

        task = self.task_generator.generate_task(previous_results=previous_results)
        self.current_task = task
        return task

    def get_current_task(self) -> List[Dict[str, Any]]:
        if self.current_task is None:
            raise ValueError(
                "No current task available - curriculum not initialized properly"
            )
        return self.current_task

    def update_curriculum(self, metrics: Dict[str, Any]) -> None:
        self.task_history.append(
            {
                "task": self.current_task,
                "metrics": metrics,
            }
        )


class ACCELCurriculum(BaseCurriculum):
    def __init__(self, task_generator, config: Dict[str, Any]):
        super().__init__(task_generator, config)

        # ACCEL parameters
        self.task_buffer_size = config.get("task_buffer_size", 100)
        self.initial_fill_ratio = config.get("initial_fill_ratio", 0.1)
        self.replay_prob = config.get("replay_prob", 0.5)
        self.score_threshold = config.get("score_threshold", 0.1)
        self.edit_prob = config.get("edit_prob", 0.3)
        self.tasks_generated = 0

        # Initialize buffer
        self.task_buffer = []
        self._initialize_buffer()
        self.current_task = None
        self.rng = random.Random(int(datetime.datetime.now().strftime("%Y%m%d%H%M%S")))

    def _initialize_buffer(self):
        """Fill initial buffer with random tasks"""
        num_initial = int(self.task_buffer_size * self.initial_fill_ratio)
        for _ in range(num_initial):
            self.task_buffer.append(
                {
                    "task": self.task_generator.generate_task(),
                    "score": 0.0,  # Will be updated after first evaluation
                    "count": 0,
                }
            )

    def update_curriculum(self, metrics: Dict[str, Any]) -> bool:
        """
        Update curriculum using only reward and positive_value_loss.
        Positive value loss serves as our regret score (Equation 5).
        """
        if self.current_task is None:
            return True  # First run needs a task

        regret_score = metrics["positive_value_loss"]
        reward = metrics["Train Reward"]

        # Only add to buffer if score meets threshold
        if regret_score >= self.score_threshold:
            self._add_or_replace_task(self.current_task, regret_score, reward)

        return True

    def _add_or_replace_task(self, task, score, reward):
        """Manage buffer with new tasks based on regret score"""
        new_entry = {
            "task": copy.deepcopy(task),
            "score": float(score),
            "reward": float(reward),
            "count": 0,
        }

        if len(self.task_buffer) < self.task_buffer_size:
            self.task_buffer.append(new_entry)
        else:
            # Replace task with lowest score that's worse than new one
            min_score = min([l["score"] for l in self.task_buffer])
            if score > min_score:
                idx = next(
                    i for i, l in enumerate(self.task_buffer) if l["score"] == min_score
                )
                self.task_buffer[idx] = new_entry

    def _edit_task(self, task):
        """Create a modified version of a task"""
        edited = copy.deepcopy(task)

        # Simple parameter perturbation - customize for your environment
        for i, param in enumerate(edited):
            if isinstance(edited[param], (int, float)):
                if (
                    self.rng.random() < 0.5
                ):  # 50% chance to modify each numeric parameter
                    if isinstance(edited[param], int):
                        edited[param] += self.rng.choice([-1, 1])
                    else:
                        edited[param] *= self.rng.uniform(0.5, 2)
                    edited[param] = min(
                        max(
                            edited[param],
                            self.task_generator.car_param_ranges[i][0],
                        ),
                        self.task_generator.car_param_ranges[i][1],
                    )

        return edited

    def generate_new_task(self) -> Dict[str, Any]:
        """ACCEL task selection logic"""
        # Decide new vs replay
        if self.rng.random() < self.replay_prob and self.tasks_generated > 0:
            # Replay path - select from buffer weighted by scores
            scores = np.array([l["score"] for l in self.task_buffer])
            probs = scores / scores.sum()
            selected = self.rng.choices(self.task_buffer, weights=probs, k=1)[0]
            selected["count"] += 1
            self.tasks_generated += 1

            if self.rng.random() < self.edit_prob:
                # Edit and return modified version
                edited_task = self._edit_task(selected["task"])
                self.current_task = edited_task
                return edited_task
            else:
                # Return original
                self.current_task = selected["task"]
                return selected["task"]
        else:
            # Generate new task
            new_task = self.task_generator.generate_task()
            self.current_task = new_task
            self.tasks_generated += 1
            return new_task

    def get_next_task(self):
        return self.generate_new_task()


class LLMCurriculumWithExploration:
    def __init__(self, task_generator: Any, config: Optional[Dict[str, Any]] = None):
        self.task_generator = task_generator
        self.config = config or {}
        self.window_size: int = self.config.get("performance_window", 3)
        self.llm_retry_attempts: int = self.config.get("llm_retry_attempts", 3)
        self.exploration_prob: float = self.config.get("exploration_prob", 0.1)
        self.exploration_metric: str = self.config.get(
            "exploration_metric", "Train Reward"
        )

        self.task_history: List[Dict[str, Any]] = []
        self.current_task: Optional[Dict[str, Any]] = None

        self.rng = random.Random(int(datetime.datetime.now().strftime("%Y%m%d%H%M%S")))
        self.get_next_task()

    def get_next_task(self) -> Dict[str, Any]:
        if self.current_task and self.rng.random() < self.exploration_prob:
            # Randomly choose a historical task with valid metric
            candidates = [
                entry
                for entry in self.task_history
                if self.exploration_metric in entry["metrics"]
            ]

            if candidates:
                random_entry = self.rng.choice(candidates)
                random_task = random_entry["task"]
                task = self.combine_tasks(self.current_task, random_task)
                self.current_task = task
                return task

        # Default behavior: use LLM to generate task
        previous_results = (
            self.task_history[-self.window_size :] if self.task_history else []
        )
        task = self.task_generator.generate_task(previous_results=previous_results)
        self.current_task = task
        return task

    def get_current_task(self) -> Dict[str, Any]:
        if self.current_task is None:
            raise ValueError(
                "No current task available - curriculum not initialized properly"
            )
        return self.current_task

    def update_curriculum(self, metrics: Dict[str, Any]) -> None:
        self.task_history.append(
            {
                "task": self.current_task,
                "metrics": metrics,
            }
        )

    def combine_tasks(
        self, task_a: Dict[str, Any], task_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        alpha = self.config.get("blend_ratio", 0.5)
        combined_task = {}

        for i, key in enumerate(task_a):
            if key in task_b and isinstance(task_a[key], (int, float)):
                val = (1 - alpha) * task_a[key] + alpha * task_b[key]

                # Enforce bounds
                min_val, max_val, dtype = self.task_generator.car_param_ranges[i]
                val = max(min_val, min(max_val, val))
                if dtype == int:
                    val = int(round(val))

                combined_task[key] = val
            else:
                combined_task[key] = task_a[key]  # fallback to base task

        return combined_task


class LLMCurriculumWithDiversity:
    def __init__(self, task_generator: Any, config: Optional[Dict[str, Any]] = None):
        self.task_generator = task_generator
        self.config = config or {}
        self.window_size: int = self.config.get("performance_window", 3)
        self.similarity_threshold: float = self.config.get("similarity_threshold", 0.1)
        self.max_similarity_count: int = self.config.get("max_similarity_count", 3)
        self.blend_ratio: float = self.config.get("blend_ratio", 0.5)

        self.task_history: List[Dict[str, Any]] = []
        self.current_task: Optional[Dict[str, Any]] = None

        self.similar_task_count = 0
        self.rng = random.Random(int(datetime.datetime.now().strftime("%Y%m%d%H%M%S")))
        self.get_next_task()

    def get_next_task(self) -> Dict[str, Any]:
        if (
            self.current_task is not None
            and self.similar_task_count >= self.max_similarity_count
            and len(self.task_history) > 0
        ):
            # Too many similar tasks — force a mutation by blending with a random historical task
            random_entry = self.rng.choice(self.task_history)
            mutated_task = self.combine_tasks(self.current_task, random_entry["task"])
            self.current_task = mutated_task
            self.similar_task_count = 0  # reset counter after mutation
            return mutated_task

        # Otherwise, generate a task normally
        previous_results = (
            self.task_history[-self.window_size :] if self.task_history else []
        )
        task = self.task_generator.generate_task(previous_results=previous_results)

        # Check similarity
        if self.current_task is not None:
            sim = self.task_similarity(self.current_task, task)
            if sim >= self.similarity_threshold:
                print("Similar task count: ", self.similar_task_count)
                self.similar_task_count += 1
            else:
                self.similar_task_count = 0

        self.current_task = task
        return task

    def get_current_task(self) -> Dict[str, Any]:
        if self.current_task is None:
            raise ValueError(
                "No current task available - curriculum not initialized properly"
            )
        return self.current_task

    def update_curriculum(self, metrics: Dict[str, Any]) -> None:
        self.task_history.append(
            {
                "task": self.current_task,
                "metrics": metrics,
            }
        )

    def task_similarity(self, task_a: Dict[str, Any], task_b: Dict[str, Any]) -> float:
        # Cosine similarity or 1 - normalized L2 distance
        distance = 0.0
        count = 0

        for key in task_a:
            if key in task_b and isinstance(task_a[key], (int, float)):
                a, b = task_a[key], task_b[key]
                norm = max(abs(a), abs(b), 1e-5)
                distance += ((a - b) / norm) ** 2
                count += 1

        l2_dist = (distance / count) ** 0.5 if count > 0 else 1.0
        similarity = 1.0 - l2_dist  # similarity in [0,1]
        return similarity

    def combine_tasks(
        self, task_a: Dict[str, Any], task_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        alpha = self.blend_ratio
        combined_task = {}

        for i, key in enumerate(task_a):
            if key in task_b and isinstance(task_a[key], (int, float)):
                val = (1 - alpha) * task_a[key] + alpha * task_b[key]
                min_val, max_val, dtype = self.task_generator.car_param_ranges[i]
                val = max(min_val, min(max_val, val))
                if dtype == int:
                    val = int(round(val))
                combined_task[key] = val
            else:
                combined_task[key] = task_a[key]

        return combined_task
