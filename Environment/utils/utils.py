import torch
import json
import numpy as np
import gzip
import shutil
import os
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import re
import networkx as nx
import random
from vllm import LLM, SamplingParams

last_use_cuda = True


def cuda(tensor, use_cuda=None):
    """
    A cuda wrapper
    """
    global last_use_cuda
    if use_cuda is None:
        use_cuda = last_use_cuda
    last_use_cuda = use_cuda
    if not use_cuda:
        return tensor
    if tensor is None:
        return None
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor


class Fake_TXSW:
    def __init__(self):
        pass

    def add_scalar(self, *x):
        pass

    def add_image(self, *x):
        pass

    def add_graph(self, *x):
        pass

    def close(self):
        pass


class WanDB_TXSW:
    def __init__(
        self,
        wandb_api_key,
        wandb_entity_name,
        wandb_project_name,
        wandb_sync_mode,
        tensorboardx_comment,
        **kwargs,
    ):
        os.environ["WANDB_API_KEY"] = wandb_api_key
        wandb.init(
            entity=wandb_entity_name,
            project=wandb_project_name,
            mode=wandb_sync_mode,
            name=tensorboardx_comment,
            config=kwargs,
        )

    def log_one(self, name, data, step):
        wandb.log({name: data}, step)

    def add_scalar(self, *x):
        self.log_one(*x)

    def add_image(self, *x):
        self.log_one(*x)

    def add_graph(self, *x):
        raise NotImplementedError

    def close(self):
        pass


def showarray(arr):
    arr = np.array(arr)
    print("max: %.2f, min: %.2f" % (arr.max(), arr.min()))
    # plt.imshow(arr)
    # plt.show()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def _real_gzip_file(filename):
    with open(filename, "rb") as f_in:
        with gzip.open(filename + ".gz", "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(filename)


def gzip_file(filename):
    # _gzip_processor(filename)
    _real_gzip_file(filename)


# CityFlow utils


def floyd(adj_mat):
    # input: adjacent np.array, disconnect is assigned a big number
    assert len(adj_mat.shape) == 2 and adj_mat.shape[0] == adj_mat.shape[1]
    n = adj_mat.shape[0]
    res = adj_mat.copy()
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if res[i][k] + res[k][j] < res[i][j]:
                    res[i][j] = res[i][k] + res[k][j]
    return res


def get_length(points):
    res = 0
    for p1, p2 in zip(points[:-1], points[1:]):
        dx = p1["x"] - p2["x"]
        dy = p1["y"] - p2["y"]
        res += (dx**2 + dy**2) ** 0.5
    return res


def parse_roadnet(filename):
    roadnet = json.load(open(filename))
    inters = roadnet["intersections"]
    roads = {}
    for road in roadnet["roads"]:
        roads[road["id"]] = road
    res = {}
    virtual_res = {}
    for inter in inters:
        one = {}
        one["roadlinks"] = inter["roadLinks"]
        for i in one["roadlinks"]:
            lanes = set()
            detail_lanes = []
            for j in i["laneLinks"]:
                lanes.add(j["startLaneIndex"])
                detail_lanes.append(
                    [
                        i["startRoad"] + "_" + str(j["startLaneIndex"]),
                        i["endRoad"] + "_" + str(j["endLaneIndex"]),
                    ]
                )
            i["lanenumber"] = len(lanes)
            i["lanelinks"] = detail_lanes
            del i["laneLinks"]
        one["connection"] = {}  # save edges to other intersections
        for road in inter["roads"]:
            if roads[road]["startIntersection"] == inter["id"]:
                one["connection"][road] = [
                    roads[road]["endIntersection"],
                    get_length(roads[road]["points"]),
                ]
        if inter["virtual"]:
            virtual_res[inter["id"]] = one
        else:
            phase = inter["trafficLight"]["lightphases"]
            phase = [x["availableRoadLinks"] for x in phase]
            one["phases"] = phase
            res[inter["id"]] = one
    return res, virtual_res


def flatten_data(datatype, data):
    if datatype == "array":
        data = list(zip(*data))
        data = list(map(lambda x: np.stack(x), data))
        return data[0]
    elif datatype == "dict":
        dic = data
        if len(dic) == 0:
            return {}
        res = {}
        for key in dic[0].keys():
            res[key] = np.stack([x[key] for x in dic])
        return res
    else:
        raise NotImplementedError("unknown flatten type " + datatype)


def unpack_flattened_data(datatype, data):
    if datatype == "array":
        raise NotImplementedError()
    elif datatype == "dict":
        res = []
        keys = list(data.keys())
        size = len(data[keys[0]])
        for i in range(size):
            one = {}
            for key in keys:
                one[key] = data[key][i]
            res.append(one)
        return res
    else:
        raise NotImplementedError("unknown flatten type " + datatype)


def get_intersection_info(filename, intername):
    j = json.load(open(filename))
    j = j["intersections"]
    for i in j:
        if i["id"] == intername:
            return i


def build_traffic_graph(config):
    """
    Build a NetworkX graph from the road network configuration.

    Args:
        config (dict): Configuration dictionary containing 'ROADNET_INFO' and 'VIRTUAL_INTERSECTION_NAMES'

    Returns:
        nx.Graph: Constructed traffic network graph
        set: Set of virtual intersection names
    """
    G = nx.Graph()
    virtual_nodes = set(config["VIRTUAL_INTERSECTION_NAMES"].keys())

    edge_name_map = {}

    for intersection, info in config["ROADNET_INFO"].items():
        for edge, connected_intersection in info["connection"].items():
            G.add_edge(intersection, connected_intersection[0], road_name=edge)
            edge_name_map[(intersection, connected_intersection[0])] = edge

    for intersection, info in config["VIRTUAL_INTERSECTION_NAMES"].items():
        for edge, connected_intersection in info["connection"].items():
            G.add_edge(intersection, connected_intersection[0], road_name=edge)
            edge_name_map[(intersection, connected_intersection[0])] = edge

    return G, virtual_nodes, edge_name_map


def generate_flow(graph, visit_probabilities, virtual_nodes, walk_sizes, edge_name_map):
    """
    Generate traffic flow as random walks that adhere to visitation probabilities while tracking road names.

    Args:
        graph (nx.Graph): Traffic network graph with edge attributes
        visit_probabilities (dict): Target visitation probabilities for real nodes
        virtual_nodes (list): List of virtual node names
        walk_sizes (list): Desired lengths for each generated walk
        edge_name_map (dict): Mapping of (u, v) node pairs to road names

    Returns:
        list: List of dictionaries containing:
              - 'nodes': List of node names in the walk
              - 'roads': List of road names between nodes
              - 'length': Total length of the walk
        dict: Actual visitation counts
        dict: Deviation from target probabilities
    """
    # Initialize data structures
    real_nodes = [n for n in visit_probabilities.keys() if n not in virtual_nodes]
    virtual_nodes = list(virtual_nodes)
    walks = []
    visits = {n: 0 for n in real_nodes}

    # Calculate target visits based on walk sizes
    total_steps = sum(walk_sizes) - 2 * len(
        walk_sizes
    )  # Subtract virtual start/end nodes
    target_visits = {
        n: max(1, int(round(visit_probabilities[n] * total_steps))) for n in real_nodes
    }

    # Generate each walk
    for walk_length in walk_sizes:
        walk_nodes = []
        walk_roads = []

        # Start at random virtual node
        current_node = random.choice(virtual_nodes)
        walk_nodes.append(current_node)

        # Move to first real node
        neighbors = [n for n in graph.neighbors(current_node) if n in real_nodes]
        if not neighbors:
            continue  # Skip if no valid starting point

        next_node = random.choice(neighbors)
        road_name = edge_name_map.get((current_node, next_node), "unknown")
        walk_nodes.append(next_node)
        walk_roads.append(road_name)
        visits[next_node] += 1

        # Continue walk
        remaining_steps = walk_length - 2  # Subtract start and end nodes
        while remaining_steps > 0 and len(walk_nodes) < walk_length:
            current_node = next_node
            neighbors = [n for n in graph.neighbors(current_node) if n in real_nodes]

            if not neighbors:
                break  # No valid neighbors

            # Choose next node based on remaining visit needs
            neighbor_weights = [max(target_visits[n] - visits[n], 0) for n in neighbors]
            total_weight = sum(neighbor_weights)

            if total_weight > 0:
                probabilities = [w / total_weight for w in neighbor_weights]
                next_node = random.choices(neighbors, weights=probabilities)[0]
            else:
                next_node = random.choice(neighbors)

            road_name = edge_name_map.get((current_node, next_node), "unknown")
            walk_nodes.append(next_node)
            walk_roads.append(road_name)
            visits[next_node] += 1
            remaining_steps -= 1

        # End at virtual node
        end_candidates = [
            n for n in graph.neighbors(walk_nodes[-1]) if n in virtual_nodes
        ]
        if end_candidates:
            end_node = random.choice(end_candidates)
            road_name = edge_name_map.get((walk_nodes[-1], end_node), "unknown")
            walk_nodes.append(end_node)
            walk_roads.append(road_name)

        walks.append(walk_roads)

    return walks


def load_model(model_name_or_path, device="cuda"):
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True, use_fast=False
    )

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.mask_token_id = tokenizer.eos_token_id
    tokenizer.sep_token_id = tokenizer.eos_token_id
    tokenizer.cls_token_id = tokenizer.eos_token_id

    # Initialize vLLM model (engine)
    model = LLM(
        model=model_name_or_path,
        tokenizer=tokenizer,
        trust_remote_code=True,
        dtype="float16",
        # device is automatically handled in vLLM
    )

    return model, tokenizer


def update_json_key(data, target_key, new_value):
    if isinstance(data, dict):
        for key in data:
            if key == target_key:
                if isinstance(new_value, (np.number, np.ndarray)):
                    data[key] = new_value.item()
                else:
                    data[key] = new_value
            else:
                data[key] = update_json_key(data[key], target_key, new_value)
    elif isinstance(data, list):
        for i in range(len(data)):
            data[i] = update_json_key(data[i], target_key, new_value)
    return data


def get_car_results_analysis_prompt(results) -> str:
    """Simplified performance analysis prompt for car parameters"""
    return f"""
Analyze these past car parameter trials and determine how to generate the next task:

{json.dumps(results, indent=2)}

Car Performance Assessment:
1. What parameter combinations were successful?
2. What weaknesses should be addressed?
3. What logical evolutions can we make?
4. Suggest specific values or parameter patterns to evolve the curriculum.

Format your response as:
- 1-2 sentences summarizing key insights
- Then "NEXT TASK SUGGESTION:" followed by a JSON object of new car parameters satisfying the bounds:

- length: (1.0-10.0)
- width: (1.0-5.0)
- maxPosAcc: (0.5-5.0)
- maxNegAcc: (0.5-5.0)
- usualPosAcc: (1.0-5.0)
- usualNegAcc: (1.0-5.0)
- minGap: (1.0-10.0)
- maxSpeed: (3.0-15.0)
- headwayTime: (1-5, integer)
"""


def get_car_evolutionary_prompt(previous_results) -> str:
    """Prompt with evolutionary guidance using only past results"""
    if not previous_results:
        return get_car_param_prompt_default()

    analysis = get_car_results_analysis_prompt(previous_results)

    return f"""
You are a curriculum designer for traffic light simulation. Your goal is to generate a curriculum for training an agent to optimize traffic flow. This curriculum needs to test the agent's ability to handle various traffic conditions.

Use the past trial data to propose the next car configuration for training.

{analysis}
"""


def get_car_param_prompt_default() -> str:
    """Fallback car parameter generation prompt if no history is available"""
    return """
You are a curriculum designer for traffic simulations. Your goal is to generate a curriculum for training an agent to optimize traffic flow. This curriculum needs to test the agent's ability to handle various traffic conditions. Generate one plausible set of car parameters within the given bounds:

- length: (1.0-10.0)
- width: (1.0-5.0)
- maxPosAcc: (0.5-5.0)
- maxNegAcc: (0.5-5.0)
- usualPosAcc: (1.0-5.0)
- usualNegAcc: (1.0-5.0)
- minGap: (1.0-10.0)
- maxSpeed: (3.0-15.0)
- headwayTime: (1-5, integer)

Output a single JSON object of parameter values.
"""


def get_results_analysis_prompt(results) -> str:
    """Simplified performance analysis prompt for flow parameters"""
    return f"""
Analyze these past flow visitation probability tasks and propose the next one:

{json.dumps(results, indent=2)}

Performance Assessment:
1. What distributions supported good learning?
2. What edge cases or failures were observed?
3. What patterns should evolve?
4. Suggest a new distribution of probabilities across nodes.

Format your response as:
- 1-2 sentences summarizing key insights
- Then "NEXT TASK SUGGESTION:" followed by a JSON array of probabilities
"""


def get_evolutionary_prompt(n_agents: int, previous_results) -> str:
    """Prompt with evolutionary guidance for flow parameters using only history"""
    if not previous_results:
        return get_flow_param_prompt_default(n_agents)

    analysis = get_results_analysis_prompt(previous_results)

    return f"""
You are a curriculum designer for traffic light simulation with {n_agents} nodes.

Based on the following past results, suggest a new probability distribution:

{analysis}
"""


def build_car_regex(parameters):
    regex_parts = []
    for param in parameters:
        if param == "headwayTime":
            pattern = f"[\"']{param}[\"']:\\s*\\d+"
        else:
            pattern = f"[\"']{param}[\"']:\\s*[\"']?[\\d\\.]+[\"']?"
        regex_parts.append(pattern)

    regex_pattern = r"\{\s*" + r",\s*".join(regex_parts) + r"\s*\}"
    return regex_pattern


def get_flow_param_prompt_default(n_agents: int) -> str:
    """Fallback flow parameter prompt if no history is available"""
    return f"""
Generate a reasonable {n_agents}-element probability array for traffic flow through light-controlled intersections. Each value should be between 0.0 and 1.0, and they must sum to exactly 1.0.

Output a single JSON array.
"""
