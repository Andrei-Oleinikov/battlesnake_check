"""
Fast MCTS-based Battlesnake with balanced heuristic fallback.
Time budget: <500 ms, typically runs 300+ simulations.
"""

import random
import time
import math
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

Point = Tuple[int, int]

DIRECTIONS: Dict[str, Point] = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}

HUNGRY_THRESHOLD = 50
MCTS_TIME_LIMIT_MS = 480      # leave 20 ms for overhead
MCTS_C = 1.2
ROLLOUT_DEPTH = 8
DEATH_SCORE = -1000.0


def get_info() -> Dict[str, str]:
    return {
        "apiversion": "1",
        "author": "fast_mcts",
        "color": "#FFC0CB",
        "head": "silly",
        "tail": "round-bum",
        "version": "3.0.0",
    }


# -------------------------------------------------------------------
#  Main move selection
# -------------------------------------------------------------------

def choose_move(game_state: Dict) -> str:
    start_time = time.time()
    # Always attempt MCTS first
    try:
        move = fast_mcts_move(game_state, MCTS_TIME_LIMIT_MS, start_time)
    except Exception:
        move = None
    if move is None:
        move = choose_move_heuristic(game_state)
    # Final safety net: if the selected move is illegal (shouldn't happen),
    # pick the first legal direction.
    board = game_state["board"]
    head = (game_state["you"]["head"]["x"], game_state["you"]["head"]["y"])
    width, height = board["width"], board["height"]
    occupied = _all_occupied_except_own_tail(game_state)
    if not _is_legal(move, head, occupied, width, height):
        for m in DIRECTIONS:
            if _is_legal(m, head, occupied, width, height):
                return m
        return "up"   # truly trapped
    return move


# -------------------------------------------------------------------
#  Fast MCTS (uses a lightweight array-based state for speed)
# -------------------------------------------------------------------

class FastState:
    """Minimal game state that supports fast copying and simulation."""
    __slots__ = ('width', 'height', 'bodies', 'health', 'food')

    def __init__(self, width, height, bodies, health, food):
        self.width = width
        self.height = height
        # bodies: list of lists of (x,y); each snake body, head first
        self.bodies = bodies          # list[list[Point]]
        self.health = health          # list[int]
        self.food = set(food)

    def copy(self) -> 'FastState':
        return FastState(
            self.width,
            self.height,
            [body[:] for body in self.bodies],
            self.health[:],
            self.food.copy()
        )

    def legal_moves(self, idx: int) -> List[str]:
        body = self.bodies[idx]
        head = body[0]
        moves = []
        for d, (dx, dy) in DIRECTIONS.items():
            nxt = (head[0] + dx, head[1] + dy)
            if not (0 <= nxt[0] < self.width and 0 <= nxt[1] < self.height):
                continue
            # collision check: own body excluding tail, other bodies including tail
            collision = False
            for i, b in enumerate(self.bodies):
                if i == idx:
                    # exclude own tail (last segment) if length > 1
                    if nxt in b[:-1]:
                        collision = True
                        break
                else:
                    if nxt in b:
                        collision = True
                        break
            if not collision:
                moves.append(d)
        return moves

    def apply_moves(self, moves: Dict[int, str]) -> 'FastState':
        """Return new state after all snakes move. moves: snake_index -> direction.
        Any snake not in moves stays stationary (dies if collided with)."""
        n_snakes = len(self.bodies)
        new_state = self.copy()

        new_heads = [None] * n_snakes
        # Compute new heads for moved snakes
        for idx, d in moves.items():
            if d is not None and d in DIRECTIONS:
                dx, dy = DIRECTIONS[d]
                hx, hy = self.bodies[idx][0]
                new_heads[idx] = (hx + dx, hy + dy)

        # Keep track of which snakes will die
        dead = [False] * n_snakes
        eaten_food = set()

        # Detect head-to-head / wall / body collisions
        # Map cell -> list of snakes moving into it
        cell_to_snakes: Dict[Point, List[int]] = {}
        for idx, nh in enumerate(new_heads):
            if nh is None:
                continue
            cell_to_snakes.setdefault(nh, []).append(idx)

        # Stationary heads (snakes that didn't move)
        stationary_heads = {}
        for idx in range(n_snakes):
            if idx not in moves or moves[idx] is None:
                h = self.bodies[idx][0]
                stationary_heads[h] = idx

        # Process collisions
        for cell, sids in cell_to_snakes.items():
            # If this cell contains a stationary head, all movers die
            if cell in stationary_heads:
                for sid in sids:
                    dead[sid] = True
                continue

            if len(sids) > 1:
                # multiple snakes head here: longest survives, if unique
                lengths = {sid: len(self.bodies[sid]) for sid in sids}
                max_len = max(lengths.values())
                longest = [sid for sid, l in lengths.items() if l == max_len]
                if len(longest) == 1:
                    survivor = longest[0]
                    for sid in sids:
                        if sid != survivor:
                            dead[sid] = True
                else:
                    for sid in sids:
                        dead[sid] = True
            else:
                sid = sids[0]
                # check food
                if cell in new_state.food:
                    eaten_food.add(cell)

        # Update bodies, health, and grow
        for idx in range(n_snakes):
            if dead[idx]:
                continue
            body = new_state.bodies[idx]
            # decrease health
            new_state.health[idx] -= 1
            if new_state.health[idx] <= 0:
                dead[idx] = True
                continue

            if idx in moves and moves[idx] is not None:
                nh = new_heads[idx]
                body.insert(0, nh)
                if nh not in eaten_food:
                    body.pop()       # didn't eat
                else:
                    new_state.health[idx] = 100  # ate, reset health
            # else stationary: nothing changes

        # Remove dead snakes and their bodies
        new_bodies = [body for i, body in enumerate(new_state.bodies) if not dead[i]]
        new_health = [h for i, h in enumerate(new_state.health) if not dead[i]]
        new_state.bodies = new_bodies
        new_state.health = new_health
        new_state.food -= eaten_food

        return new_state

    def is_terminal(self, my_idx: int) -> bool:
        return my_idx >= len(self.bodies) or len(self.bodies) <= 1

    def score(self, my_idx: int) -> float:
        if my_idx >= len(self.bodies):
            return DEATH_SCORE
        me = self.bodies[my_idx]
        head = me[0]
        my_len = len(me)
        health = self.health[my_idx]

        # Flood fill available space (excluding all tails for optimism)
        occupied = set()
        for i, b in enumerate(self.bodies):
            # exclude tail of each snake
            occupied.update(b[:-1] if len(b) > 1 else b)
        space = _flood_fill_count(head, occupied, self.width, self.height, 150)

        # Nearest food distance
        food_dist = min((_manhattan(head, f) for f in self.food), default=999)
        # Enemy lengths
        enemy_lens = [len(b) for i, b in enumerate(self.bodies) if i != my_idx]
        max_enemy = max(enemy_lens) if enemy_lens else 0

        # Danger: adjacent to bigger or equal head
        danger = 0
        for i, b in enumerate(self.bodies):
            if i == my_idx:
                continue
            ehead = b[0]
            if _manhattan(head, ehead) == 1 and len(b) >= my_len:
                danger += 1

        # Composite score
        return (space * 2.0
                + health * 0.3
                - min(food_dist, 20) * 1.2
                - danger * 8.0
                + (20.0 if my_len > max_enemy else 0))


def _flood_fill_count(start: Point, blocked: Set[Point], w: int, h: int, limit: int) -> int:
    if start in blocked:
        return 0
    seen = {start}
    stack = [start]
    cnt = 0
    while stack and cnt < limit:
        x, y = stack.pop()
        cnt += 1
        for dx, dy in DIRECTIONS.values():
            nb = (x + dx, y + dy)
            if nb in seen or nb in blocked:
                continue
            if 0 <= nb[0] < w and 0 <= nb[1] < h:
                seen.add(nb)
                stack.append(nb)
    return cnt


def _manhattan(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class MCTSNode:
    __slots__ = ('state', 'move', 'parent', 'children', 'visits', 'total_score', 'untried')

    def __init__(self, state: FastState, move=None, parent=None):
        self.state = state
        self.move = move
        self.parent = parent
        self.children: Dict[str, MCTSNode] = {}
        self.visits = 0
        self.total_score = 0.0
        self.untried: List[str] = []

    def ucb1(self, c: float) -> float:
        if self.visits == 0:
            return float('inf')
        return self.total_score / self.visits + c * math.sqrt(math.log(self.parent.visits) / self.visits)


def fast_mcts_move(game_state: Dict, time_limit_ms: float, start_time: float) -> str:
    # Convert to fast state
    board = game_state["board"]
    width, height = board["width"], board["height"]
    snakes = board["snakes"]
    my_id = game_state["you"]["id"]

    # Map snake id to index
    id_to_idx = {}
    bodies = []
    health = []
    for s in snakes:
        id_to_idx[s["id"]] = len(bodies)
        bodies.append([(seg["x"], seg["y"]) for seg in s["body"]])
        health.append(s["health"])

    my_idx = id_to_idx[my_id]
    food = {(f["x"], f["y"]) for f in board["food"]}
    root_state = FastState(width, height, bodies, health, food)

    legal = root_state.legal_moves(my_idx)
    if not legal:
        return None

    root = MCTSNode(root_state)
    root.untried = legal[:]

    while (time.time() - start_time) * 1000 < time_limit_ms:
        node = root
        # Selection
        while not node.untried and node.children:
            node = max(node.children.values(), key=lambda n: n.ucb1(MCTS_C))
        # Expansion
        if node.untried:
            move = random.choice(node.untried)
            node.untried.remove(move)
            # Build moves for all snakes
            moves = {}
            for idx in range(len(node.state.bodies)):
                if idx == my_idx:
                    moves[idx] = move
                else:
                    opp_moves = node.state.legal_moves(idx)
                    if opp_moves:
                        moves[idx] = random.choice(opp_moves)
            child_state = node.state.apply_moves(moves)
            child = MCTSNode(child_state, move, node)
            node.children[move] = child
            # Rollout
            score = _rollout(child.state, my_idx, ROLLOUT_DEPTH)
            node = child
        else:
            # Terminal or fully expanded, evaluate directly
            score = node.state.score(my_idx)

        # Backpropagation
        while node is not None:
            node.visits += 1
            node.total_score += score
            node = node.parent

    # Pick best child by visits
    if not root.children:
        return random.choice(legal)
    best_move = max(root.children.items(), key=lambda kv: kv[1].visits)[0]
    return best_move


def _rollout(state: FastState, my_idx: int, depth: int) -> float:
    for _ in range(depth):
        if state.is_terminal(my_idx):
            break
        moves = {}
        for idx in range(len(state.bodies)):
            lm = state.legal_moves(idx)
            if lm:
                moves[idx] = random.choice(lm)
        state = state.apply_moves(moves)
    return state.score(my_idx)


# -------------------------------------------------------------------
#  Balanced heuristic (fallback)
# -------------------------------------------------------------------

def choose_move_heuristic(game_state: Dict) -> str:
    board = game_state["board"]
    you = game_state["you"]
    width, height = board["width"], board["height"]
    head = (you["head"]["x"], you["head"]["y"])
    my_length = you["length"]
    health = you["health"]

    snakes = board["snakes"]
    occupied = _all_occupied_except_own_tail(game_state)
    foods = [(f["x"], f["y"]) for f in board["food"]]
    my_id = you["id"]

    # Danger: cells adjacent to bigger/equal heads
    danger = set()
    for s in snakes:
        if s["id"] == my_id or s["length"] < my_length:
            continue
        eh = (s["head"]["x"], s["head"]["y"])
        for dx, dy in DIRECTIONS.values():
            danger.add((eh[0] + dx, eh[1] + dy))

    best_move = None
    best_score = -float('inf')
    center_x, center_y = (width - 1) / 2, (height - 1) / 2

    for move, (dx, dy) in DIRECTIONS.items():
        nxt = (head[0] + dx, head[1] + dy)
        if not _is_legal(move, head, occupied, width, height):
            continue

        # Reachable space (excluding tails)
        occupied_no_tails = set()
        for s in snakes:
            body = [(seg["x"], seg["y"]) for seg in s["body"]]
            occupied_no_tails.update(body[:-1] if len(body) > 1 else body)
        space = _flood_fill_count(nxt, occupied_no_tails, width, height, my_length + 5)

        # Food attraction always, stronger when hungry
        nearest_food = min((_manhattan(nxt, f) for f in foods), default=999)
        food_bonus = max(0, (width + height - nearest_food) * (3 if health < HUNGRY_THRESHOLD else 0.8))

        # Center pull (avoid hugging walls forever)
        center_dist = abs(nxt[0] - center_x) + abs(nxt[1] - center_y)
        center_bonus = (width + height - center_dist) * 0.3

        # Penalties
        penalty = 0
        if nxt in danger:
            penalty = 10000

        score = space * 2.0 + food_bonus + center_bonus - penalty
        if score > best_score:
            best_score = score
            best_move = move

    return best_move if best_move else "up"


# -------------------------------------------------------------------
#  Helper functions
# -------------------------------------------------------------------

def _all_occupied_except_own_tail(game_state: Dict) -> Set[Point]:
    board = game_state["board"]
    you = game_state["you"]
    occupied = set()
    for s in board["snakes"]:
        for seg in s["body"]:
            occupied.add((seg["x"], seg["y"]))
    # Remove own tail if we didn't just eat (conservatively we allow moving there)
    my_body = you["body"]
    if len(my_body) > 1:
        tail = (my_body[-1]["x"], my_body[-1]["y"])
        if tail in occupied:
            occupied.remove(tail)
    return occupied


def _is_legal(move: str, head: Point, occupied: Set[Point], w: int, h: int) -> bool:
    if move not in DIRECTIONS:
        return False
    dx, dy = DIRECTIONS[move]
    nxt = (head[0] + dx, head[1] + dy)
    return 0 <= nxt[0] < w and 0 <= nxt[1] < h and nxt not in occupied
