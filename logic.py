"""
MCTS-based move-selection for Battlesnake with heuristic fallback and
defensive out-of-bounds protection.

Board: (0,0) bottom-left.
up=+y, down=-y, left=-x, right=+x.

Time budget: 500 ms per move.
"""

from collections import deque
import random
import time
from typing import Dict, List, Optional, Set, Tuple

Point = Tuple[int, int]

DIRECTIONS: Dict[str, Point] = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}

HEAD_TO_HEAD_PENALTY = 10_000
HUNGRY_THRESHOLD = 50

# MCTS constants
MCTS_TIME_LIMIT_MS = 450  # leave 50 ms safety margin
MCTS_C = 1.4               # UCB exploration constant
ROLLOUT_DEPTH = 15         # max rollout depth
DEATH_SCORE = -1000.0


def get_info() -> Dict[str, str]:
    return {
        "apiversion": "1",
        "author": "hackathon_mcts_v2",
        "color": "#6434eb",
        "head": "smart-caterpillar",
        "tail": "weight",
        "version": "2.0.1",
    }


# -------------------------------------------------------------------
#  Main entry point with final legality check
# -------------------------------------------------------------------

def choose_move(game_state: Dict) -> str:
    """Choose move with MCTS; fall back to enhanced heuristic.
    Final move is validated against real game state to prevent
    out-of-bounds / collision moves from ever being returned."""
    start = time.time()
    # If almost no time, skip MCTS and go straight to heuristic
    if time.time() - start > 0.4:
        move = choose_move_heuristic_enhanced(game_state)
    else:
        try:
            move = mcts_move(game_state, MCTS_TIME_LIMIT_MS)
        except Exception:
            move = choose_move_heuristic_enhanced(game_state)

    # --- Defensive check: ensure the move is actually legal right now ---
    board = game_state["board"]
    width, height = board["width"], board["height"]
    head = (game_state["you"]["head"]["x"], game_state["you"]["head"]["y"])
    occupied = _occupied_cells(board["snakes"])
    # Our own tail might be free – allow moving there if safe
    you = game_state["you"]
    my_tail = (you["body"][-1]["x"], you["body"][-1]["y"]) if len(you["body"]) > 1 else None
    if my_tail and my_tail in occupied:
        occupied.remove(my_tail)

    # If chosen move is illegal, pick the first legal move from all directions
    if not _is_move_legal(move, head, occupied, width, height):
        for m in DIRECTIONS:
            if _is_move_legal(m, head, occupied, width, height):
                return m
        # truly trapped – return a harmless direction (server will ignore anyway)
        return "up"

    return move


def _is_move_legal(move: str, head: Point, occupied: Set[Point],
                   width: int, height: int) -> bool:
    if move not in DIRECTIONS:
        return False
    dx, dy = DIRECTIONS[move]
    nxt = (head[0] + dx, head[1] + dy)
    if not (0 <= nxt[0] < width and 0 <= nxt[1] < height):
        return False
    if nxt in occupied:
        return False
    return True


# -------------------------------------------------------------------
#  Enhanced heuristic (fallback) – tail‑aware, safe
# -------------------------------------------------------------------

def choose_move_heuristic_enhanced(game_state: Dict) -> str:
    """Improved heuristic with tail‑awareness and better flood‑fill."""
    board = game_state["board"]
    you = game_state["you"]
    width, height = board["width"], board["height"]
    head = (you["head"]["x"], you["head"]["y"])
    my_length = you["length"]
    health = you["health"]

    snakes = board["snakes"]
    occupied = _occupied_cells_except_our_tail(snakes, you)
    danger = _head_to_head_cells(snakes, you["id"], my_length)
    foods = [(f["x"], f["y"]) for f in board["food"]]

    best_move = None
    best_score = float("-inf")

    for move, (dx, dy) in DIRECTIONS.items():
        nxt = (head[0] + dx, head[1] + dy)
        if not _in_bounds(nxt, width, height):
            continue
        if nxt in occupied:
            continue

        space = _flood_fill_loose(nxt, snakes, you["id"], width, height,
                                  limit=my_length + 1)
        score = float(space)

        if nxt in danger:
            score -= HEAD_TO_HEAD_PENALTY

        if foods and health < HUNGRY_THRESHOLD:
            nearest = min(_manhattan(nxt, f) for f in foods)
            score += (width + height - nearest) * 2

        if score > best_score:
            best_score = score
            best_move = move

    return best_move or "up"


def _occupied_cells_except_our_tail(snakes: List[Dict], you: Dict) -> Set[Point]:
    """All occupied cells, but remove our own tail if we won't grow."""
    occupied = set()
    my_id = you["id"]
    for snake in snakes:
        for seg in snake["body"]:
            occupied.add((seg["x"], seg["y"]))
    my_body = [(seg["x"], seg["y"]) for seg in you["body"]]
    my_tail = my_body[-1] if my_body else None
    if my_tail and my_tail in occupied:
        occupied.remove(my_tail)
    return occupied


def _flood_fill_loose(start: Point, snakes: List[Dict], my_id: str,
                      width: int, height: int, limit: int) -> int:
    """Count reachable cells, assuming enemy tails may free up."""
    occupied = set()
    for snake in snakes:
        body = [(seg["x"], seg["y"]) for seg in snake["body"]]
        for i, seg in enumerate(body):
            if i == len(body) - 1:
                # tail may disappear – don't block
                pass
            else:
                occupied.add(seg)
    seen = {start}
    stack = [start]
    count = 0
    while stack:
        x, y = stack.pop()
        count += 1
        if count >= limit:
            break
        for dx, dy in DIRECTIONS.values():
            nbr = (x + dx, y + dy)
            if nbr in seen:
                continue
            if not (0 <= nbr[0] < width and 0 <= nbr[1] < height):
                continue
            if nbr in occupied:
                continue
            seen.add(nbr)
            stack.append(nbr)
    return count


# -------------------------------------------------------------------
#  MCTS infrastructure
# -------------------------------------------------------------------

class GameState:
    """Lightweight, fast-to-copy game state for simulation."""
    __slots__ = ('width', 'height', 'snakes', 'food', 'alive')

    def __init__(self, width, height, snakes, food, alive):
        self.width = width
        self.height = height
        self.snakes = snakes        # dict id -> {'body': list of Points, 'health': int}
        self.food = set(food)       # set of Points
        self.alive = set(alive)

    def copy(self) -> 'GameState':
        new_snakes = {}
        for sid, s in self.snakes.items():
            new_snakes[sid] = {
                'body': s['body'][:],
                'health': s['health']
            }
        return GameState(self.width, self.height, new_snakes,
                         self.food.copy(), self.alive.copy())

    def legal_moves(self, snake_id: str) -> List[str]:
        """All directions that don't immediately hit wall or occupied body
        (excluding own tail if not growing)."""
        s = self.snakes[snake_id]
        head = s['body'][0]
        moves = []
        for move, (dx, dy) in DIRECTIONS.items():
            nxt = (head[0] + dx, head[1] + dy)
            if not (0 <= nxt[0] < self.width and 0 <= nxt[1] < self.height):
                continue
            # Check collision with all snake bodies (excluding our own tail)
            collision = False
            for sid2, s2 in self.snakes.items():
                if sid2 == snake_id:
                    # exclude own tail from collision check
                    body_to_check = s2['body'][:-1]  # all but tail
                    if nxt in body_to_check:
                        collision = True
                        break
                else:
                    if nxt in s2['body']:
                        collision = True
                        break
            if not collision:
                moves.append(move)
        return moves

    def apply_moves(self, moves: Dict[str, str]) -> 'GameState':
        """Return a new state after all given snakes move.
        Handles head-on-head, stationary heads, food, and health."""
        state = self.copy()

        # 1. Determine new heads for moving snakes
        new_heads = {}
        for sid, move in moves.items():
            if sid not in state.alive:
                continue
            s = state.snakes[sid]
            dx, dy = DIRECTIONS[move]
            hx, hy = s['body'][0]
            new_heads[sid] = (hx + dx, hy + dy)

        # 2. Resolve collisions and eating
        eaten_food = set()
        dead = set()

        # Build collision map: cell -> list of snake ids with new heads there
        cell_to_snakes: Dict[Point, List[str]] = {}
        for sid, nh in new_heads.items():
            cell_to_snakes.setdefault(nh, []).append(sid)

        # Also need to account for snakes that didn't move: their head cells
        # are occupied and moving into them is death for the mover.
        stationary_heads: Dict[Point, List[str]] = {}
        for sid in state.alive:
            if sid not in moves:  # didn't move
                head = state.snakes[sid]['body'][0]
                stationary_heads.setdefault(head, []).append(sid)

        for cell, sids in cell_to_snakes.items():
            # First, if this cell is also a stationary head cell,
            # the movers die (since stationary head is already there).
            if cell in stationary_heads:
                # All moving snakes into this cell die
                for sid in sids:
                    dead.add(sid)
                continue

            # Now regular head-to-head among movers
            if len(sids) > 1:
                lengths = {sid: len(state.snakes[sid]['body']) for sid in sids}
                max_len = max(lengths.values())
                max_count = sum(1 for l in lengths.values() if l == max_len)
                if max_count == 1:
                    survivor = [sid for sid, l in lengths.items() if l == max_len][0]
                    for sid in sids:
                        if sid != survivor:
                            dead.add(sid)
                else:
                    for sid in sids:
                        dead.add(sid)
            else:
                sid = sids[0]
                # Single mover to this cell, check food
                if cell in state.food:
                    eaten_food.add(cell)

        # 3. Update health and bodies for survivors
        for sid in state.alive:
            if sid in dead:
                continue
            s = state.snakes[sid]
            s['health'] -= 1
            if s['health'] <= 0:
                dead.add(sid)
                continue

            move = moves.get(sid)
            if move is None:  # stayed still
                continue
            new_head = new_heads[sid]
            body = s['body']
            body.insert(0, new_head)
            if new_head not in eaten_food:
                body.pop()
            else:
                s['health'] = 100  # ate food

        # 4. Remove eaten food
        state.food -= eaten_food

        # 5. Remove dead snakes
        for sid in dead:
            state.alive.discard(sid)
            if sid in state.snakes:
                del state.snakes[sid]

        return state

    def is_terminal(self, my_id: str) -> bool:
        return my_id not in self.alive or len(self.alive) == 1

    def score(self, my_id: str) -> float:
        if my_id not in self.alive:
            return DEATH_SCORE
        me = self.snakes[my_id]
        head = me['body'][0]
        health = me['health']
        my_len = len(me['body'])

        # Reachable space (flood fill with tail awareness)
        space = self._flood_fill_self(head, my_id)

        food_dist = float('inf')
        if self.food:
            food_dist = min(_manhattan(head, f) for f in self.food)

        enemy_lengths = [len(s['body']) for sid, s in self.snakes.items() if sid != my_id]
        max_enemy_len = max(enemy_lengths) if enemy_lengths else 0
        enemy_heads = [s['body'][0] for sid, s in self.snakes.items() if sid != my_id]

        danger_near = 0.0
        for ehead in enemy_heads:
            d = _manhattan(head, ehead)
            if d == 1:
                eid = [sid for sid in self.snakes if sid != my_id][0]  # simplistic
                if eid in self.snakes and len(self.snakes[eid]['body']) >= my_len:
                    danger_near += 10.0

        score = 0.0
        score += space * 2.0
        score += health * 0.5
        score -= min(food_dist, 20) * 1.0
        score -= danger_near
        if my_len > max_enemy_len:
            score += 20.0
        return score

    def _flood_fill_self(self, start: Point, my_id: str) -> int:
        """Flood fill from start, own tail is free, others' tails maybe free."""
        occupied = set()
        for sid, s in self.snakes.items():
            body = s['body']
            if sid == my_id and len(body) > 1:
                # exclude own tail
                occupied.update(body[:-1])
            else:
                # exclude tail of other snakes (conservative: may block)
                occupied.update(body[:-1] if len(body) > 1 else body)
        seen = {start}
        stack = [start]
        count = 0
        while stack and count < 200:
            x, y = stack.pop()
            count += 1
            for dx, dy in DIRECTIONS.values():
                nbr = (x + dx, y + dy)
                if nbr in seen:
                    continue
                if not (0 <= nbr[0] < self.width and 0 <= nbr[1] < self.height):
                    continue
                if nbr in occupied:
                    continue
                seen.add(nbr)
                stack.append(nbr)
        return count


def state_from_game(game_state: Dict) -> GameState:
    """Convert API game_state into our lightweight GameState."""
    board = game_state["board"]
    width, height = board["width"], board["height"]
    snakes = {}
    for s in board["snakes"]:
        body = [(seg["x"], seg["y"]) for seg in s["body"]]
        snakes[s["id"]] = {
            "body": body,
            "health": s["health"]
        }
    food = {(f["x"], f["y"]) for f in board["food"]}
    alive = {s["id"] for s in board["snakes"]}
    return GameState(width, height, snakes, food, alive)


class MCTSNode:
    __slots__ = ('state', 'move', 'parent', 'children', 'visits',
                 'total_score', 'untried_moves')

    def __init__(self, state: GameState, move: Optional[str] = None,
                 parent: Optional['MCTSNode'] = None):
        self.state = state
        self.move = move
        self.parent = parent
        self.children: Dict[str, MCTSNode] = {}
        self.visits = 0
        self.total_score = 0.0
        self.untried_moves: List[str] = []

    def ucb1(self, C: float) -> float:
        if self.visits == 0:
            return float('inf')
        return self.total_score / self.visits + C * (
            __import__('math').sqrt(__import__('math').log(self.parent.visits) / self.visits)
        )


def mcts_move(game_state: Dict, time_limit_ms: float) -> str:
    """Run MCTS from current state, return best move."""
    start_time = time.time()
    root_state = state_from_game(game_state)
    my_id = game_state["you"]["id"]

    root = MCTSNode(root_state)
    legal = root_state.legal_moves(my_id)
    if not legal:
        return "up"
    root.untried_moves = legal[:]

    while (time.time() - start_time) * 1000 < time_limit_ms:
        node = root
        # 1. Selection
        while node.untried_moves == [] and node.children:
            node = max(node.children.values(), key=lambda c: c.ucb1(MCTS_C))

        # 2. Expansion
        if not node.state.is_terminal(my_id) and node.untried_moves:
            move = random.choice(node.untried_moves)
            node.untried_moves.remove(move)

            # Build moves for all alive snakes
            moves_dict = {}
            for sid in node.state.alive:
                if sid == my_id:
                    moves_dict[sid] = move
                else:
                    opp_moves = node.state.legal_moves(sid)
                    if opp_moves:
                        moves_dict[sid] = random.choice(opp_moves)
                    # else snake can't move – stays still (dies if head‑on)

            next_state = node.state.apply_moves(moves_dict)
            child = MCTSNode(next_state, move=move, parent=node)
            node.children[move] = child

            # 3. Rollout
            score = _rollout(child.state, my_id, ROLLOUT_DEPTH)
            node = child
        else:
            # no untried moves and not terminal – evaluate directly
            score = node.state.score(my_id)

        # 4. Backpropagation
        while node is not None:
            node.visits += 1
            node.total_score += score
            node = node.parent

    # Pick move with most visits
    if not root.children:
        return random.choice(legal) if legal else "up"
    best_move = max(root.children.items(), key=lambda kv: kv[1].visits)[0]
    return best_move


def _rollout(state: GameState, my_id: str, depth: int) -> float:
    """Simulate random moves for all snakes up to depth or terminal."""
    current = state.copy()
    for _ in range(depth):
        if current.is_terminal(my_id):
            break
        moves = {}
        for sid in current.alive:
            lm = current.legal_moves(sid)
            if lm:
                moves[sid] = random.choice(lm)
        current = current.apply_moves(moves)
    return current.score(my_id)


# -------------------------------------------------------------------
#  Utility functions
# -------------------------------------------------------------------

def _occupied_cells(snakes: List[Dict]) -> Set[Point]:
    occupied = set()
    for snake in snakes:
        for seg in snake["body"]:
            occupied.add((seg["x"], seg["y"]))
    return occupied


def _head_to_head_cells(snakes: List[Dict], my_id: str, my_length: int) -> Set[Point]:
    danger = set()
    for snake in snakes:
        if snake["id"] == my_id:
            continue
        if snake["length"] < my_length:
            continue
        ehead = (snake["head"]["x"], snake["head"]["y"])
        for dx, dy in DIRECTIONS.values():
            danger.add((ehead[0] + dx, ehead[1] + dy))
    return danger


def _in_bounds(p: Point, width: int, height: int) -> bool:
    return 0 <= p[0] < width and 0 <= p[1] < height


def _manhattan(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
