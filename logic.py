"""
MCTS-based move-selection for Battlesnake with heuristic fallback.

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
ROLLOUT_DEPTH = 15         # max rollout depth in MCTS
DEATH_SCORE = -1000.0      # score when our snake dies


def get_info() -> Dict[str, str]:
    return {
        "apiversion": "1",
        "author": "hackathon_mcts",
        "color": "#6434eb",
        "head": "smart-caterpillar",
        "tail": "weight",
        "version": "2.0.0",
    }


# -------------------------------------------------------------------
#  Main entry point
# -------------------------------------------------------------------

def choose_move(game_state: Dict) -> str:
    """Choose move with MCTS; fall back to enhanced heuristic if low on time."""
    start = time.time()
    # fast fail-safe: if almost no time left, skip MCTS
    if time.time() - start > 0.4:   # unlikely at first call
        return choose_move_heuristic_enhanced(game_state)
    try:
        return mcts_move(game_state, MCTS_TIME_LIMIT_MS)
    except Exception:
        return choose_move_heuristic_enhanced(game_state)


# -------------------------------------------------------------------
#  Enhanced heuristic (fallback) – smarter version of your original
# -------------------------------------------------------------------

def choose_move_heuristic_enhanced(game_state: Dict) -> str:
    """Improved heuristic with tail-awareness and better flood-fill."""
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

        # Reachable space from nxt, taking into account that enemy tails might free
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
    my_body = [(seg["x"], seg["y"]) for seg in you["body"]]
    my_tail = my_body[-1] if my_body else None
    for snake in snakes:
        for seg in snake["body"]:
            occupied.add((seg["x"], seg["y"]))
    # Remove our tail if we are not eating (conservative: we don't know if we eat this turn,
    # but when evaluating a move we already know if nxt is food – handled in MCTS, not here).
    # For heuristic we assume we will NOT eat (so tail frees up) to open more options.
    if my_tail and my_tail in occupied:
        occupied.remove(my_tail)
    return occupied


def _flood_fill_loose(start: Point, snakes: List[Dict], my_id: str,
                      width: int, height: int, limit: int) -> int:
    """Count reachable cells from start, assuming enemy tails may free up."""
    # Build initial occupied set: all snake bodies, but exclude tail of any snake
    # (they might move). For safety we keep heads and all other segments.
    occupied = set()
    tails = set()
    for snake in snakes:
        body = [(seg["x"], seg["y"]) for seg in snake["body"]]
        for i, seg in enumerate(body):
            if i == len(body) - 1:
                tails.add(seg)   # tail might disappear
            else:
                occupied.add(seg)
    # Our start must not be in occupied (already checked in caller)
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
            # Allow moving into a tail cell (might become free)
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
        # deep copy everything needed
        new_snakes = {}
        for sid, s in self.snakes.items():
            new_snakes[sid] = {
                'body': s['body'][:],   # copy list of tuples
                'health': s['health']
            }
        return GameState(self.width, self.height, new_snakes,
                         self.food.copy(), self.alive.copy())

    def legal_moves(self, snake_id: str) -> List[str]:
        """All directions that don't immediately hit wall or occupied body
        (excluding own tail if not growing)."""
        s = self.snakes[snake_id]
        head = s['body'][0]
        body_set = set(s['body'])
        # Determine if tail will be freed (not growing). We can't know for sure,
        # so we allow moving into own tail if we're not eating now.
        # In the move generation phase we don't yet know if we'll eat.
        # We'll conservatively allow moving into own tail always, but disallow
        # if it would cause a collision with another snake's body.
        own_tail = s['body'][-1]
        moves = []
        for move, (dx, dy) in DIRECTIONS.items():
            nxt = (head[0] + dx, head[1] + dy)
            if not (0 <= nxt[0] < self.width and 0 <= nxt[1] < self.height):
                continue
            # Check collision with all snake bodies (including ours)
            collision = False
            for sid2, s2 in self.snakes.items():
                # For our own snake, exclude tail if it's ours (allowed)
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
        moves: dict snake_id -> direction string."""
        state = self.copy()

        # 1. Determine new heads
        new_heads = {}
        for sid, move in moves.items():
            if sid not in state.alive:
                continue
            s = state.snakes[sid]
            dx, dy = DIRECTIONS[move]
            hx, hy = s['body'][0]
            new_heads[sid] = (hx + dx, hy + dy)

        # 2. Resolve collisions and eating
        # Multiple snakes can collide in a cell; equal-length head-to-head kills both.
        # Also check wall/body (already ensured by legal_moves, but re-check for safety).
        eaten_food = set()
        dead = set()

        # Collision map: cell -> list of snake ids that moved there
        cell_to_snakes: Dict[Point, List[str]] = {}
        for sid, nh in new_heads.items():
            cell_to_snakes.setdefault(nh, []).append(sid)

        for cell, sids in cell_to_snakes.items():
            if len(sids) > 1:
                # Head-to-head collision between those snakes
                # Sort by length descending? Rules: if all same length, all die;
                # otherwise only the longest survive? Actually Battlesnake rules:
                # If multiple snakes enter the same cell, all die. Wait:
                # Official: "If two snakes move into the same cell, both are eliminated
                # (unless one is longer, in which case the shorter dies)."
                # Let's implement: keep only the longest, if unique; else all die.
                lengths = {sid: len(state.snakes[sid]['body']) for sid in sids}
                max_len = max(lengths.values())
                # count how many have max_len
                max_count = sum(1 for l in lengths.values() if l == max_len)
                if max_count == 1:
                    # only one longest survives
                    survivor = [sid for sid, l in lengths.items() if l == max_len][0]
                    for sid in sids:
                        if sid != survivor:
                            dead.add(sid)
                else:
                    # all tied for longest die
                    for sid in sids:
                        dead.add(sid)
            else:
                sid = sids[0]
                # Single snake moving here: check if cell contains body of any snake
                # that didn't free up. Bodies after movement are old bodies minus tails.
                # We'll handle bodies later; for now, just check food.
                if cell in state.food:
                    eaten_food.add(cell)

        # 3. Update health and bodies
        for sid in state.alive:
            if sid in dead:
                continue
            s = state.snakes[sid]
            # Decrease health
            s['health'] -= 1
            if s['health'] <= 0:
                dead.add(sid)
                continue

            # Move body
            move = moves.get(sid)
            if move is None:   # didn't move (shouldn't happen)
                continue
            new_head = new_heads[sid]
            body = s['body']
            # Insert new head
            body.insert(0, new_head)
            # Remove tail if not eating
            if new_head not in eaten_food:
                body.pop()
            else:
                # Ate food: reset health to 100 (max health)
                s['health'] = 100
                # food removed below

        # 4. Remove eaten food
        state.food -= eaten_food

        # 5. Remove dead snakes' bodies from future collisions?
        # They disappear immediately. So remove them from state.
        for sid in dead:
            state.alive.discard(sid)
            if sid in state.snakes:
                del state.snakes[sid]

        return state

    def is_terminal(self, my_id: str) -> bool:
        """Terminal if we are dead or no enemies alive."""
        return my_id not in self.alive or len(self.alive) == 1

    def score(self, my_id: str) -> float:
        """Heuristic evaluation from our perspective (non-terminal)."""
        if my_id not in self.alive:
            return DEATH_SCORE
        me = self.snakes[my_id]
        head = me['body'][0]
        health = me['health']
        my_len = len(me['body'])

        # Flood fill space accessible to us (with tail awareness)
        space = self._flood_fill_self(head, my_id)

        # Distance to nearest food
        food_dist = float('inf')
        if self.food:
            food_dist = min(_manhattan(head, f) for f in self.food)

        # Enemy metrics
        enemy_lengths = [len(s['body']) for sid, s in self.snakes.items() if sid != my_id]
        max_enemy_len = max(enemy_lengths) if enemy_lengths else 0
        enemy_heads = [s['body'][0] for sid, s in self.snakes.items() if sid != my_id]
        danger_near = 0.0
        for ehead in enemy_heads:
            d = _manhattan(head, ehead)
            if d == 1:  # adjacent
                enemy_len = len(self.snakes[[sid for sid in self.snakes if sid != my_id][0]]['body'])
                if enemy_len >= my_len:
                    danger_near += 10.0

        # Composite score (weights tuned empirically)
        score = 0.0
        score += space * 2.0             # open space is king
        score += health * 0.5            # survival
        score -= min(food_dist, 20) * 1.0  # prefer closer food
        score -= danger_near             # avoid risky heads
        # Bonus for being longer than all
        if my_len > max_enemy_len:
            score += 20.0

        return score

    def _flood_fill_self(self, start: Point, my_id: str) -> int:
        """Flood fill from start, considering that our tail will free up
        and enemy tails might free. Conservative: treat only our tail as sure to free."""
        # Occupied cells: all snake bodies except our tail.
        occupied = set()
        for sid, s in self.snakes.items():
            body = s['body']
            if sid == my_id and len(body) > 1:
                # exclude our tail
                occupied.update(body[:-1])
            else:
                occupied.update(body)
        seen = {start}
        stack = [start]
        count = 0
        while stack and count < 200:  # cap
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
    __slots__ = ('state', 'move', 'parent', 'children', 'visits', 'total_score', 'untried_moves')

    def __init__(self, state: GameState, move: Optional[str] = None, parent: Optional['MCTSNode'] = None):
        self.state = state
        self.move = move          # move that led to this state from parent
        self.parent = parent
        self.children: Dict[str, MCTSNode] = {}
        self.visits = 0
        self.total_score = 0.0
        my_id = list(state.alive)[0] if state.alive else ""   # placeholder, real my_id passed later
        # We'll initialise untried_moves properly in search
        self.untried_moves: List[str] = []

    def ucb1(self, C: float) -> float:
        if self.visits == 0:
            return float('inf')
        return self.total_score / self.visits + C * (__import__('math').sqrt(__import__('math').log(self.parent.visits) / self.visits))


def mcts_move(game_state: Dict, time_limit_ms: float) -> str:
    """Run MCTS from the current game state and return best move."""
    start_time = time.time()
    root_state = state_from_game(game_state)
    my_id = game_state["you"]["id"]

    root = MCTSNode(root_state)
    # legal moves for us
    legal = root_state.legal_moves(my_id)
    if not legal:
        return "up"
    root.untried_moves = legal[:]

    # Main MCTS loop
    while (time.time() - start_time) * 1000 < time_limit_ms:
        node = root
        state = root_state.copy()

        # 1. Selection
        while node.untried_moves == [] and node.children:
            # choose child with max UCB1
            best = max(node.children.values(), key=lambda c: c.ucb1(MCTS_C))
            node = best
            # apply move to state: need to simulate all snakes' moves.
            # But our tree only branches on our moves; opponent moves were sampled.
            # We stored the state in the child node, so we can just use child.state.
            # However, for consistent selection we must update state to child.state.
            # Actually, to keep state accurate along the path, we can set state = node.state.
            # That works because each node already holds the exact state after the move.
            # So just state = node.state (already done by construction).
            pass

        # Now node is a leaf in the tree (might be terminal or have untried moves)
        if node.state.is_terminal(my_id) or not node.untried_moves:
            # If terminal or no moves, evaluate directly (no rollout needed)
            score = node.state.score(my_id)
        else:
            # 2. Expansion: pick a random untried move
            move = random.choice(node.untried_moves)
            node.untried_moves.remove(move)

            # Simulate opponent moves for this step (random)
            # Build moves dict: our move + random for others
            moves_dict = {}
            for sid in node.state.alive:
                if sid == my_id:
                    moves_dict[sid] = move
                else:
                    opp_moves = node.state.legal_moves(sid)
                    if opp_moves:
                        moves_dict[sid] = random.choice(opp_moves)
                    # else snake can't move – dies? We'll handle by not adding,
                    # apply_moves will just skip, snake remains and may collide.

            # Create new state after this full step
            next_state = node.state.apply_moves(moves_dict)
            child = MCTSNode(next_state, move=move, parent=node)
            node.children[move] = child

            # 3. Rollout from child
            score = _rollout(child.state, my_id, ROLLOUT_DEPTH)
            node = child

        # 4. Backpropagation
        while node is not None:
            node.visits += 1
            node.total_score += score
            node = node.parent

    # After time is up, pick move with most visits from root
    best_move = max(root.children.items(), key=lambda kv: kv[1].visits)[0]
    return best_move


def _rollout(state: GameState, my_id: str, depth: int) -> float:
    """Simulate random moves for all snakes up to depth or terminal state."""
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
#  Utility functions (kept from original)
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
