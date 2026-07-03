"""
Ultra-fast heuristic Battlesnake – no MCTS, guaranteed <1 ms per move.
Focus: survival, food, centre, length‑aware danger avoidance.
"""

from collections import deque
from typing import Dict, List, Set, Tuple

Point = Tuple[int, int]

DIRECTIONS: Dict[str, Point] = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}

HUNGRY_THRESHOLD = 50
HEAD_TO_HEAD_PENALTY = 10_000


def get_info() -> Dict[str, str]:
    return {
        "apiversion": "1",
        "author": "fast_heuristic",
        "color": "#FFC0CB",
        "head": "silly",
        "tail": "round-bum",
        "version": "4.0.0",
    }


def choose_move(game_state: Dict) -> str:
    """Return best move using fast heuristic evaluation."""
    board = game_state["board"]
    you = game_state["you"]
    width = board["width"]
    height = board["height"]
    head = (you["head"]["x"], you["head"]["y"])
    my_length = you["length"]
    health = you["health"]
    my_id = you["id"]
    snakes = board["snakes"]
    foods = [(f["x"], f["y"]) for f in board["food"]]

    # --- Быстрый расчёт занятых клеток с учётом своего хвоста ---
    occupied = set()
    for s in snakes:
        for seg in s["body"]:
            occupied.add((seg["x"], seg["y"]))
    # Разрешаем пойти в свой хвост, если он освободится
    my_body = [(seg["x"], seg["y"]) for seg in you["body"]]
    if len(my_body) > 1:
        tail = my_body[-1]
        if tail in occupied:
            occupied.remove(tail)

    # --- Опасные клетки: соседи голов врагов, которые ≥ нас по длине ---
    danger = set()
    for s in snakes:
        if s["id"] == my_id or s["length"] < my_length:
            continue
        eh = (s["head"]["x"], s["head"]["y"])
        for dx, dy in DIRECTIONS.values():
            danger.add((eh[0] + dx, eh[1] + dy))

    # --- Вычисляем центр карты (притяжение к нему) ---
    center_x, center_y = (width - 1) / 2, (height - 1) / 2

    # --- Оценка каждого легального хода ---
    best_move = None
    best_score = -float("inf")

    for move, (dx, dy) in DIRECTIONS.items():
        nxt = (head[0] + dx, head[1] + dy)

        # Проверка границ и занятости
        if not (0 <= nxt[0] < width and 0 <= nxt[1] < height):
            continue
        if nxt in occupied:
            continue

        # 1. Пространство: быстрый BFS с ограничением (не больше длины тела + 3)
        space = _fast_flood_fill(nxt, snakes, my_id, width, height, my_length + 3)

        # 2. Еда: расстояние до ближайшей еды из этой клетки
        food_dist = min((_manhattan(nxt, f) for f in foods), default=999)
        # Голодный режим даёт больший вес еде
        food_score = (width + height - food_dist) * (3.0 if health < HUNGRY_THRESHOLD else 1.0)

        # 3. Притяжение к центру (чтобы не прижиматься к стенам)
        center_dist = abs(nxt[0] - center_x) + abs(nxt[1] - center_y)
        center_score = (width + height - center_dist) * 0.5

        # 4. Штраф за опасную клетку
        danger_penalty = HEAD_TO_HEAD_PENALTY if nxt in danger else 0

        # 5. Дополнительный бонус, если мы можем съесть еду прямо сейчас
        immediate_food = 15.0 if nxt in foods else 0.0

        total = space * 2.0 + food_score + center_score + immediate_food - danger_penalty

        if total > best_score:
            best_score = total
            best_move = move

    # Если вдруг все ходы заблокированы, смиряемся и идём вверх
    return best_move if best_move else "up"


# -------------------------------------------------------------------
#  Вспомогательные быстрые функции
# -------------------------------------------------------------------

def _fast_flood_fill(start: Point, snakes: List[Dict], my_id: str,
                     width: int, height: int, limit: int) -> int:
    """Число доступных клеток из start, считая хвосты врагов свободными."""
    # Строим множество препятствий: все сегменты змей, кроме хвостов.
    blocked = set()
    for s in snakes:
        body = [(seg["x"], seg["y"]) for seg in s["body"]]
        # все, кроме последнего (хвост может освободиться)
        for seg in body[:-1] if len(body) > 1 else body:
            blocked.add(seg)

    # BFS с простым стеком
    if start in blocked:
        return 0
    seen = {start}
    stack = [start]
    count = 0
    while stack and count < limit:
        x, y = stack.pop()
        count += 1
        for dx, dy in DIRECTIONS.values():
            nb = (x + dx, y + dy)
            if nb in seen or nb in blocked:
                continue
            if 0 <= nb[0] < width and 0 <= nb[1] < height:
                seen.add(nb)
                stack.append(nb)
    return count


def _manhattan(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
