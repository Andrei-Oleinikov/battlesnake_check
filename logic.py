"""
Adaptive strategy Battlesnake:
- Multi-enemy: safe food gathering + center tendency
- 1v1 (duel): aggressive space denial and cutting off
Fast flood-fill evaluation, <1ms per move.
"""

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
DUEL_SPACE_WEIGHT = 10.0
FOOD_WEIGHT_MULTI = 3.0


def get_info() -> Dict[str, str]:
    return {
        "apiversion": "1",
        "author": "adaptive_duelist",
        "color": "#FFA500",
        "head": "silly",
        "tail": "round-bum",
        "version": "5.0.0",
    }


def choose_move(game_state: Dict) -> str:
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

    # --- Занятые клетки (кроме нашего хвоста) ---
    occupied = set()
    for s in snakes:
        for seg in s["body"]:
            occupied.add((seg["x"], seg["y"]))
    my_body = [(seg["x"], seg["y"]) for seg in you["body"]]
    if len(my_body) > 1 and my_body[-1] in occupied:
        occupied.remove(my_body[-1])

    # --- Опасные клетки (рядом с большими головами) ---
    danger = set()
    enemies = [s for s in snakes if s["id"] != my_id]
    for s in enemies:
        if s["length"] >= my_length:
            eh = (s["head"]["x"], s["head"]["y"])
            for dx, dy in DIRECTIONS.values():
                danger.add((eh[0] + dx, eh[1] + dy))

    # Определяем режим
    if len(enemies) == 1:
        # Дуэль
        enemy = enemies[0]
        enemy_head = (enemy["head"]["x"], enemy["head"]["y"])
        enemy_length = enemy["length"]
        best_move = _duel_move(head, my_length, health, enemy_head, enemy_length,
                               snakes, my_id, width, height, occupied, danger)
    else:
        # Мультирежим
        best_move = _multi_move(head, my_length, health, foods, enemies,
                                width, height, occupied, danger, my_id, snakes)

    # Финальная защита
    if best_move is None:
        for move in DIRECTIONS:
            if _is_legal(move, head, occupied, width, height):
                return move
        return "up"
    return best_move


# -------------------------------------------------------------------
#  Режим нескольких врагов: сбор еды и безопасность
# -------------------------------------------------------------------

def _multi_move(head, my_length, health, foods, enemies,
                width, height, occupied, danger, my_id, snakes):
    center_x, center_y = (width - 1) / 2, (height - 1) / 2
    best_move = None
    best_score = -float("inf")

    for move, (dx, dy) in DIRECTIONS.items():
        nxt = (head[0] + dx, head[1] + dy)
        if not _is_legal(move, head, occupied, width, height):
            continue

        # Пространство (ограниченный flood fill)
        space = _flood_fill(nxt, snakes, my_id, width, height, my_length + 5)

        # Еда
        if foods:
            nearest_food = min(_manhattan(nxt, f) for f in foods)
            food_score = (width + height - nearest_food) * (
                FOOD_WEIGHT_MULTI if health < HUNGRY_THRESHOLD else 1.2)
        else:
            food_score = 0

        # Центр
        center_dist = abs(nxt[0] - center_x) + abs(nxt[1] - center_y)
        center_score = (width + height - center_dist) * 0.4

        # Опасность
        danger_penalty = HEAD_TO_HEAD_PENALTY if nxt in danger else 0

        # Мгновенная еда
        instant_food = 20.0 if nxt in foods else 0.0

        total = space * 2.5 + food_score + center_score + instant_food - danger_penalty
        if total > best_score:
            best_score = total
            best_move = move

    return best_move


# -------------------------------------------------------------------
#  Режим дуэли: пространственный контроль и подрезание
# -------------------------------------------------------------------

def _duel_move(head, my_length, health, enemy_head, enemy_length,
               snakes, my_id, width, height, occupied, danger):
    """Стратегия 1 на 1: максимизируем разницу доступного пространства,
    при возможности атакуем или уходим в зависимости от длины."""
    best_move = None
    best_score = -float("inf")

    # Предварительно строим карту препятствий без хвостов
    base_blocked = _build_block_set(snakes, my_id)

    for move, (dx, dy) in DIRECTIONS.items():
        nxt = (head[0] + dx, head[1] + dy)
        if not _is_legal(move, head, occupied, width, height):
            continue

        # 1. Разница пространства после нашего хода (наше - вражеское)
        # Симулируем временное занятие nxt
        temp_blocked = set(base_blocked)
        temp_blocked.add(nxt)
        # Пространство для нас из nxt (исключая вражескую голову и тело, но враг тоже может двигаться – упрощаем)
        our_space = _flood_fill_custom(nxt, temp_blocked, width, height, 20)
        # Пространство для врага из его головы (блокируем нас и его собственное тело без хвоста)
        enemy_blocked = _build_block_set(snakes, my_id)  # без наших клеток, кроме тела
        # Добавляем нашу новую клетку как препятствие
        enemy_blocked.add(nxt)
        enemy_space = _flood_fill_custom(enemy_head, enemy_blocked, width, height, 20)
        space_diff = our_space - enemy_space

        # 2. Расстояние до врага: если мы длиннее, хотим быть ближе (атака); если короче – дальше
        dist_to_enemy = _manhattan(nxt, enemy_head)
        if my_length > enemy_length:
            # Агрессия: предпочитаем уменьшать расстояние
            aggression_score = (width + height - dist_to_enemy) * 2.0
        elif my_length < enemy_length:
            # Отступление: увеличиваем дистанцию
            aggression_score = dist_to_enemy * 2.0
        else:
            # Равная длина – нейтрально
            aggression_score = 0

        # 3. Безопасность: избегаем прямого столкновения с большим врагом
        danger_penalty = HEAD_TO_HEAD_PENALTY if nxt in danger else 0

        # 4. Еда всё ещё важна, если здоровье низкое
        food_bonus = 0
        if health < HUNGRY_THRESHOLD:
            # Быстрый поиск ближайшей еды для этого хода
            foods = [(f["x"], f["y"]) for f in board["food"]]
            if foods:
                nearest = min(_manhattan(nxt, f) for f in foods)
                food_bonus = (width + height - nearest) * 5.0

        total = (space_diff * DUEL_SPACE_WEIGHT
                 + aggression_score
                 + food_bonus
                 - danger_penalty)
        if total > best_score:
            best_score = total
            best_move = move

    return best_move


# -------------------------------------------------------------------
#  Вспомогательные функции
# -------------------------------------------------------------------

def _is_legal(move: str, head: Point, occupied: Set[Point], w: int, h: int) -> bool:
    if move not in DIRECTIONS:
        return False
    dx, dy = DIRECTIONS[move]
    nxt = (head[0] + dx, head[1] + dy)
    return 0 <= nxt[0] < w and 0 <= nxt[1] < h and nxt not in occupied


def _manhattan(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _build_block_set(snakes, my_id):
    """Строит множество препятствий: все сегменты змей, кроме хвостов (они могут освободиться)."""
    blocked = set()
    for s in snakes:
        body = [(seg["x"], seg["y"]) for seg in s["body"]]
        if s["id"] == my_id:
            # Своё тело без хвоста
            if len(body) > 1:
                blocked.update(body[:-1])
        else:
            # Вражеское тело полностью (консервативно) – кроме хвоста, но для простоты считаем хвост опасным
            blocked.update(body)
    return blocked


def _flood_fill(start: Point, snakes: List[Dict], my_id: str,
                width: int, height: int, limit: int) -> int:
    """Быстрый flood fill, считая хвосты свободными."""
    blocked = _build_block_set(snakes, my_id)
    return _flood_fill_custom(start, blocked, width, height, limit)


def _flood_fill_custom(start: Point, blocked: Set[Point],
                       width: int, height: int, limit: int) -> int:
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
