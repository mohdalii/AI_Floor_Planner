from __future__ import annotations

import math
import torch

# NOTE: type 6 ("stair") is part of the geometry model's fixed room-type
# vocabulary (it was trained on real plans that sometimes include one),
# but the MVP generator (python/app/ml/predict.py) never instantiates a
# stair room and never asks the model to condition on stairs>0. Nothing
# in this file treats it specially any more - removing the entry here
# outright would just mean an unmapped id, so it's kept as a label only.
ROOM_NAMES = {
    0: "bedroom",
    1: "bathroom",
    2: "kitchen",
    3: "living",
    4: "balcony",
    5: "storage",
    6: "stair",
    7: "front_door",
}

# Minimum/maximum reasonable area per room type, as a fraction of total
# plot area. Used only to (a) repair pathologically-sized boxes and
# (b) estimate how much real-world space a room list needs - never to
# force every room to an exact size.
SIZE_RANGES = {
    0: (0.08, 0.24),   # bedroom
    1: (0.035, 0.10),  # bathroom
    2: (0.07, 0.18),   # kitchen
    3: (0.16, 0.42),   # living
    4: (0.025, 0.12),  # balcony
    5: (0.02, 0.08),   # storage
    6: (0.035, 0.12),  # stair (unused by the MVP generator - see above)
    7: (0.006, 0.035), # front door
}

PRIORITY = {
    3: 100,  # living
    2: 90,   # kitchen
    0: 80,   # bedroom
    1: 50,   # bathroom
    5: 35,   # storage
    4: 25,   # balcony
    7: 10,   # front door
}


# ------------------------------------------------------------
# BASIC GEOMETRY HELPERS
# ------------------------------------------------------------

def _clamp_box(box: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = box.unbind(-1)
    w = w.clamp(0.008, 0.80)
    h = h.clamp(0.008, 0.80)
    cx = cx.clamp(w / 2, 1.0 - w / 2)
    cy = cy.clamp(h / 2, 1.0 - h / 2)
    return torch.stack([cx, cy, w, h], dim=-1)


def _overlap(a, b):
    ax, ay, aw, ah = [float(v) for v in a]
    bx, by, bw, bh = [float(v) for v in b]
    ow = min(ax + aw / 2, bx + bw / 2) - max(ax - aw / 2, bx - bw / 2)
    oh = min(ay + ah / 2, by + bh / 2) - max(ay - ah / 2, by - bh / 2)
    return ow, oh


def _iou(a, b):
    ax, ay, aw, ah = [float(v) for v in a]
    bx, by, bw, bh = [float(v) for v in b]
    ax1, ay1, ax2, ay2 = ax-aw/2, ay-ah/2, ax+aw/2, ay+ah/2
    bx1, by1, bx2, by2 = bx-bw/2, by-bh/2, bx+bw/2, by+bh/2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = aw*ah + bw*bh - inter
    return inter / max(union, 1e-8)


def _priority(t):
    return PRIORITY.get(int(t), 50)


def _target_size(room_type, area):
    lo, hi = SIZE_RANGES.get(int(room_type), (0.02, 0.30))
    return min(max(float(area), lo), hi)


def _resize_box(box, room_type):
    """Only called for boxes flagged as pathological (near-zero or
    absurdly large area) - normally-sized predictions never reach here."""
    cx, cy, w, h = [float(v) for v in box]
    area = max(w * h, 1e-6)
    target = _target_size(room_type, area)

    aspect = max(0.55, min(w / max(h, 1e-6), 1.9))
    nw = math.sqrt(target * aspect)
    nh = target / max(nw, 1e-6)

    if int(room_type) == 7:  # door should stay small and shallow
        nw = min(max(nw, 0.018), 0.055)
        nh = min(max(nh, 0.012), 0.045)

    return _clamp_box(torch.tensor(
        [cx, cy, nw, nh], dtype=box.dtype, device=box.device
    ))


def _center(box):
    return float(box[0]), float(box[1])


def _distance(a, b):
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _room_area(boxes, i):
    return float(boxes[i, 2] * boxes[i, 3])


def _gap_to_room(boxes, i, j):
    """Boundary-to-boundary distance from room i to one specific room j
    (0 if touching or overlapping)."""
    ax, ay, aw, ah = [float(v) for v in boxes[i]]
    ax1, ay1, ax2, ay2 = ax - aw/2, ay - ah/2, ax + aw/2, ay + ah/2

    bx, by, bw, bh = [float(v) for v in boxes[j]]
    bx1, by1, bx2, by2 = bx - bw/2, by - bh/2, bx + bw/2, by + bh/2

    dx = max(ax1 - bx2, bx1 - ax2, 0.0)
    dy = max(ay1 - by2, by1 - ay2, 0.0)
    return math.hypot(dx, dy)


def _gap_to_nearest(boxes, i, ids):
    """Boundary-to-boundary distance to the closest other room."""
    best = None
    for j in ids:
        if j == i:
            continue
        gap = _gap_to_room(boxes, i, j)
        if best is None or gap < best:
            best = gap
    return best if best is not None else 0.0


# ------------------------------------------------------------
# RULE ENGINE: WHICH ROOM SHOULD ATTACH TO WHICH
# ------------------------------------------------------------

def _build_attach_map(boxes, room_types, ids):
    """Decide which specific room each bathroom / kitchen / storage /
    balcony / front door should attach to, following ordinary house
    conventions instead of just clustering toward whatever happens to
    be nearest (which could pair a bathroom with a balcony just
    because it was closer).

      - Bedrooms are ranked by size; the largest is treated as the
        master bedroom and gets first pick of bathroom/balcony.
      - Bathrooms pair one-to-one with bedrooms, largest bedroom
        first, until either runs out. Bathrooms left over (more baths
        than bedrooms) become shared/guest bathrooms attached to the
        living room. Bedrooms left over (more bedrooms than baths)
        have no dedicated en-suite and rely on being near the shared
        ones (a softer preference, handled in the relationship cost).
      - Kitchen attaches to the living room. (A separate dining room
        would take priority here, but the geometry model's room
        vocabulary doesn't include one yet, so "kitchen near the
        common hall" is the only supported case for this MVP.)
      - Storage attaches to the kitchen (service zone), falling back
        to the living room. (No staircase fallback - the MVP never
        generates one.)
      - The first balcony attaches to the master bedroom; any further
        balconies attach to the living room.
      - The front door always attaches to the living room / hall, so
        the entrance never opens directly into a bedroom.
    """
    ordered = sorted(ids)
    bedrooms = [i for i in ordered if int(room_types[i]) == 0]
    baths = [i for i in ordered if int(room_types[i]) == 1]
    kitchens = [i for i in ordered if int(room_types[i]) == 2]
    living = [i for i in ordered if int(room_types[i]) == 3]
    balconies = [i for i in ordered if int(room_types[i]) == 4]
    storages = [i for i in ordered if int(room_types[i]) == 5]
    doors = [i for i in ordered if int(room_types[i]) == 7]

    living_id = living[0] if living else None
    attach = {}

    bedrooms_by_size = sorted(bedrooms, key=lambda i: -_room_area(boxes, i))

    n_pairs = min(len(baths), len(bedrooms_by_size))
    for k in range(n_pairs):
        attach[baths[k]] = bedrooms_by_size[k]
    for k in range(n_pairs, len(baths)):
        if living_id is not None:
            attach[baths[k]] = living_id

    for ki in kitchens:
        if living_id is not None:
            attach[ki] = living_id

    service_target = kitchens[0] if kitchens else living_id
    for si in storages:
        if service_target is not None:
            attach[si] = service_target

    for k, bi in enumerate(balconies):
        if k == 0 and bedrooms_by_size:
            attach[bi] = bedrooms_by_size[0]
        elif living_id is not None:
            attach[bi] = living_id

    for di in doors:
        if living_id is not None:
            attach[di] = living_id

    return attach


# ------------------------------------------------------------
# TIERED COST: (hard violations, architectural relationships, geometry)
#
# Candidates are compared with plain tuple "<", which in Python is
# lexicographic: the hard-violation tier always dominates the
# relationship tier, which always dominates the geometry tier. A
# candidate that clears a collision always beats one that doesn't,
# no matter how much it would move a room. Among candidates that are
# equally valid on collisions, the one that best satisfies the rule
# engine wins. Only among candidates that are *both* collision-free
# *and* equally good architecturally does staying close to the raw
# V4 prediction break the tie. This directly replaces the previous
# single weighted-sum cost (and the flat +500 collision penalty hack
# it needed to approximate the same priority ordering).
# ------------------------------------------------------------

def _hard_violation_cost(boxes, i, ids):
    """Priority 1. 0.0 means this room doesn't overlap anything."""
    b = boxes[i]
    total = 0.0
    for j in ids:
        if j == i:
            continue
        ow, oh = _overlap(b, boxes[j])
        if ow > 0 and oh > 0:
            total += ow * oh
            total += 0.25 * min(
                ow / max(float(b[2]), 1e-6),
                oh / max(float(b[3]), 1e-6),
            )
    return total


def _relationship_cost(boxes, i, room_types, ids, attach_map, bedroom_ids, living_id):
    """Priority 2: how well this room satisfies the architectural rule
    engine - attach-target adjacency, exterior-wall placement for the
    door/balcony, and bedrooms clustering into one private zone away
    from the main living/circulation space."""
    t = int(room_types[i])
    cx, cy = _center(boxes[i])
    cost = 0.0

    target = attach_map.get(i)
    if target is not None:
        cost += _gap_to_room(boxes, i, target)
    elif t != 3:
        # No specific target (e.g. an unpaired bedroom) - still prefer
        # being near *something* rather than floating alone.
        cost += 0.4 * _gap_to_nearest(boxes, i, ids)

    edge = min(cx, cy, 1 - cx, 1 - cy)
    if t == 7:  # front door must stay on/near an exterior wall
        cost += 8.0 * max(edge - 0.05, 0.0)
    elif t == 4:  # balcony strongly prefers an exterior wall
        cost += 5.0 * max(edge - 0.12, 0.0)
    elif t in (1, 2, 3, 5):
        # interior rooms shouldn't unnecessarily hug the boundary
        cost += 1.2 * max(0.05 - edge, 0.0)

    if t == 0:
        others = [r for r in bedroom_ids if r != i]
        if others:
            ocx = sum(_center(boxes[r])[0] for r in others) / len(others)
            ocy = sum(_center(boxes[r])[1] for r in others) / len(others)
            # pull toward the other bedrooms (private zone cohesion)
            cost += 0.7 * _distance((cx, cy), (ocx, ocy))
        if living_id is not None:
            # push away from sitting right on top of the common area
            d = _distance((cx, cy), _center(boxes[living_id]))
            cost += 1.0 * max(0.22 - d, 0.0)

    return cost


def _geometry_cost(boxes, i, original):
    """Priority 3: distance moved from the raw V4 prediction."""
    b = boxes[i]
    dx = float(b[0] - original[i, 0])
    dy = float(b[1] - original[i, 1])
    return math.hypot(dx, dy)


def _total_priority_cost(boxes, room_types, original, ids, attach_map, bedroom_ids, living_id):
    hard = 0.0
    relationship = 0.0
    geometry = 0.0
    for i in ids:
        hard += _hard_violation_cost(boxes, i, ids)
        relationship += _relationship_cost(
            boxes, i, room_types, ids, attach_map, bedroom_ids, living_id
        )
        geometry += _geometry_cost(boxes, i, original)
    return (hard, relationship, geometry)


# ------------------------------------------------------------
# CANDIDATE SEARCH
# ------------------------------------------------------------

def _candidate_centers(box, room_type, living_center=None, attach_center=None):
    cx, cy, w, h = [float(v) for v in box]
    candidates = [(cx, cy)]

    # Local moves only, at increasing scale - correct just enough to
    # clear a collision, don't teleport to a generic template slot.
    for step in (0.03, 0.06, 0.10, 0.16, 0.24):
        for dx, dy in [
            (-step, 0), (step, 0), (0, -step), (0, step),
            (-step, -step), (-step, step), (step, -step), (step, step),
        ]:
            candidates.append((cx + dx, cy + dy))

    t = int(room_type)

    # If living exists, allow moving toward/away from it - still a
    # local, relative adjustment, not a fixed global slot.
    if living_center is not None and t != 3:
        lx, ly = living_center
        candidates += [
            (lx - 0.28, ly), (lx + 0.28, ly),
            (lx, ly - 0.28), (lx, ly + 0.28),
        ]

    # A room with a specific attachment target (an en-suite bathroom's
    # bedroom, a shared bathroom's or the door's living room) needs a
    # way to jump toward THAT room directly - local steps alone can get
    # stuck behind other rooms and never reach it.
    if attach_center is not None:
        tx, ty = attach_center
        for r in (0.12, 0.20, 0.30):
            candidates += [
                (tx - r, ty), (tx + r, ty),
                (tx, ty - r), (tx, ty + r),
            ]

    return candidates


def _place_one(boxes, i, room_types, original, ids, attach_map, bedroom_ids, living_id):
    current = boxes[i].clone()
    best = current.clone()
    best_cost = _total_priority_cost(
        boxes, room_types, original, ids, attach_map, bedroom_ids, living_id
    )

    t = int(room_types[i])
    target = attach_map.get(i)
    attach_center = _center(boxes[target]) if target is not None else None
    living_center = _center(boxes[living_id]) if living_id is not None else None

    for cx, cy in _candidate_centers(current, t, living_center, attach_center):
        trial = boxes.clone()
        trial[i, 0] = cx
        trial[i, 1] = cy
        trial[i] = _clamp_box(trial[i].unsqueeze(0))[0]

        cost = _total_priority_cost(
            trial, room_types, original, ids, attach_map, bedroom_ids, living_id
        )
        if cost < best_cost:
            best_cost = cost
            best = trial[i].clone()

    changed = not torch.allclose(best, current, atol=1e-6)
    boxes[i] = best
    return changed


def _enforce_exterior(boxes, room_types, ids, attach_map=None):
    """Snap balcony/front-door onto the exterior wall - toward whichever
    edge is nearest their attach target (e.g. the master bedroom a
    balcony belongs to), not just whichever edge the room happens to be
    sitting closest to right now. That was the previous failure mode:
    exterior snapping that ignored the room's actual relationship."""
    for i in ids:
        t = int(room_types[i])
        if t not in (4, 7):
            continue

        cx, cy, w, h = [float(v) for v in boxes[i]]

        ref_x, ref_y = cx, cy
        target = attach_map.get(i) if attach_map else None
        if target is not None:
            ref_x, ref_y = _center(boxes[target])

        candidates = [
            (cx, h/2 + 0.01),
            (cx, 1-h/2-0.01),
            (w/2 + 0.01, cy),
            (1-w/2-0.01, cy),
        ]
        x, y = min(candidates, key=lambda p: (p[0]-ref_x)**2 + (p[1]-ref_y)**2)
        boxes[i, 0] = x
        boxes[i, 1] = y
    return _clamp_box(boxes)


# ------------------------------------------------------------
# SINGLE PLAN
# ------------------------------------------------------------

def refine_single_layout(
    prediction,
    room_types,
    mask,
    iterations=70,
    margin=0.008,
):
    original = _clamp_box(prediction.detach().clone())
    boxes = original.clone()

    ids = [i for i in range(len(boxes)) if bool(mask[i])]
    if not ids:
        return boxes * 0

    # Only repair pathologically-sized boxes (near-zero or absurdly
    # large) - leave normally-sized model predictions untouched rather
    # than forcing every room into a hand-tuned architectural range.
    for i in ids:
        t = int(room_types[i])
        area = max(float(boxes[i, 2] * boxes[i, 3]), 1e-6)
        lo, hi = SIZE_RANGES.get(t, (0.02, 0.30))
        if area < lo * 0.3 or area > hi * 2.0:
            boxes[i] = _resize_box(boxes[i], t)

    bedroom_ids = [i for i in ids if int(room_types[i]) == 0]

    living_ids = [i for i in ids if int(room_types[i]) == 3]
    living_id = None
    if living_ids:
        living_id = max(living_ids, key=lambda i: float(boxes[i, 2] * boxes[i, 3]))

    # Rule engine: decide real house adjacency once (which bedroom, if
    # any, each bathroom is en-suite to; that the front door leads into
    # the living room; etc.) - without forcibly relocating anything yet.
    attach_map = _build_attach_map(boxes, room_types, ids)

    boxes = _enforce_exterior(boxes, room_types, ids, attach_map)

    # High-priority rooms settle first, then smaller/support rooms.
    ids.sort(
        key=lambda i: (
            -_priority(int(room_types[i])),
            -float(boxes[i, 2] * boxes[i, 3]),
        )
    )

    for _ in range(iterations):
        changed = False

        for i in ids:
            changed |= _place_one(
                boxes, i, room_types, original, ids, attach_map, bedroom_ids, living_id
            )

        boxes = _enforce_exterior(boxes, room_types, ids, attach_map)

        # Safety net: if two rooms still overlap after the priority
        # search (every candidate for the mover happened to lose on
        # tiers 2/3 despite still overlapping - rare, but possible),
        # force the smallest separating nudge. The hard-violation tier
        # in _total_priority_cost means this can never make collisions
        # worse, only better.
        for p in range(len(ids)):
            for q in range(p + 1, len(ids)):
                i, j = ids[p], ids[q]
                ow, oh = _overlap(boxes[i], boxes[j])
                if ow <= 1e-5 or oh <= 1e-5:
                    continue

                ti, tj = int(room_types[i]), int(room_types[j])
                mover = j if _priority(ti) >= _priority(tj) else i
                anchor = i if mover == j else j

                m = boxes[mover].clone()
                a = boxes[anchor]
                mx, my, mw, mh = [float(v) for v in m]
                ax, ay, aw, ah = [float(v) for v in a]

                candidates = [
                    (ax - aw/2 - mw/2 - margin, my),
                    (ax + aw/2 + mw/2 + margin, my),
                    (mx, ay - ah/2 - mh/2 - margin),
                    (mx, ay + ah/2 + mh/2 + margin),
                ]

                best = boxes[mover].clone()
                best_cost = _total_priority_cost(
                    boxes, room_types, original, ids, attach_map, bedroom_ids, living_id
                )

                for x, y in candidates:
                    trial = boxes.clone()
                    trial[mover, 0] = x
                    trial[mover, 1] = y
                    trial[mover] = _clamp_box(trial[mover].unsqueeze(0))[0]
                    c = _total_priority_cost(
                        trial, room_types, original, ids, attach_map, bedroom_ids, living_id
                    )
                    if c < best_cost:
                        best_cost = c
                        best = trial[mover].clone()

                if not torch.allclose(best, boxes[mover], atol=1e-6):
                    boxes[mover] = best
                    changed = True

        if not changed:
            break

    return boxes * (mask > 0).unsqueeze(-1).to(boxes.dtype)


# ------------------------------------------------------------
# BATCH API
# ------------------------------------------------------------

def refine_layout(
    prediction,
    room_types,
    mask,
    iterations=70,
    margin=0.008,
):
    return torch.stack([
        refine_single_layout(
            prediction[b],
            room_types[b],
            mask[b],
            iterations,
            margin,
        )
        for b in range(prediction.shape[0])
    ], dim=0)


# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------

def collision_rate(prediction, mask):
    total = bad = 0
    for b in range(prediction.shape[0]):
        ids = [
            i for i in range(prediction.shape[1])
            if bool(mask[b, i])
        ]
        for p in range(len(ids)):
            for q in range(p + 1, len(ids)):
                total += 1
                ow, oh = _overlap(
                    prediction[b, ids[p]],
                    prediction[b, ids[q]],
                )
                bad += int(ow > 1e-5 and oh > 1e-5)
    return bad / max(total, 1)


def boundary_violation_rate(prediction, mask):
    total = bad = 0
    for b in range(prediction.shape[0]):
        ids = [
            i for i in range(prediction.shape[1])
            if bool(mask[b, i])
        ]
        for i in ids:
            total += 1
            cx, cy, w, h = [float(v) for v in prediction[b, i]]
            bad += int(
                cx - w/2 < -1e-5
                or cy - h/2 < -1e-5
                or cx + w/2 > 1.00001
                or cy + h/2 > 1.00001
            )
    return bad / max(total, 1)


if __name__ == "__main__":
    print("Geometry Layout Solver V3 ready (priority-tiered, no staircase objective).")
