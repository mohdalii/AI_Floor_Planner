from __future__ import annotations

import math
import torch

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

# Soft architectural priors.  These are deliberately broad so the solver
# corrects obviously bad V3 boxes without trying to reproduce one exact plan.
SIZE_RANGES = {
    0: (0.08, 0.24),   # bedroom
    1: (0.035, 0.10),  # bathroom
    2: (0.07, 0.18),   # kitchen
    3: (0.16, 0.42),   # living
    4: (0.025, 0.12),  # balcony
    5: (0.02, 0.08),   # storage
    6: (0.035, 0.12),  # stair
    7: (0.006, 0.035), # front door
}

PRIORITY = {
    3: 100,  # living
    2: 90,   # kitchen
    0: 80,   # bedroom
    6: 75,   # stair
    1: 50,   # bathroom
    5: 35,   # storage
    4: 25,   # balcony
    7: 10,   # front door
}


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
    target_area = min(max(float(area), lo), hi)
    return target_area


def _resize_box(box, room_type):
    cx, cy, w, h = [float(v) for v in box]
    area = max(w * h, 1e-6)
    target = _target_size(room_type, area)

    # Preserve the predicted aspect ratio, but prevent pathological skinny boxes.
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


def _boundary_distance(box):
    cx, cy, w, h = [float(v) for v in box]
    return min(cx - w/2, cy - h/2, 1-cx-w/2, 1-cy-h/2)


def _gap_to_nearest(boxes, i, ids):
    """Boundary-to-boundary distance to the closest other room (0 if
    touching or overlapping) - a proxy for "does this look connected"."""
    ax, ay, aw, ah = [float(v) for v in boxes[i]]
    ax1, ay1, ax2, ay2 = ax - aw/2, ay - ah/2, ax + aw/2, ay + ah/2

    best = None
    for j in ids:
        if j == i:
            continue
        bx, by, bw, bh = [float(v) for v in boxes[j]]
        bx1, by1, bx2, by2 = bx - bw/2, by - bh/2, bx + bw/2, by + bh/2

        dx = max(ax1 - bx2, bx1 - ax2, 0.0)
        dy = max(ay1 - by2, by1 - ay2, 0.0)
        gap = math.hypot(dx, dy)

        if best is None or gap < best:
            best = gap

    return best if best is not None else 0.0


def _gap_to_room(boxes, i, j):
    """Boundary-to-boundary distance from room i to one specific room j."""
    ax, ay, aw, ah = [float(v) for v in boxes[i]]
    ax1, ay1, ax2, ay2 = ax - aw/2, ay - ah/2, ax + aw/2, ay + ah/2

    bx, by, bw, bh = [float(v) for v in boxes[j]]
    bx1, by1, bx2, by2 = bx - bw/2, by - bh/2, bx + bw/2, by + bh/2

    dx = max(ax1 - bx2, bx1 - ax2, 0.0)
    dy = max(ay1 - by2, by1 - ay2, 0.0)
    return math.hypot(dx, dy)


def _build_attach_map(room_types, ids):
    """Decide which specific room each bathroom/front door should attach
    to, so the layout follows real house conventions instead of just
    clustering toward whatever happens to be nearest:

      - front door always attaches to the living room / entrance hall.
      - if there are at least as many bathrooms as bedrooms, each
        bedroom gets its own en-suite bathroom (paired in order).
      - otherwise bathrooms are shared, so they attach to the living
        room / hall for central access instead of any one bedroom.
    """
    ordered = sorted(ids)
    bedrooms = [i for i in ordered if int(room_types[i]) == 0]
    baths = [i for i in ordered if int(room_types[i]) == 1]
    doors = [i for i in ordered if int(room_types[i]) == 7]
    living = [i for i in ordered if int(room_types[i]) == 3]

    living_id = living[0] if living else None
    attach = {}

    ensuite = bool(bedrooms) and len(baths) >= len(bedrooms)
    for k, bi in enumerate(baths):
        if ensuite and k < len(bedrooms):
            attach[bi] = bedrooms[k]
        elif living_id is not None:
            attach[bi] = living_id

    for di in doors:
        if living_id is not None:
            attach[di] = living_id

    return attach


def _room_cost(boxes, i, room_types, original, ids, displacement_weight=1.8,
                gap_weight=0.0, attach_map=None, attach_weight=0.0):
    t = int(room_types[i])
    b = boxes[i]
    cost = 0.0

    # Keep the final layout reasonably close to the neural prediction.
    dx = float(b[0] - original[i, 0])
    dy = float(b[1] - original[i, 1])
    cost += displacement_weight * (dx*dx + dy*dy)

    # Penalize excessive resizing, but allow strong correction of pathological boxes.
    oa = max(float(original[i, 2] * original[i, 3]), 1e-6)
    aa = max(float(b[2] * b[3]), 1e-6)
    cost += 0.8 * abs(math.log(aa / oa))

    # Room-size prior.
    lo, hi = SIZE_RANGES.get(t, (0.02, 0.30))
    if aa < lo:
        cost += 8.0 * (lo - aa) ** 2 / max(lo*lo, 1e-6)
    elif aa > hi:
        cost += 8.0 * (aa - hi) ** 2 / max(hi*hi, 1e-6)

    cx, cy = _center(b)

    # Exterior rooms belong near an exterior edge.
    if t == 4:
        edge = min(cx, cy, 1-cx, 1-cy)
        cost += 18.0 * max(edge - 0.10, 0.0) ** 2
    elif t == 7:
        edge = min(cx, cy, 1-cx, 1-cy)
        cost += 35.0 * max(edge - 0.045, 0.0) ** 2

    # Interior rooms should not hug the exterior boundary.
    elif t in (1, 2, 3, 5, 6):
        edge = min(cx, cy, 1-cx, 1-cy)
        cost += 2.0 * max(0.06 - edge, 0.0) ** 2

    # Penalize overlaps strongly - the flat term means the search will
    # never trade a gap-closing win for even a sliver of new overlap.
    for j in ids:
        if j == i:
            continue
        ow, oh = _overlap(b, boxes[j])
        if ow > 0 and oh > 0:
            cost += 500.0
            cost += 160.0 * ow * oh
            # Stronger penalty for large interpenetration.
            cost += 35.0 * min(ow / max(float(b[2]), 1e-6),
                                oh / max(float(b[3]), 1e-6))

    # Gap-closing. Rooms with a specific attachment target (an en-suite
    # bathroom's bedroom, a shared bathroom's or the front door's living
    # room) are pulled toward exactly that room, not just whatever's
    # closest - otherwise a bathroom could end up snapped to a balcony
    # instead of its bedroom just because it happened to be nearer.
    # Everything else still closes to its nearest neighbor.
    target = attach_map.get(i) if attach_map else None
    if target is not None and attach_weight:
        cost += attach_weight * _gap_to_room(boxes, i, target)
    elif gap_weight:
        cost += gap_weight * _gap_to_nearest(boxes, i, ids)

    return cost


def _relationship_cost(boxes, room_types, ids):
    living = [i for i in ids if int(room_types[i]) == 3]
    kitchen = [i for i in ids if int(room_types[i]) == 2]
    baths = [i for i in ids if int(room_types[i]) == 1]
    bedrooms = [i for i in ids if int(room_types[i]) == 0]
    balconies = [i for i in ids if int(room_types[i]) == 4]
    doors = [i for i in ids if int(room_types[i]) == 7]

    cost = 0.0

    if living:
        lc = _center(boxes[living[0]])
        for i in kitchen:
            d = _distance(_center(boxes[i]), lc)
            cost += 1.8 * max(d - 0.42, 0.0) ** 2
        for i in baths:
            d = _distance(_center(boxes[i]), lc)
            cost += 0.7 * max(d - 0.62, 0.0) ** 2

    if kitchen and living:
        for i in kitchen:
            d = _distance(_center(boxes[i]), _center(boxes[living[0]]))
            cost += 2.2 * max(d - 0.35, 0.0) ** 2

    if baths and bedrooms:
        for b in baths:
            d = min(_distance(_center(boxes[b]), _center(boxes[r]))
                    for r in bedrooms)
            cost += 1.4 * max(d - 0.45, 0.0) ** 2

    # Encourage bedrooms to occupy different regions.
    for p in range(len(bedrooms)):
        for q in range(p+1, len(bedrooms)):
            d = _distance(_center(boxes[bedrooms[p]]), _center(boxes[bedrooms[q]]))
            cost += 0.8 * max(0.16 - d, 0.0) ** 2

    if living and balconies:
        lc = _center(boxes[living[0]])
        for i in balconies:
            # Balcony should be reasonably accessible from the main living zone.
            cost += 0.6 * max(_distance(_center(boxes[i]), lc) - 0.72, 0.0) ** 2

    if doors:
        # Entrance should be exterior and reasonably close to living.
        if living:
            lc = _center(boxes[living[0]])
            for d in doors:
                dc = _center(boxes[d])
                cost += 0.9 * max(_distance(dc, lc) - 0.80, 0.0) ** 2

    return cost


def _total_cost(boxes, room_types, original, ids, displacement_weight=1.8,
                 gap_weight=0.0, attach_map=None, attach_weight=0.0):
    value = sum(
        _room_cost(boxes, i, room_types, original, ids, displacement_weight,
                   gap_weight, attach_map, attach_weight)
        for i in ids
    )
    value += 6.0 * _relationship_cost(boxes, room_types, ids)
    return value


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


def _place_one(boxes, i, room_types, original, ids, living_center,
                displacement_weight=1.8, gap_weight=0.0,
                attach_map=None, attach_weight=0.0):
    current = boxes[i].clone()
    best = current.clone()
    best_cost = _total_cost(boxes, room_types, original, ids, displacement_weight,
                             gap_weight, attach_map, attach_weight)

    t = int(room_types[i])
    target = attach_map.get(i) if attach_map else None
    attach_center = _center(boxes[target]) if target is not None else None

    for cx, cy in _candidate_centers(current, t, living_center, attach_center):
        trial = boxes.clone()
        trial[i, 0] = cx
        trial[i, 1] = cy
        trial[i] = _clamp_box(trial[i].unsqueeze(0))[0]

        cost = _total_cost(trial, room_types, original, ids, displacement_weight,
                            gap_weight, attach_map, attach_weight)
        if cost + 1e-7 < best_cost:
            best_cost = cost
            best = trial[i].clone()

    changed = not torch.allclose(best, current, atol=1e-6)
    boxes[i] = best
    return changed


def _enforce_exterior(boxes, room_types, ids):
    for i in ids:
        t = int(room_types[i])
        if t not in (4, 7):
            continue

        cx, cy, w, h = [float(v) for v in boxes[i]]
        candidates = [
            (cx, h/2 + 0.01),
            (cx, 1-h/2-0.01),
            (w/2 + 0.01, cy),
            (1-w/2-0.01, cy),
        ]
        # Pick the exterior position closest to the current center.
        cx0, cy0 = cx, cy
        x, y = min(candidates, key=lambda p: (p[0]-cx0)**2 + (p[1]-cy0)**2)
        boxes[i, 0] = x
        boxes[i, 1] = y
    return _clamp_box(boxes)


def refine_single_layout(
    prediction,
    room_types,
    mask,
    iterations=60,
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

    # Establish a living anchor (if one exists) for relative placement
    # of other rooms - without forcibly relocating it.
    living_ids = [i for i in ids if int(room_types[i]) == 3]
    living_center = None
    if living_ids:
        li = max(living_ids, key=lambda i: float(boxes[i, 2] * boxes[i, 3]))
        living_center = _center(boxes[li])

    # Decide real house adjacency once: which bedroom (if any) each
    # bathroom is en-suite to, and that the front door leads into the
    # living room / entrance hall.
    attach_map = _build_attach_map(room_types, ids)

    boxes = _enforce_exterior(boxes, room_types, ids)

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
                boxes, i, room_types, original, ids, living_center,
                attach_map=attach_map, attach_weight=20.0,
            )

        boxes = _enforce_exterior(boxes, room_types, ids)

        # Final pairwise separation pass.
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

                old_cost = _total_cost(boxes, room_types, original, ids)
                best = boxes[mover].clone()
                best_cost = old_cost

                for x, y in candidates:
                    trial = boxes.clone()
                    trial[mover, 0] = x
                    trial[mover, 1] = y
                    trial[mover] = _clamp_box(trial[mover].unsqueeze(0))[0]
                    c = _total_cost(trial, room_types, original, ids)
                    if c < best_cost:
                        best_cost = c
                        best = trial[mover].clone()

                if not torch.allclose(best, boxes[mover], atol=1e-6):
                    boxes[mover] = best
                    changed = True

        if not changed:
            break

    # Consolidation pass: the layout is now collision-free but rooms
    # can still be left floating apart with dead space between them.
    # Pull each room toward its nearest neighbor so walls end up
    # shared like a real house, using a much lighter pull back toward
    # the original prediction (we're rearranging within an already-
    # valid layout, not re-solving collisions) - the flat +500
    # overlap penalty in _room_cost keeps this from ever reintroducing
    # a collision to save a smaller gap.
    for _ in range(50):
        changed = False

        for i in ids:
            changed |= _place_one(
                boxes, i, room_types, original, ids, living_center,
                displacement_weight=0.15, gap_weight=90.0,
                attach_map=attach_map, attach_weight=140.0,
            )

        boxes = _enforce_exterior(boxes, room_types, ids)

        if not changed:
            break

    return boxes * (mask > 0).unsqueeze(-1).to(boxes.dtype)


def refine_layout(
    prediction,
    room_types,
    mask,
    iterations=60,
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
    print("Geometry Layout Solver V2 ready.")
