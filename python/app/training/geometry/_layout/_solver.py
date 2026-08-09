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


def _room_cost(boxes, i, room_types, original, ids):
    t = int(room_types[i])
    b = boxes[i]
    cost = 0.0

    # Keep the final layout reasonably close to the neural prediction.
    dx = float(b[0] - original[i, 0])
    dy = float(b[1] - original[i, 1])
    cost += 1.8 * (dx*dx + dy*dy)

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

    # Penalize overlaps strongly.
    for j in ids:
        if j == i:
            continue
        ow, oh = _overlap(b, boxes[j])
        if ow > 0 and oh > 0:
            cost += 160.0 * ow * oh
            # Stronger penalty for large interpenetration.
            cost += 35.0 * min(ow / max(float(b[2]), 1e-6),
                                oh / max(float(b[3]), 1e-6))

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
            cost += 2.2 * d*d

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


def _total_cost(boxes, room_types, original, ids):
    value = sum(_room_cost(boxes, i, room_types, original, ids) for i in ids)
    value += 6.0 * _relationship_cost(boxes, room_types, ids)
    return value


def _candidate_centers(box, room_type, living_center=None):
    cx, cy, w, h = [float(v) for v in box]
    candidates = [(cx, cy)]

    # Small local moves first.
    step = 0.06
    for dx, dy in [
        (-step, 0), (step, 0), (0, -step), (0, step),
        (-step, -step), (-step, step), (step, -step), (step, step),
        (-0.12, 0), (0.12, 0), (0, -0.12), (0, 0.12),
    ]:
        candidates.append((cx + dx, cy + dy))

    t = int(room_type)

    # Architectural fallback slots.
    if t == 4:  # balcony
        candidates += [(0.18, 0.90), (0.50, 0.90), (0.82, 0.90),
                       (0.10, 0.50), (0.90, 0.50)]
    elif t == 7:  # front door
        candidates += [(0.50, 0.04), (0.18, 0.04), (0.82, 0.04),
                       (0.04, 0.50), (0.96, 0.50)]
    elif t == 3:  # living
        candidates += [(0.50, 0.50), (0.42, 0.50), (0.58, 0.50)]
    elif t == 2:  # kitchen
        candidates += [(0.34, 0.50), (0.66, 0.50), (0.50, 0.34),
                       (0.50, 0.66)]
    elif t == 0:  # bedrooms
        candidates += [(0.25, 0.75), (0.50, 0.78), (0.75, 0.75),
                       (0.25, 0.25), (0.50, 0.22), (0.75, 0.25)]
    elif t == 1:  # bathrooms
        candidates += [(0.38, 0.62), (0.62, 0.62), (0.38, 0.38),
                       (0.62, 0.38), (0.50, 0.62)]
    elif t == 6:
        candidates += [(0.10, 0.50), (0.90, 0.50), (0.50, 0.15),
                       (0.50, 0.85)]
    else:
        candidates += [(0.25, 0.50), (0.75, 0.50), (0.50, 0.25),
                       (0.50, 0.75)]

    # If living exists, add positions around it.
    if living_center is not None and t != 3:
        lx, ly = living_center
        candidates += [
            (lx - 0.28, ly), (lx + 0.28, ly),
            (lx, ly - 0.28), (lx, ly + 0.28),
        ]

    return candidates


def _place_one(boxes, i, room_types, original, ids, living_center):
    current = boxes[i].clone()
    best = current.clone()
    best_cost = _total_cost(boxes, room_types, original, ids)

    t = int(room_types[i])
    for cx, cy in _candidate_centers(current, t, living_center):
        trial = boxes.clone()
        trial[i, 0] = cx
        trial[i, 1] = cy
        trial[i] = _clamp_box(trial[i].unsqueeze(0))[0]

        cost = _total_cost(trial, room_types, original, ids)
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

    # First repair pathological sizes.
    for i in ids:
        boxes[i] = _resize_box(boxes[i], int(room_types[i]))

    # Establish a central living anchor when one exists.
    living_ids = [i for i in ids if int(room_types[i]) == 3]
    living_center = None
    if living_ids:
        li = max(living_ids, key=lambda i: float(boxes[i, 2] * boxes[i, 3]))
        # Only move a wildly misplaced living room toward the center.
        lx, ly = _center(boxes[li])
        if lx < 0.18 or lx > 0.82 or ly < 0.18 or ly > 0.82:
            boxes[li, 0] = 0.50
            boxes[li, 1] = 0.50
        living_center = _center(boxes[li])

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
                boxes, i, room_types, original, ids, living_center
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
