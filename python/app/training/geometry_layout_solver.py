from __future__ import annotations

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


# ------------------------------------------------------------
# BASIC BOX HELPERS
# ------------------------------------------------------------

def _clamp_box(box):
    cx, cy, w, h = box.unbind(-1)

    w = w.clamp(0.002, 0.90)
    h = h.clamp(0.002, 0.90)

    cx = cx.clamp(w / 2, 1.0 - w / 2)
    cy = cy.clamp(h / 2, 1.0 - h / 2)

    return torch.stack(
        [cx, cy, w, h],
        dim=-1,
    )


def _overlap(a, b):
    ax, ay, aw, ah = [float(v) for v in a]
    bx, by, bw, bh = [float(v) for v in b]

    ow = (
        min(ax + aw / 2, bx + bw / 2)
        - max(ax - aw / 2, bx - bw / 2)
    )

    oh = (
        min(ay + ah / 2, by + bh / 2)
        - max(ay - ah / 2, by - bh / 2)
    )

    return ow, oh


def _priority(room_type):
    return {
        3: 100,  # living
        2: 90,   # kitchen
        0: 80,   # bedroom
        6: 75,   # stair
        1: 40,   # bathroom
        5: 30,   # storage
        4: 20,   # balcony
        7: 10,   # front door
    }.get(
        int(room_type),
        50,
    )


def _area(box):
    return float(box[2] * box[3])


# ------------------------------------------------------------
# MOVE ROOM AWAY FROM ANOTHER ROOM
# ------------------------------------------------------------

def _separate_pair(
    boxes,
    i,
    j,
    room_types,
    margin,
):
    ow, oh = _overlap(
        boxes[i],
        boxes[j],
    )

    if ow <= 0 or oh <= 0:
        return False

    ti = int(room_types[i])
    tj = int(room_types[j])

    pi = _priority(ti)
    pj = _priority(tj)

    # Lower priority room moves.
    if pi > pj:
        mover = j
        anchor = i
    elif pj > pi:
        mover = i
        anchor = j
    else:
        # If same priority, move the smaller room.
        if _area(boxes[i]) <= _area(boxes[j]):
            mover = i
            anchor = j
        else:
            mover = j
            anchor = i

    m = boxes[mover].clone()
    a = boxes[anchor]

    mx, my, mw, mh = [
        float(v)
        for v in m
    ]

    ax, ay, aw, ah = [
        float(v)
        for v in a
    ]

    # Required center positions for separating
    # horizontally and vertically.

    left_x = (
        ax
        - aw / 2
        - mw / 2
        - margin
    )

    right_x = (
        ax
        + aw / 2
        + mw / 2
        + margin
    )

    top_y = (
        ay
        - ah / 2
        - mh / 2
        - margin
    )

    bottom_y = (
        ay
        + ah / 2
        + mh / 2
        + margin
    )

    candidates = [
        (
            abs(mx - left_x),
            left_x,
            my,
        ),
        (
            abs(mx - right_x),
            right_x,
            my,
        ),
        (
            abs(my - top_y),
            mx,
            top_y,
        ),
        (
            abs(my - bottom_y),
            mx,
            bottom_y,
        ),
    ]

    valid = []

    for distance, x, y in candidates:

        if (
            mw / 2 <= x <= 1.0 - mw / 2
            and
            mh / 2 <= y <= 1.0 - mh / 2
        ):
            valid.append(
                (
                    distance,
                    x,
                    y,
                )
            )

    if not valid:

        # Room cannot move without leaving the plot.
        # Try shrinking it slightly.

        new_w = max(
            mw * 0.94,
            0.002,
        )

        new_h = max(
            mh * 0.94,
            0.002,
        )

        m[2] = new_w
        m[3] = new_h

        boxes[mover] = _clamp_box(
            m.unsqueeze(0)
        )[0]

        return True

    _, x, y = min(
        valid,
        key=lambda item: item[0],
    )

    m[0] = x
    m[1] = y

    boxes[mover] = _clamp_box(
        m.unsqueeze(0)
    )[0]

    return True


# ------------------------------------------------------------
# REMOVE REMAINING COLLISIONS
# ------------------------------------------------------------

def _resolve_collisions(
    boxes,
    room_types,
    ids,
    iterations,
    margin,
):
    for _ in range(iterations):

        changed = False

        for p in range(len(ids)):

            for q in range(
                p + 1,
                len(ids),
            ):

                i = ids[p]
                j = ids[q]

                ow, oh = _overlap(
                    boxes[i],
                    boxes[j],
                )

                if ow > 1e-5 and oh > 1e-5:

                    changed |= _separate_pair(
                        boxes,
                        i,
                        j,
                        room_types,
                        margin,
                    )

        boxes = _clamp_box(boxes)

        # Check whether ANY collision remains.

        collision = False

        for p in range(len(ids)):

            for q in range(
                p + 1,
                len(ids),
            ):

                ow, oh = _overlap(
                    boxes[ids[p]],
                    boxes[ids[q]],
                )

                if (
                    ow > 1e-5
                    and
                    oh > 1e-5
                ):
                    collision = True
                    break

            if collision:
                break

        if not collision:
            break

        if not changed:
            break

    return boxes


# ------------------------------------------------------------
# SINGLE PLAN
# ------------------------------------------------------------

def refine_single_layout(
    prediction,
    room_types,
    mask,
    iterations=160,
    margin=0.008,
):
    boxes = _clamp_box(
        prediction.detach().clone()
    )

    ids = [
        i
        for i in range(
            len(boxes)
        )
        if bool(mask[i])
    ]

    # Important rooms first.
    ids.sort(
        key=lambda i: (
            -_priority(
                int(room_types[i])
            ),
            -_area(boxes[i]),
        )
    )

    boxes = _resolve_collisions(
        boxes,
        room_types,
        ids,
        iterations,
        margin,
    )

    # --------------------------------------------------------
    # FINAL MICRO-PASS
    # --------------------------------------------------------

    # Use a slightly larger margin to eliminate
    # tiny numerical contacts.

    boxes = _resolve_collisions(
        boxes,
        room_types,
        ids,
        iterations=80,
        margin=0.004,
    )

    boxes = _clamp_box(boxes)

    return (
        boxes
        * (mask > 0)
        .unsqueeze(-1)
        .to(boxes.dtype)
    )


# ------------------------------------------------------------
# BATCH API
# ------------------------------------------------------------

def refine_layout(
    prediction,
    room_types,
    mask,
    iterations=160,
    margin=0.008,
):
    return torch.stack(
        [
            refine_single_layout(
                prediction[b],
                room_types[b],
                mask[b],
                iterations,
                margin,
            )
            for b in range(
                prediction.shape[0]
            )
        ],
        dim=0,
    )


# ------------------------------------------------------------
# COLLISION METRIC
# ------------------------------------------------------------

def collision_rate(
    prediction,
    mask,
):
    total = 0
    bad = 0

    for b in range(
        prediction.shape[0]
    ):

        ids = [
            i
            for i in range(
                prediction.shape[1]
            )
            if bool(mask[b, i])
        ]

        for p in range(
            len(ids)
        ):

            for q in range(
                p + 1,
                len(ids),
            ):

                total += 1

                ow, oh = _overlap(
                    prediction[
                        b,
                        ids[p],
                    ],
                    prediction[
                        b,
                        ids[q],
                    ],
                )

                if (
                    ow > 1e-5
                    and
                    oh > 1e-5
                ):
                    bad += 1

    return bad / max(
        total,
        1,
    )


if __name__ == "__main__":
    print(
        "Geometry Layout Solver ready."
    )
