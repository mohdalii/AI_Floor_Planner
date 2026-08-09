"""
DXF export for generated floor plans.

Converts a generated floor plan (rooms + boxes, from
python/app/ml/predict.py) into a DXF file: one closed rectangle per
room plus a text label, scaled into real-world units so it opens at a
sensible scale in AutoCAD.
"""

from pathlib import Path

import ezdxf
from ezdxf.enums import TextEntityAlignment

ROOT = Path(__file__).resolve().parents[3]


def export_to_dxf(rooms, boxes, output_path, plot_size_m=10.0):
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    doc.layers.add(name="ROOMS", color=7)
    doc.layers.add(name="LABELS", color=3)

    for room, box in zip(rooms, boxes):
        cx, cy, w, h = [float(v) * plot_size_m for v in box]
        x1, y1 = cx - w / 2, cy - h / 2
        x2, y2 = cx + w / 2, cy + h / 2

        msp.add_lwpolyline(
            [(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
            close=True,
            dxfattribs={"layer": "ROOMS"},
        )

        label = msp.add_text(
            f"{room['type']}_{room['room_index']}",
            dxfattribs={"layer": "LABELS", "height": plot_size_m * 0.02},
        )
        label.set_placement((cx, cy), align=TextEntityAlignment.MIDDLE_CENTER)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(output_path)
    return output_path


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(ROOT / "python" / "app" / "ml"))
    from predict import generate_floor_plan  # noqa: E402

    demo_requirements = {
        "bedrooms": 3,
        "bathrooms": 2,
        "kitchens": 1,
        "living_rooms": 1,
        "balconies": 2,
        "storages": 1,
        "stairs": 0,
    }

    print("=" * 70)
    print("AI FLOOR PLANNER - DXF EXPORT")
    print("=" * 70)

    result = generate_floor_plan(demo_requirements)

    output_path = ROOT / "dataset" / "generated_plans" / "demo_plan.dxf"
    export_to_dxf(result["rooms"], result["solved_boxes"], output_path)

    print(f"\nRooms exported: {len(result['rooms'])}")
    print(f"Saved DXF to  : {output_path}")
    print("\nDone.")
