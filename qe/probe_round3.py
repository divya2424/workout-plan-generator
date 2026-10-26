"""Exploration probes for round 3 of qe-bug-fixes (no network). Run: uv run python -m qe.probe_round3"""

from backend.core.utils import (
    find_equipment_conflicts,
    find_limitation_conflicts,
    is_stretch,
)

LIMITS = {
    "knee": "bad knees",
    "overhead": "no overhead pressing, shoulder pain",
    "wrist": "sore wrist",
    "back": "lower back pain",
}

SAFE_NAMES = [
    # names that must NOT be flagged for any guard
    "Glute Bridge",
    "Wall Sit",
    "Calf Raise",
    "Dead Bug",
    "Bird Dog",
    "Forearm Plank",
    "Side Plank (forearm)",
    "Seated Row",
    "Lat Pulldown",
    "Leg Press",
    "Chest Press",
    "Hip Thrust",
    "Push Upright Row",
    "Press Upright Hold",
    "Dumbbell Press Up to Lockout",
    "Bench Press",
    "Floor Press",
    "Spiderman Crawl",
    "Goblet Squat",
    "Step Back Lunge",
    "Sit Back Squat",
    "Box Squat",
    "Situational drill",
    "Upright Row",
    "Supported Row",
    "Pressure Breathing",
    "Good Mornings Stretch",
    "Glute Kickback",
    "Jackknife",
    "Single-Arm Kettlebell Swing",
    "One-Arm Swing",
    "Single Arm Swing",
    "Kettlebell Swing",
    "Hip Circle Band Walk",
    "Leg Swing Kick",
    "Thoracic Bridge",
    "Pullover",
    "Hip Hinge",
    "Superman Hold",
    "Tricep Kickback",
    "Hip Abduction (lying)",
    "Lying Hip Adductor Squeeze",
    "Side-Lying Leg Raise",
    "Arm Swing Squat Jump",
    "Pike Push-up",
]
for name in SAFE_NAMES:
    flagged = {k: find_limitation_conflicts([name], v) for k, v in LIMITS.items()}
    hits = [k for k, v in flagged.items() if v]
    print(
        f"{name:40} guards={hits} stretch={is_stretch(name)}"
        f" none={bool(find_equipment_conflicts([name], ['No equipment']))}"
        f" db={bool(find_equipment_conflicts([name], ['Home dumbbells']))}"
    )
