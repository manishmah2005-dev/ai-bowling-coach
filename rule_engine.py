# ================================================================
# rule_engine.py — AI Bowling Recommendation Engine
# Maps batsman weaknesses -> bowling strategies
# ================================================================

import csv
import os
from collections import Counter

# ================================================================
# RULE BASE — weakness -> bowling strategies
# Each rule has:
#   - condition  : which weaknesses trigger it
#   - line       : where to bowl (Off stump / Middle / Leg / Yorker)
#   - length     : how full/short (Short / Good Length / Full / Yorker)
#   - speed      : pace recommendation
#   - variation  : type of delivery
#   - reason     : why this works
# ================================================================

RULES = [
    {
        "id": "R1",
        "condition": ["Head falling to off-side"],
        "line": "Middle stump",
        "length": "Good Length",
        "speed": "Medium-Fast (130-140 kmph)",
        "variation": "Straight / Slight Inswing",
        "reason": "Head falling off-side opens LBW gate. Straight ball hits stumps.",
        "priority": 1
    },
    {
        "id": "R2",
        "condition": ["Head falling to leg-side"],
        "line": "Off stump",
        "length": "Good Length",
        "speed": "Medium (120-130 kmph)",
        "variation": "Outswing",
        "reason": "Head falling leg-side means bat comes from wrong angle — outside edge likely.",
        "priority": 1
    },
    {
        "id": "R3",
        "condition": ["Lunging on front foot"],
        "line": "Off stump",
        "length": "Short of Good Length",
        "speed": "Fast (140-150 kmph)",
        "variation": "Bouncer / Short Pitch",
        "reason": "Batsman commits forward early — short ball hits body or gets top edge.",
        "priority": 1
    },
    {
        "id": "R4",
        "condition": ["Wide back-lift / open bat face"],
        "line": "Off stump",
        "length": "Good Length",
        "speed": "Medium-Fast (130-140 kmph)",
        "variation": "Inswing / Late Swing",
        "reason": "Open bat face = outside edge to slip. Inswing maximizes edge chance.",
        "priority": 2
    },
    {
        "id": "R5",
        "condition": ["Weight falling back"],
        "line": "Yorker Line (Stumps)",
        "length": "Full / Yorker",
        "speed": "Fast (140-150 kmph)",
        "variation": "Straight Yorker",
        "reason": "Weight back = no drive power. Full ball jams the batsman for LBW/bowled.",
        "priority": 1
    },
    {
        "id": "R6",
        "condition": ["Elbow dropped (collapsed arm)"],
        "line": "Short of Off stump",
        "length": "Short",
        "speed": "Fast (145+ kmph)",
        "variation": "Rising Delivery / Bouncer",
        "reason": "Collapsed elbow = weak pull shot. Rising ball gets top edge or hits glove.",
        "priority": 2
    },
    # Combo rules — multiple weaknesses together
    {
        "id": "R7",
        "condition": ["Head falling to off-side", "Wide back-lift / open bat face"],
        "line": "Middle-Off stump",
        "length": "Good Length",
        "speed": "Medium-Fast (130-140 kmph)",
        "variation": "Inswing",
        "reason": "COMBO: Head + open face = perfect inswing trap. High LBW/bowled chance.",
        "priority": 0  # highest priority
    },
    {
        "id": "R8",
        "condition": ["Weight falling back", "Elbow dropped (collapsed arm)"],
        "line": "Yorker Line",
        "length": "Yorker",
        "speed": "Fast (145+ kmph)",
        "variation": "Fast Yorker",
        "reason": "COMBO: Weight back + elbow drop = completely jammed. Full fast yorker = bowled.",
        "priority": 0
    },
]

# ================================================================
# Core: match weaknesses to rules
# ================================================================
def get_recommendations(weaknesses):
    """
    Input : list of weakness strings (from pose_analysis.py)
    Output: list of matched rules sorted by priority
    """
    if not weaknesses:
        return []

    matched = []

    for rule in RULES:
        # Check if ANY condition in rule matches detected weaknesses
        match_count = sum(1 for cond in rule["condition"] if cond in weaknesses)

        if match_count > 0:
            score = match_count / len(rule["condition"])  # match quality 0-1
            matched.append({**rule, "match_score": round(score, 2)})

    # Sort: combo rules first (priority 0), then by match score
    matched.sort(key=lambda x: (x["priority"], -x["match_score"]))

    return matched

# ================================================================
# Display recommendations nicely
# ================================================================
def display_recommendations(weaknesses, recommendations):
    print("\n" + "=" * 60)
    print("  AI BOWLING COACH - STRATEGY REPORT")
    print("=" * 60)

    if not weaknesses:
        print("  No weaknesses detected — batsman has good technique!")
        print("  Bowl: Vary pace and length to create uncertainty.")
        return

    print(f"\n  Weaknesses Detected ({len(weaknesses)}):")
    for w in weaknesses:
        print(f"    - {w}")

    if not recommendations:
        print("\n  No specific strategy found.")
        return

    print(f"\n  Recommended Strategies ({len(recommendations)} found):\n")

    for i, rec in enumerate(recommendations[:3]):  # top 3 only
        tag = "[COMBO ATTACK]" if rec["priority"] == 0 else f"[Strategy {i+1}]"
        print(f"  {tag}")
        print(f"  Rule ID   : {rec['id']}")
        print(f"  Line      : {rec['line']}")
        print(f"  Length    : {rec['length']}")
        print(f"  Speed     : {rec['speed']}")
        print(f"  Variation : {rec['variation']}")
        print(f"  Why       : {rec['reason']}")
        print(f"  Match     : {int(rec['match_score']*100)}% condition match")
        print()

    print("=" * 60)

# ================================================================
# Load weaknesses from CSV and recommend
# ================================================================
def recommend_from_csv(csv_file="progress_log.csv", session=None):
    if not os.path.exists(csv_file):
        print("No progress_log.csv found. Run pose_analysis.py first!")
        return

    weakness_counts = Counter()

    with open(csv_file, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if session and str(row["Session"]) != str(session):
                continue
            if row["Head_Falling"]       == "1": weakness_counts["Head falling to off-side"] += 1
            if row["Lunging_Front_Foot"] == "1": weakness_counts["Lunging on front foot"] += 1
            if row["Wide_Backlift"]      == "1": weakness_counts["Wide back-lift / open bat face"] += 1
            if row["Weight_Back"]        == "1": weakness_counts["Weight falling back"] += 1
            if row["Elbow_Drop"]         == "1": weakness_counts["Elbow dropped (collapsed arm)"] += 1

    # Only include weaknesses seen more than once
    active_weaknesses = [w for w, count in weakness_counts.items() if count > 0]

    recommendations = get_recommendations(active_weaknesses)
    display_recommendations(active_weaknesses, recommendations)

    return recommendations

# ================================================================
# Run directly
# ================================================================
if __name__ == "__main__":
    import sys
    session = sys.argv[1] if len(sys.argv) > 1 else None

    if session:
        print(f"Analyzing Session {session}...")
    else:
        print("Analyzing all sessions...")

    recommend_from_csv(session=session)