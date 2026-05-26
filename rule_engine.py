def get_strategies(scores, weaknesses):
    """
    Numeric scores ke basis par personalized bowling strategies generate karo.
    Har player ke scores alag honge — toh strategies bhi alag hongi.
    """
    strategies = []

    balance = scores.get('balance', 100)
    footwork = scores.get('footwork', 100)
    timing = scores.get('timing', 100)
    bat_swing = scores.get('bat_swing', 100)
    overall = scores.get('overall', 100)

    # ── BALANCE based rules ──────────────────────────────────────
    if balance < 40:
        strategies.append({
            'priority': 'HIGH',
            'line': 'Middle-Off stump',
            'length': 'Good Length',
            'speed': 'Fast (140-150 kmph)',
            'variation': 'Inswing',
            'why': f'Balance score {balance}/100 — head falling badly. Inswing will follow the head.'
        })
    elif balance < 65:
        strategies.append({
            'priority': 'MEDIUM',
            'line': 'Off stump',
            'length': 'Good Length',
            'speed': 'Medium-Fast (125-135 kmph)',
            'variation': 'Outswing',
            'why': f'Balance score {balance}/100 — head slightly off. Outswing outside off stump.'
        })

    # ── FOOTWORK based rules ─────────────────────────────────────
    if footwork < 40:
        strategies.append({
            'priority': 'HIGH',
            'line': 'Stump line',
            'length': 'Short (Back of length)',
            'speed': 'Fast (140+ kmph)',
            'variation': 'Bouncer',
            'why': f'Footwork score {footwork}/100 — lunging badly. Short ball will rush the batsman.'
        })
    elif footwork < 65:
        strategies.append({
            'priority': 'MEDIUM',
            'line': 'Middle stump',
            'length': 'Short-Good Length',
            'speed': 'Medium-Fast (130-140 kmph)',
            'variation': 'Seam movement',
            'why': f'Footwork score {footwork}/100 — footwork weak. Variable length will create confusion.'
        })

    # ── TIMING based rules ───────────────────────────────────────
    if timing < 40:
        strategies.append({
            'priority': 'HIGH',
            'line': 'Yorker line (toes)',
            'length': 'Full / Yorker',
            'speed': 'Fast (140-150 kmph)',
            'variation': 'Straight yorker',
            'why': f'Timing score {timing}/100 — elbow dropped. Fast yorker before bat comes down.'
        })
    elif timing < 65:
        strategies.append({
            'priority': 'MEDIUM',
            'line': 'Off stump',
            'length': 'Full',
            'speed': 'Medium (120-130 kmph)',
            'variation': 'Slower ball',
            'why': f'Timing score {timing}/100 — timing off. Slower ball will induce mistimed shot.'
        })

    # ── BAT SWING based rules ────────────────────────────────────
    if bat_swing < 40:
        strategies.append({
            'priority': 'HIGH',
            'line': 'Off stump',
            'length': 'Good Length',
            'speed': 'Medium-Fast (130-140 kmph)',
            'variation': 'Away swing',
            'why': f'Bat swing score {bat_swing}/100 — open face. Away swing will find outside edge.'
        })
    elif bat_swing < 65:
        strategies.append({
            'priority': 'MEDIUM',
            'line': 'Middle-Off',
            'length': 'Good Length',
            'speed': 'Medium (125-135 kmph)',
            'variation': 'Off cutter',
            'why': f'Bat swing score {bat_swing}/100 — back-lift issue. Off cutter will beat the bat.'
        })

    # ── COMBO ATTACKS (multiple weak areas) ──────────────────────
    if balance < 65 and footwork < 65:
        strategies.append({
            'priority': 'COMBO',
            'line': 'Middle stump',
            'length': 'Good Length',
            'speed': 'Fast (140+ kmph)',
            'variation': 'Inswing',
            'why': f'Balance {balance} + Footwork {footwork} — both weak. Inswing on stumps is deadly combo.'
        })

    if timing < 65 and bat_swing < 65:
        strategies.append({
            'priority': 'COMBO',
            'line': 'Off stump',
            'length': 'Full',
            'speed': 'Medium-Fast (130-140 kmph)',
            'variation': 'Outswing + Slower ball mix',
            'why': f'Timing {timing} + Bat swing {bat_swing} — both weak. Mix pace to create confusion.'
        })

    # ── DEFAULT (agar sab scores achhe hain) ─────────────────────
    if not strategies:
        strategies.append({
            'priority': 'STANDARD',
            'line': 'Off stump',
            'length': 'Good Length',
            'speed': 'Medium-Fast (130-140 kmph)',
            'variation': 'Seam movement',
            'why': f'Overall score {overall}/100 — batsman technically sound. Standard line and length.'
        })

    # Priority ke hisaab se sort karo
    priority_order = {'COMBO': 0, 'HIGH': 1, 'MEDIUM': 2, 'STANDARD': 3}
    strategies.sort(key=lambda x: priority_order.get(x['priority'], 4))

    return strategies


def generate_report(scores, weaknesses):
    """Terminal mein readable report print karo"""
    print("\n" + "=" * 55)
    print("   AI BOWLING COACH — PERSONALIZED STRATEGY REPORT")
    print("=" * 55)

    print("\n  SCORES:")
    for key, val in scores.items():
        if key != 'overall':
            bar = '#' * (val // 10) + '-' * (10 - val // 10)
            status = "GOOD" if val >= 65 else "WEAK"
            print(f"    {key.title():12} [{bar}] {val}/100  {status}")

    print(f"\n  Overall Rating: {scores.get('overall', 0)}/100")

    if weaknesses:
        print(f"\n  Weaknesses Detected ({len(weaknesses)}):")
        for w in weaknesses:
            print(f"    - {w}")
    else:
        print("\n  No major weaknesses detected!")

    strategies = get_strategies(scores, weaknesses)

    print(f"\n  BOWLING STRATEGIES ({len(strategies)} recommended):")
    for i, s in enumerate(strategies, 1):
        print(f"\n  [{s['priority']}] Strategy {i}")
        print(f"    Line      : {s['line']}")
        print(f"    Length    : {s['length']}")
        print(f"    Speed     : {s['speed']}")
        print(f"    Variation : {s['variation']}")
        print(f"    Why       : {s['why']}")

    print("\n" + "=" * 55)
    return strategies


if __name__ == "__main__":
    # Test with different player profiles
    print("\n--- TEST: Player with poor balance and footwork ---")
    scores1 = {'balance': 35, 'footwork': 42, 'timing': 80, 'bat_swing': 75, 'overall': 55}
    weaknesses1 = ['Head falling to off-side', 'Lunging on front foot']
    generate_report(scores1, weaknesses1)

    print("\n--- TEST: Player with poor timing and bat swing ---")
    scores2 = {'balance': 78, 'footwork': 82, 'timing': 38, 'bat_swing': 41, 'overall': 61}
    weaknesses2 = ['Elbow dropped - late swing', 'Wide back-lift / open bat face']
    generate_report(scores2, weaknesses2)

    print("\n--- TEST: Strong player (Virat-like) ---")
    scores3 = {'balance': 91, 'footwork': 88, 'timing': 85, 'bat_swing': 90, 'overall': 89}
    weaknesses3 = []
    generate_report(scores3, weaknesses3)