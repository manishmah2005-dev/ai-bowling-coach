import csv
import os
from datetime import datetime

CSV_FILE = "progress_log.csv"

# -------------------------------------------------------
# Create CSV file if it doesn't exist
# -------------------------------------------------------
def init_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Date", "Time", "Session",
                "Head_Falling", "Lunging_Front_Foot",
                "Wide_Backlift", "Weight_Back", "Elbow_Drop",
                "Total_Weaknesses"
            ])
        print(f"Created: {CSV_FILE}")

# -------------------------------------------------------
# Save one session's weaknesses to CSV
# -------------------------------------------------------
def save_session(weaknesses, session_num):
    init_csv()

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    # Check which weaknesses were detected
    head      = 1 if any("Head"   in w for w in weaknesses) else 0
    lunging   = 1 if any("Lunging" in w for w in weaknesses) else 0
    backlift  = 1 if any("back-lift" in w for w in weaknesses) else 0
    weight    = 1 if any("Weight" in w for w in weaknesses) else 0
    elbow     = 1 if any("Elbow"  in w for w in weaknesses) else 0
    total     = head + lunging + backlift + weight + elbow

    with open(CSV_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            date_str, time_str, session_num,
            head, lunging, backlift, weight, elbow,
            total
        ])

    print(f"[Saved] Session {session_num} | Weaknesses: {total} | {weaknesses}")

# -------------------------------------------------------
# Show progress summary
# -------------------------------------------------------
def show_summary():
    if not os.path.exists(CSV_FILE):
        print("No data yet. Run a session first!")
        return

    print("\n===== PROGRESS SUMMARY =====")
    print(f"{'Session':<10} {'Date':<12} {'Total':<8} {'Weaknesses Detected'}")
    print("-" * 60)

    weakness_totals = {
        "Head Falling": 0,
        "Lunging": 0,
        "Wide Backlift": 0,
        "Weight Back": 0,
        "Elbow Drop": 0
    }

    with open(CSV_FILE, mode='r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

        for row in rows:
            detected = []
            if row["Head_Falling"]      == "1": detected.append("Head"); weakness_totals["Head Falling"] += 1
            if row["Lunging_Front_Foot"] == "1": detected.append("Lunging"); weakness_totals["Lunging"] += 1
            if row["Wide_Backlift"]     == "1": detected.append("Backlift"); weakness_totals["Wide Backlift"] += 1
            if row["Weight_Back"]       == "1": detected.append("Weight"); weakness_totals["Weight Back"] += 1
            if row["Elbow_Drop"]        == "1": detected.append("Elbow"); weakness_totals["Elbow Drop"] += 1

            detected_str = ", ".join(detected) if detected else "None"
            print(f"{row['Session']:<10} {row['Date']:<12} {row['Total_Weaknesses']:<8} {detected_str}")

    print("\n----- Most Common Weaknesses -----")
    for weakness, count in sorted(weakness_totals.items(), key=lambda x: x[1], reverse=True):
        bar = "#" * count
        print(f"{weakness:<20} {bar} ({count})")

    print("=" * 60)


if __name__ == "__main__":
    show_summary()