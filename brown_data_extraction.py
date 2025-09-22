import requests
import pandas as pd
import json

def fetch_all_courses(srcdb="202510", offset=0, limit=500):
    """Fetch courses for a given term (srcdb)."""
    url = "https://cab.brown.edu/api/?page=fose&route=search"
    payload = {
        "other": {"srcdb": srcdb},
        "criteria": [],
        "offset": offset,
        "limit": limit
    }
    headers = {"Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    return resp.json()

def normalize_courses(data):
    """Normalize CAB API course JSON into a structured DataFrame."""
    results = data.get("results", [])
    cleaned = []
    
    for c in results:
        # Skip cancelled courses
        if c.get("isCancelled") == "1":
            continue
            
        # meetingTimes comes as a JSON string
        meeting_times = []
        if c.get("meetingTimes"):
            try:
                mt = json.loads(c["meetingTimes"])
                meeting_times = [
                    f"Day {m['meet_day']} {m['start_time']}-{m['end_time']}"
                    for m in mt
                ]
            except Exception:
                pass
        
        # Extract prerequisites (this may need adjustment based on actual API response)
        prerequisites = c.get("prereq", "") or c.get("prerequisites", "")
        
        cleaned.append({
            "institute" : "Brown University",
            "course_code": c.get("code"),
            "title": c.get("title"),
            "instructor": c.get("instr"),
            "meeting_times": "; ".join(meeting_times),
            "prerequisites": prerequisites,
            "department": c.get("code", "").split()[0] if c.get("code") else None,
            "description": c.get("desc", "") or ""
        })
    
    # Save to file (fixed the json serialization issue)
    # with open("courses.txt", "w") as f:
    #     json.dump(cleaned, f, indent=2)
    
    return pd.DataFrame(cleaned)

def get_brown_courses_dataframe(srcdb="202510"):
    raw = fetch_all_courses(srcdb)
    df = normalize_courses(raw)
    return df

if __name__ == "__main__":
    srcdb = "202510"  # Fall 2025
    df = get_brown_courses_dataframe(srcdb)
    print(df)
    print(f"\nTotal normalized courses (non-cancelled): {len(df)}")