import requests
import pandas as pd
import json
import re

def fetch_all_courses(offset=0, limit=1000):
    """Fetch MIT OCW courses using the official API."""
    url = "https://open.mit.edu/api/v0/search/"
    
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Origin": "https://ocw.mit.edu",
        "Referer": "https://ocw.mit.edu/",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.6 Safari/605.1.15"
    }
    
    search_term = ""  # Empty string to get all courses
    
    payload = {
        "from": offset,
        "size": limit,
        "post_filter": {
            "bool": {
                "must": [
                    {"bool": {"should": [{"term": {"object_type.keyword": "course"}}]}},
                    {"bool": {"should": [{"term": {"offered_by": "OCW"}}]}}
                ]
            }
        },
        "query": {
            "bool": {
                "should": [
                    {"bool": {
                        "filter": {"bool": {"must": [
                            {"term": {"object_type": "course"}},
                            {"bool": {"should": [
                                {"multi_match": {"query": search_term, "fields": ["title.english^3", "short_description.english^2", "full_description.english", "topics", "platform", "course_id", "offered_by", "department_name", "course_feature_tags"]}},
                                {"wildcard": {"coursenum": {"value": f"{search_term}*", "boost": 100, "rewrite": "constant_score"}}},
                                {"nested": {"path": "runs", "query": {"multi_match": {"query": search_term, "fields": ["runs.year", "runs.semester", "runs.level", "runs.instructors^5", "department_name"]}}}},
                                {"has_child": {"type": "resourcefile", "query": {"multi_match": {"query": search_term, "fields": ["content", "title.english^3", "short_description.english^2", "department_name", "resource_type"]}}, "score_mode": "avg"}}
                            ]}}
                        ]}},
                        "should": [
                            {"multi_match": {"query": search_term, "fields": ["title.english^3", "short_description.english^2", "full_description.english", "topics", "platform", "course_id", "offered_by", "department_name", "course_feature_tags"]}},
                            {"wildcard": {"coursenum": {"value": f"{search_term}*", "boost": 100, "rewrite": "constant_score"}}},
                            {"nested": {"path": "runs", "query": {"multi_match": {"query": search_term, "fields": ["runs.year", "runs.semester", "runs.level", "runs.instructors^5", "department_name"]}}}},
                            {"has_child": {"type": "resourcefile", "query": {"multi_match": {"query": search_term, "fields": ["content", "title.english^3", "short_description.english^2", "department_name", "resource_type"]}}, "score_mode": "avg"}}
                        ]
                    }}]
                }
            }
        }
    
    
    resp = requests.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    
    # Convert Elasticsearch response to match expected format
    raw_response = resp.json()
    hits = raw_response.get('hits', {})
    results = hits.get('hits', [])
    
    # Extract course data from _source field and return in expected format
    courses = [hit.get('_source', {}) for hit in results]
    
    return {
        "results": courses,
        "total": hits.get('total', {}).get('value', 0) if isinstance(hits.get('total'), dict) else hits.get('total', 0)
    }

def normalize_courses(data):
    """Normalize MIT OCW API course JSON into a structured DataFrame."""
    results = data.get("results", [])
    cleaned = []
    
    for c in results:
        # Extract course code
        course_code = c.get('coursenum', '') or c.get('course_id', '') or 'Unknown'
        
        # Extract title
        title = c.get('title', 'No title')
        
        # Extract instructors from runs
        instructors = []
        meeting_times = []
        runs = c.get('runs', [])
        
        for run in runs:
            # Get instructors
            run_instructors = run.get('instructors', [])
            if isinstance(run_instructors, list):
                instructors.extend(run_instructors)
            elif isinstance(run_instructors, str):
                instructors.append(run_instructors)
            
            # Get meeting times info (semester, year, level)
            year = run.get('year', '')
            semester = run.get('semester', '')
            level = run.get('level', '')
            if year or semester or level:
                meeting_times.append(f"{semester} {year} ({level})".strip())
        
        # Remove duplicates and join
        unique_instructors = list(dict.fromkeys(instructors))
        instructor = "; ".join(unique_instructors) if unique_instructors else "Not specified"
        
        # Meeting times - use archived course info or runs info
        if meeting_times:
            meeting_times_str = "; ".join(set(meeting_times))
        else:
            meeting_times_str = "Archived Course - No Meeting Times"
        
        # Extract prerequisites
        prerequisites = extract_prerequisites(c)
        
        # Extract department - handle both string and list formats
        department_raw = c.get('department_name', 'Unknown Department')
        if isinstance(department_raw, list):
            department = "; ".join(str(d) for d in department_raw if d) if department_raw else 'Unknown Department'
        else:
            department = str(department_raw) if department_raw else 'Unknown Department'
        
        # Extract description - handle both string and list formats
        description_raw = c.get('short_description', '') or c.get('full_description', '') or "No description available"
        if isinstance(description_raw, list):
            description = " ".join(str(d) for d in description_raw if d) if description_raw else "No description available"
        else:
            description = str(description_raw) if description_raw else "No description available"
        
        if len(description) > 500:
            description = description[:500] + "..."
        
        # Ensure all fields are strings to avoid pandas issues
        cleaned.append({
            "institute": "MIT OpenCourseWare",
            "course_code": str(course_code) if course_code else "Unknown",
            "title": str(title) if title else "No title",
            "instructor": str(instructor) if instructor else "Not specified",
            "meeting_times": str(meeting_times_str) if meeting_times_str else "Archived Course - No Meeting Times",
            "prerequisites": str(prerequisites) if prerequisites else "Not specified",
            "department": str(department) if department else "Unknown Department",
            "description": str(description) if description else "No description available"
        })
    
    # Save to file (fixed the json serialization issue)
    # with open("mit_courses.txt", "w") as f:
    #     json.dump(cleaned, f, indent=2)
    
    return pd.DataFrame(cleaned)

def extract_prerequisites(course):
    """Extract prerequisites from course data."""
    prereq_sources = [
        course.get('full_description', ''),
        course.get('short_description', ''),
        course.get('course_feature_tags', [])
    ]
    
    prereq_patterns = [
        r'prerequisite[s]?[:\s]+([^\.]+)',
        r'prereq[s]?[:\s]+([^\.]+)', 
        r'requirements?[:\s]+([^\.]+)',
        r'assumes[:\s]+([^\.]+)',
        r'students should have[:\s]+([^\.]+)'
    ]
    
    for source in prereq_sources:
        if isinstance(source, str) and source:
            text = source.lower()
            for pattern in prereq_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    prereq = match.group(1).strip()
                    if len(prereq) > 10 and len(prereq) < 300:
                        return prereq
        elif isinstance(source, list):
            for tag in source:
                if isinstance(tag, str) and ('prerequisite' in tag.lower() or 'prereq' in tag.lower()):
                    return tag
    
    return "Not specified"

def fetch_all_courses_paginated():
    """Fetch all MIT OCW courses using pagination."""
    all_courses = []
    offset = 0
    limit = 1000
    
    while True:
        print(f"Fetching courses {offset} to {offset + limit}...")
        
        try:
            batch_data = fetch_all_courses(offset=offset, limit=limit)
            batch_results = batch_data.get("results", [])
            
            if not batch_results:
                print("No more courses found")
                break
            
            all_courses.extend(batch_results)
            print(f"Retrieved {len(batch_results)} courses (Total: {len(all_courses)})")
            
            # If we got fewer results than limit, we're done
            if len(batch_results) < limit:
                print("Reached last page")
                break
                
            offset += limit
            
        except Exception as e:
            print(f"Error fetching batch at offset {offset}: {e}")
            break
    
    return {"results": all_courses, "total": len(all_courses)}

def get_mit_courses_dataframe():
    print("Fetching all MIT OpenCourseWare courses...")
    raw = fetch_all_courses_paginated()
    print(f"Total courses fetched: {raw.get('total', 0)}")
    df = normalize_courses(raw)
    return df

if __name__ == "__main__":
    df = get_mit_courses_dataframe()
    
    print("\nSample courses:")
    print(df.head())
    print(f"\nTotal normalized courses: {len(df)}")
    
    # Save to CSV as well
    df.to_csv("mit_courses.csv", index=False)
    print("Data saved to mit_courses.txt and mit_courses.csv")
    
    # Show some statistics
    print(f"\nDataset Statistics:")
    print(f"Departments: {df['department'].nunique()}")
    print(f"Courses with instructors: {len(df[df['instructor'] != 'Not specified'])}")
    print(f"Courses with prerequisites: {len(df[df['prerequisites'] != 'Not specified'])}")
    
    print(f"\nTop departments:")
    print(df['department'].value_counts().head(10))