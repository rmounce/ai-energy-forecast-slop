import json
import requests
from datetime import datetime, timedelta, timezone

# ==========================================
# CONFIGURATION
# ==========================================
HA_URL = "https://hass.REDACTED"
TOKEN = "REDACTED"
ENTITY_ID = "input_text.emhass_battery_action"
JSON_FILE = "solcast-generation.json"
# ==========================================


def get_ha_history():
    print(f"Fetching history for {ENTITY_ID}...")
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "content-type": "application/json",
    }
    
    # Explicitly set start and end times to get the full 10-day window
    now = datetime.now(timezone.utc)
    start_time = (now - timedelta(days=10)).strftime('%Y-%m-%dT%H:%M:%S') + 'Z'
    end_time = now.strftime('%Y-%m-%dT%H:%M:%S') + 'Z'
    
    url = f"{HA_URL}/api/history/period/{start_time}?end_time={end_time}&filter_entity_id={ENTITY_ID}&significant_changes_only=0"
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Error fetching data: {response.status_code} - {response.text}")
        return[]
    
    data = response.json()
    return data[0] if data else[]

def parse_history(history):
    """Converts the raw HA history into a list of cleanly formatted time ranges."""
    parsed =[]
    
    for i, state_obj in enumerate(history):
        state_str = state_obj.get('state', 'unknown')
        
        # Clean up timestamp strings for safe parsing
        start_str = state_obj.get('last_changed', state_obj.get('last_updated'))
        start_str = start_str.replace('Z', '+00:00')
        start_time = datetime.fromisoformat(start_str)
        
        # The state is active until the next state change
        if i + 1 < len(history):
            next_str = history[i+1].get('last_changed', history[i+1].get('last_updated'))
            next_str = next_str.replace('Z', '+00:00')
            end_time = datetime.fromisoformat(next_str)
        else:
            # If it's the last recorded state, it's active until right now
            end_time = datetime.now(timezone.utc)
            
        parsed.append({
            'state': state_str,
            'start': start_time,
            'end': end_time
        })
        
    return parsed

def main():
    raw_history = get_ha_history()
    if not raw_history:
        print(f"No history found for {ENTITY_ID}. Check your entity name and recorder settings.")
        return

    # Parse and print the debug log
    print("\n--- EMHASS STATE HISTORY DEBUG LOG ---")
    parsed_history = parse_history(raw_history)
    for ph in parsed_history:
        # Convert to local strings just for easier reading in the console
        start_local = ph['start'].astimezone().strftime('%Y-%m-%d %H:%M:%S')
        end_local = ph['end'].astimezone().strftime('%Y-%m-%d %H:%M:%S')
        print(f"State: '{ph['state']}' | Active: {start_local} TO {end_local}")
    print("--------------------------------------\n")

    print("Reading solcast-generation.json...")
    try:
        with open(JSON_FILE, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {JSON_FILE} not found in the current directory.")
        return

    updates = 0
    print("Analyzing 30-minute Solcast periods against history...")
    
    for period in data.get('generation',[]):
        # Parse Solcast period (e.g., "2026-03-21T13:30:00+00:00")
        period_start_str = period['period_start'].replace('Z', '+00:00')
        period_start = datetime.fromisoformat(period_start_str)
        period_end = period_start + timedelta(minutes=30)
        
        # Check if any "curtail" state overlaps with this 30-minute window
        is_curtailed = False
        for ph in parsed_history:
            # Overlap logic: State starts before period ends AND State ends after period starts
            if ph['start'] < period_end and ph['end'] > period_start:
                if "curtail" in ph['state'].lower():
                    is_curtailed = True
                    break # Found a curtailment overlap, no need to check other states
        
        if is_curtailed:
            # Only count it as an update if we are actively changing it from false -> true
            if not period.get('export_limiting', False):
                period['export_limiting'] = True
                updates += 1

    print(f"\nIdentified {updates} NEW curtailed 30-minute periods to update.")
    
    if updates > 0:
        print("Saving updated JSON...")
        with open(JSON_FILE, 'w') as f:
            json.dump(data, f, separators=(',', ':'))
        print("Done! Upload the file back to HA and restart.")
    else:
        print("No changes made to the JSON file.")

if __name__ == "__main__":
    main()
