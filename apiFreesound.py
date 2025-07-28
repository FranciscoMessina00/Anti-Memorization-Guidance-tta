import csv
import requests
import time  # Optional: to add a delay between requests if needed

# Replace with your actual OAuth2 access token
ACCESS_TOKEN = ""

# Input and output CSV file paths
input_csv = "input.csv"   # CSV file must include an "id" column
output_csv = "output.csv"

def saveCsv(results, file):
    # Write the results to the output CSV file
    with open(file, mode="w", newline="", encoding="utf-8") as outfile:
        fieldnames = ["id", "description"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for sound_id, description in results.items():
            writer.writerow({"id": sound_id, "description": description})

# This dictionary will hold the mapping {id: description}
results = {}

# Open the input CSV and iterate over rows
with open(input_csv, mode="r", newline="", encoding="utf-8-sig") as infile:
    reader = csv.DictReader(infile, delimiter=';')
    print(reader.fieldnames)
    for i, row in enumerate(reader):
        sound_id = row["id"]
        # Check if the sound_id matches the exit condition
        if sound_id == "48447":  # Replace with the actual value you want to stop at
            print(f"Stopping at id {sound_id}")
            break
        # Construct the API URL for the current sound id
        url = f"https://freesound.org/apiv2/sounds/{sound_id}/?fields=description"
        headers = {
            "Authorization": f"Token {ACCESS_TOKEN}",
            "Accept": "application/json"
        }
        try:
            response = requests.get(url, headers=headers)
            # Check if the request was successful
            if response.ok:
                json_response = response.json()
                # Get the "description" from the JSON, defaulting to an empty string if missing
                description = json_response.get("description", "")
            else:
                description = f"Error: {response.status_code}"
        except Exception as e:
            description = f"Exception: {e}"

        # Store the result in the dictionary
        results[sound_id] = description
        
        if i % 100 == 0:
            print(f"Row: {sound_id}\n")
            saveCsv(results, output_csv)
        # Optional: Sleep for a short while if you're making many requests to avoid rate limiting
        time.sleep(0.1)

saveCsv(results, output_csv)

print(f"Data saved to {output_csv}")
