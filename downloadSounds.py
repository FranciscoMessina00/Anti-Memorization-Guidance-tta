import csv
import requests
import os
from tqdm import tqdm
import time

def download_sound(sound_id, access_token, output_filename):
    """
    Downloads a sound file from Freesound given the sound ID and OAuth2 access token.
    
    Parameters:
        sound_id (int or str): The ID of the sound to download.
        access_token (str): OAuth2 Bearer token.
        output_filename (str): The filename where the downloaded sound will be saved.
    """
    # Construct the URL for the given sound_id.
    url = f"https://freesound.org/apiv2/sounds/{sound_id}/download/"
    
    # Set up the header with the OAuth2 token.
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an error for bad responses

        os.makedirs(os.path.dirname(output_filename), exist_ok=True)

        # Write the downloaded content to a file in binary mode.
        with open(output_filename, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded sound {sound_id} and saved as '{output_filename}'.")
    
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred for sound {sound_id}: {http_err} - {response.text}")
    except Exception as err:
        print(f"An error occurred for sound {sound_id}: {err}")

def process_csv(file_path, access_token):
    """
    Reads the CSV file, extracts sound IDs, and downloads each sound.
    
    Parameters:
        file_path (str): Path to the CSV file containing sound IDs.
        access_token (str): OAuth2 Bearer token.
    """
    try:
        with open(file_path, mode="r", newline="", encoding="utf-8-sig") as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            rows = list(reader)
            for row in tqdm(rows, desc="Downloading sounds"):
                # Extract sound ID. Change 'sound_id' if your column header is different.
                sound_id = row.get('id')
                if not sound_id:
                    print("Sound ID not found in row:", row)
                    continue

                # Define an output filename based on the sound id; adjust extension if necessary.
                output_filename = f"soundDataset/sound_{sound_id}.wav"
                if not os.path.exists(output_filename):
                    download_sound(sound_id, access_token, output_filename)
                    time.sleep(1)
    except FileNotFoundError:
        print(f"CSV file not found at path: {file_path}")
    except Exception as err:
        print(f"An error occurred while processing the CSV file: {err}")

if __name__ == "__main__":
    # Replace these values with your own
    ACCESS_TOKEN = ""  # Your Freesound OAuth2 access token
    CSV_FILE_PATH = "cleaned_output.csv"          # CSV file containing the sound IDs

    # Process the CSV file to download each sound.
    process_csv(CSV_FILE_PATH, ACCESS_TOKEN)
