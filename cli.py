import argparse
import requests
import os
import json
import sys

# Define the API endpoint URL
API_URL = "http://localhost:8000/v1/transcribe"

def transcribe_file(filepath: str, enable_separation: bool):
    """
    Sends an audio file to the transcription API and prints the result.
    """
    # --- 1. Validate Input ---
    if not os.path.exists(filepath):
        print(f"Error: The file '{filepath}' was not found.", file=sys.stderr)
        sys.exit(1)

    # --- 2. Prepare the Request Payload ---
    # The API expects multipart/form-data.
    # The 'file' part contains the audio file itself.
    # The 'config' part contains a JSON string with our options.
    
    config = {
        "enable_separation": enable_separation
    }

    try:
        with open(filepath, "rb") as audio_file:
            files = {
                'file': (os.path.basename(filepath), audio_file, 'audio/mpeg'), # MIME type is generic, API handles it
                'config': (None, json.dumps(config), 'application/json')
            }
            
            print(f"Uploading and processing '{os.path.basename(filepath)}'...")
            
            # --- 3. Make the API Call ---
            # We use a long timeout because transcription can take time.
            response = requests.post(API_URL, files=files, timeout=300)

            # --- 4. Handle the Response ---
            if response.status_code == 200:
                result = response.json()
                print("\n" + "="*25 + " Transcription Result " + "="*25)
                print(result.get("text", "No text found in the response."))
                print("="*72 + "\n")
            else:
                # Print a helpful error message if the API returns an error
                print(f"Error: API returned status code {response.status_code}", file=sys.stderr)
                try:
                    error_detail = response.json()
                    print(f"Detail: {error_detail.get('detail', response.text)}", file=sys.stderr)
                except json.JSONDecodeError:
                    print(f"Raw Response: {response.text}", file=sys.stderr)
    
    except requests.ConnectionError:
        print("\nError: Could not connect to the transcription API.", file=sys.stderr)
        print(f"Please ensure the service is running at {API_URL}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """
    Main function to parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="A command-line utility to transcribe audio files using the transcription service."
    )
    
    parser.add_argument(
        "filepath", 
        type=str, 
        help="The path to the audio file (e.g., audio.mp3, recording.wav)."
    )
    
    parser.add_argument(
        "--no-separation", 
        action="store_false", 
        dest="separation",
        help="Disable the vocal separation step before transcription."
    )
    
    # Set default value for separation
    parser.set_defaults(separation=True)

    args = parser.parse_args()
    
    transcribe_file(args.filepath, args.separation)


if __name__ == "__main__":
    main()

