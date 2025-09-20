import requests
import os
import pytest

# The base URL for our running service. Assumes it's running locally on port 8000.
API_URL = "http://localhost:8000/v1/transcribe"

SAMPLE_AUDIO_PATH = os.path.join(os.path.dirname(__file__), "test_data", "sample.wav")

@pytest.fixture
def api_service_is_running():
    """A pytest fixture to check if the API is reachable before running tests."""
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        # We expect a 200 OK for the root health check endpoint
        assert response.status_code == 200
    except requests.ConnectionError:
        pytest.fail(
            "The API service is not running or not reachable at http://localhost:8000. "
            "Please start the service with 'docker-compose up -d' before running tests."
        )

def test_transcribe_endpoint_success(api_service_is_running):
    """
    Tests the /v1/transcribe endpoint with a valid audio file.
    """
    # Check if the sample audio file exists
    assert os.path.exists(SAMPLE_AUDIO_PATH), f"Sample audio file not found at: {SAMPLE_AUDIO_PATH}"

    # Open the audio file in binary-read mode
    with open(SAMPLE_AUDIO_PATH, "rb") as audio_file:
        # Prepare the multipart/form-data payload
        files = {"file": (os.path.basename(SAMPLE_AUDIO_PATH), audio_file, "audio/wav")}
        
        # Send the POST request to the API
        response = requests.post(API_URL, files=files)

    # 1. Assert that the request was successful
    assert response.status_code == 200, f"API returned an error: {response.text}"

    # 2. Assert that the response is valid JSON
    try:
        response_data = response.json()
    except ValueError:
        pytest.fail("Response is not valid JSON.")

    # 3. Assert that the response contains the expected top-level keys
    expected_keys = ["request_id", "duration_sec", "text", "language", "segments", "pipeline"]
    for key in expected_keys:
        assert key in response_data, f"Response JSON is missing expected key: '{key}'"
    
    # 4. Assert that the pipeline and segments look reasonable
    assert "separation" in response_data["pipeline"]
    assert "transcription" in response_data["pipeline"]
    assert isinstance(response_data["segments"], list)
    
    print(f"\nSuccessfully transcribed: '{response_data['text']}'")

def test_transcribe_endpoint_invalid_file_format():
    """
    Tests the API's response when a non-audio file is uploaded.
    """
    # Create a dummy text file to simulate an invalid upload
    dummy_file_content = b"this is not an audio file"
    files = {"file": ("invalid.txt", dummy_file_content, "text/plain")}

    response = requests.post(API_URL, files=files)

    # Assert that the API correctly identifies it as a bad request (400)
    assert response.status_code == 400
    response_data = response.json()
    assert "detail" in response_data
    assert "invalid file format" in response_data["detail"].lower()