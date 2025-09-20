import requests
import os
import pytest

# The base URL for our running service. Assumes it's running locally on port 8000.
API_URL = "http://localhost:8000/v1/transcribe"

# IMPORTANT: Make sure you have a file named 'sample.mp3' inside the 'tests/test_data/' directory.
SAMPLE_AUDIO_PATH = os.path.join(os.path.dirname(__file__), "test_data", "sample.mp3")

@pytest.fixture
def api_service_is_running():
    """A pytest fixture to check if the API is reachable before running tests."""
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
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
    assert os.path.exists(SAMPLE_AUDIO_PATH), f"Sample audio file not found at: {SAMPLE_AUDIO_PATH}"

    with open(SAMPLE_AUDIO_PATH, "rb") as audio_file:
        files = {"file": (os.path.basename(SAMPLE_AUDIO_PATH), audio_file, "audio/mp3")}
        
        response = requests.post(API_URL, files=files)

    assert response.status_code == 200, f"API returned an error: {response.text}"

    try:
        response_data = response.json()
    except ValueError:
        pytest.fail("Response is not valid JSON.")

    expected_keys = ["request_id", "duration_sec", "text", "language", "segments", "pipeline"]
    for key in expected_keys:
        assert key in response_data, f"Response JSON is missing expected key: '{key}'"
    
    assert "separation" in response_data["pipeline"]
    assert "transcription" in response_data["pipeline"]
    assert isinstance(response_data["segments"], list)
    
    print(f"\nSuccessfully transcribed: '{response_data['text']}'")

def test_transcribe_endpoint_invalid_file_format(api_service_is_running):
    """
    Tests the API's response when a non-audio file is uploaded.
    """
    dummy_file_content = b"this is not an audio file"
    files = {"file": ("invalid.txt", dummy_file_content, "text/plain")}

    response = requests.post(API_URL, files=files)

    assert response.status_code == 400
    response_data = response.json()
    assert "detail" in response_data
    assert "invalid file format" in response_data["detail"].lower()