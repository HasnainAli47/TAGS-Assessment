# AI-Powered Audio Transcription Service

This project is a containerized microservice that accepts an audio file, separates vocals from background noise, and returns a high-quality transcription of the speech. It is built with a focus on robustness, scalability, and ease of use.

## Features

-   **Vocal Separation**: Utilizes the Demucs model to isolate speech from background noise before transcription, significantly improving accuracy in noisy environments.
-   **Accurate Transcription**: Employs OpenAI's Whisper model for state-of-the-art speech-to-text conversion.
-   **Robust API**: A clean, documented RESTful API built with FastAPI that accepts various audio formats.
-   **Fully Containerized**: Packaged with Docker and Docker Compose for predictable, one-command deployment.
-   **GPU Acceleration**: Automatically detects and uses a CUDA-enabled GPU for faster processing, while remaining fully functional on CPU-only machines.
-   **CLI Utility**: A simple command-line interface for easy interaction with the service.

## Tech Stack

-   **Runtime**: Python 3.11+
-   **API Framework**: FastAPI with Uvicorn
-   **Containerization**: Docker & Docker Compose
-   **Source Separation**: [Demucs (htdemucs model)](https://github.com/facebookresearch/demucs)
-   **Transcription (ASR)**: [OpenAI Whisper (small model)](https://github.com/openai/whisper)
-   **Audio Processing**: PyTorch, Torchaudio, FFmpeg
-   **Testing**: Pytest, Requests

---

## Getting Started

### Prerequisites

-   [Docker](https://www.docker.com/get-started) and [Docker Compose](https://docs.docker.com/compose/install/) are installed and running.
-   An internet connection is required for the initial build to download models.

### How to Run the Service

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/HasnainAli47/TAGS-Assessment
    cd TAGS-Assessment
    ```

2.  **Build and run the service using Docker Compose:**
    ```bash
    docker-compose up --build -d
    ```
    -   The first build will take several minutes as it needs to download the Demucs and Whisper models into the Docker image.
    -   The `-d` flag runs the container in detached mode (in the background).

3.  **The service is now running!** The API is available at `http://localhost:8000`.
    -   Interactive API documentation (Swagger UI) is available at `http://localhost:8000/docs`.

### How to Stop the Service

To stop and remove the running container, execute:
```bash
docker-compose down
```

---

## Usage

### 1. Using the API Endpoint

You can send a `POST` request to the `/v1/transcribe` endpoint with `multipart/form-data`.

**`curl` Example:**

```bash
curl -X 'POST' \
  'http://localhost:8000/v1/transcribe' \
  -F 'file=@./sample_audio/noisy_speech.mp3' \
  -F 'config={"enable_separation": true}'
```

The response will be a JSON object containing the transcription and other metadata.

### 2. Using the CLI Utility

The `cli.py` script provides a simple way to transcribe a local file.

**Prerequisites:** You need Python and `requests` installed locally (`pip install requests`).

**Example:**
```bash
# Transcribe with vocal separation (default)
python cli.py ./sample_audio/noisy_speech.mp3

# Transcribe without vocal separation
python cli.py ./sample_audio/clean_speech.wav --no-separation
```

---

## Running Tests

To verify that the service is working correctly, you can run the integration tests.

1.  **Ensure the service is running** (use `docker-compose up -d`).

2.  **Install development dependencies:**
    ```bash
    pip install -r requirements-dev.txt
    ```

3.  **Run pytest from the project root:**
    ```bash
    pytest -v
    ```

---

## Architecture Decision Record (ADR)

### Pipeline Stages

The audio processing pipeline is designed for optimal quality and robustness:

`decode -> normalize -> separation -> ASR -> format`

1.  **Decode & Normalize**: We first decode any input audio format (MP3, WAV, etc.) into a raw, standardized format (16kHz, mono, 32-bit float). This is handled by a robust `ffmpeg-python` process, which is more reliable than relying on `torchaudio`'s backend discovery. This standardization ensures all subsequent stages work with a predictable input.
2.  **Separation**: If enabled, the Demucs model is applied to the normalized audio. Performing this step before ASR is crucial for isolating the speech signal from noise, which is the primary driver of transcription accuracy improvement.
3.  **ASR**: The Whisper model transcribes the cleaned (or original) audio waveform.
4.  **Format**: The final output from Whisper is structured into the API's specified JSON response format.

### Model Selection

-   **Separation (Demucs)**: We chose `htdemucs` (Hybrid Transformer Demucs) as it provides excellent, high-fidelity vocal separation. While large, its quality justifies the resource cost for this application's core requirement. Baking it into the Docker image ensures it's always available.
-   **ASR (Whisper)**: We chose the `whisper-small` model as it offers an outstanding balance between transcription accuracy, speed, and resource requirements. It is robust against various accents and languages and provides timestamped segments, which fulfills a core API requirement.

### GPU/CPU Strategy

The service is designed to be "GPU-aware" but not "GPU-dependent".

-   **Auto-Detection**: On startup, the application checks for a CUDA-enabled GPU with `torch.cuda.is_available()`. All models and tensors are then moved to the appropriate device (`cuda` or `cpu`).
-   **Dockerfile Portability**: The `Dockerfile` is based on a standard Python image and does not require a specific CUDA version, making it runnable on any machine with Docker.
-   **Docker Compose for GPU**: The `docker-compose.yml` file includes a commented-out `deploy` section. Users with an NVIDIA GPU and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) can uncomment this section to grant the container direct access to the GPU, significantly accelerating inference.

### Memory and Concurrency Model

-   **Lazy Loading**: Models are not loaded when the application starts. Instead, they are loaded into a global dictionary cache the *first time* they are requested by an API call. This ensures a fast server startup (`uvicorn` is ready in seconds) and efficient memory usage, as models are only loaded if needed.
-   **Per-Process Singleton**: When running with multiple `uvicorn` workers (e.g., via Gunicorn `gunicorn -w 4 -k uvicorn.workers.UvicornWorker`), each worker process will maintain its own singleton instance of the models in memory. This is a standard, effective concurrency model that avoids sharing GPU memory across processes, which can be problematic.

## Project Structure
```
.
├── Dockerfile
├── cli.py
├── docker-compose.yml
├── download_models.py
├── README.md
├── requirements-dev.txt
├── requirements.txt
├── sample_audio
│   ├── clean_speech.wav
│   └── noisy_speech.mp3
├── src
│   ├── api
│   │   └── endpoints.py
│   ├── core
│   │   └── audio_processing.py
│   └── main.py
└── tests
    ├── test_api.py
    └── test_data
        └── hello-48300.mp3
```