# Shadow Generator Backend

FastAPI backend for realistic shadow generation.

## Technology Stack

- **Framework**: FastAPI (Python web framework)
- **Server**: Uvicorn (ASGI server for running FastAPI)
- **Image Processing**: OpenCV, Pillow, rembg
- **Other**: NumPy for array operations

## Setup

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Make sure you're in the `backend` directory, then run the FastAPI server using uvicorn:
```bash
# Important: Run this command from the backend directory
cd backend  # If not already there
uvicorn app.main:app --reload --port 8000
```

Alternatively, you can run from any directory by setting PYTHONPATH:
```bash
PYTHONPATH=/path/to/backend uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

FastAPI automatically provides interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
