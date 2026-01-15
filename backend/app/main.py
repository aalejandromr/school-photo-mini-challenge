from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import routes

app = FastAPI(title="Shadow Generator API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(routes.router, prefix="/api", tags=["shadow"])

@app.get("/")
async def root():
    return {"message": "Shadow Generator API", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {"status": "healthy"}
