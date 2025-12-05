from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Инициализация приложения
app = FastAPI(
    title="Passport OCR Service API",
    description="Сервис для загрузки, обработки и распознавания изображений паспортов РФ",
    version="0.1.0"
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене заменить на конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Passport OCR Service API", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
