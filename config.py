import os

# Настройки Google Drive
CSV_URL = "https://drive.google.com/uc?export=download&id=YOUR_CSV_FILE_ID"
EMBEDDINGS_URL = "https://drive.google.com/uc?export=download&id=YOUR_PKL_FILE_ID"

# Или локальные файлы для разработки
LOCAL_CSV_PATH = None
LOCAL_EMBEDDINGS_PATH = None

# Директория для кэша
CACHE_DIR = "data"
os.makedirs(CACHE_DIR, exist_ok=True)

# Модель для эмбеддингов
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Настройки рекомендаций
DEFAULT_TOP_K = 20
SIMILARITY_THRESHOLD = 0.3
