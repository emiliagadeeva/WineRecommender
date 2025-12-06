import os

# Настройки Google Drive
CSV_URL = "https://drive.google.com/file/d/18mwRZRlY3f6M6nN6VmiHKzDAAZxfEF7A"
EMBEDDINGS_URL = "https://drive.google.com/file/d/1w7to6R0qf2h0-yBXwJl62-pRWN5LP60I"

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
