import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import requests
from config import *

class WineRecommender:
    def __init__(self, csv_path=None, embeddings_path=None, model_name=None):
        self.df = None
        self.model = None
        self.wine_embeddings = None
        self.wine_descriptions = None
        
        # Загружаем данные
        if csv_path and os.path.exists(csv_path):
            self.df = pd.read_csv(csv_path)
        elif CSV_URL:
            self.df = self.load_from_url(CSV_URL, 'wines.csv')
        
        if model_name:
            self.model = SentenceTransformer(model_name)
        
        # Загружаем эмбеддинги
        if embeddings_path and os.path.exists(embeddings_path):
            self.load_embeddings(embeddings_path)
        elif EMBEDDINGS_URL and self.df is not None:
            self.load_embeddings_from_url()
    
    def load_from_url(self, url, filename):
        """Загрузка CSV с Google Drive"""
        cache_path = os.path.join(CACHE_DIR, filename)
        
        if os.path.exists(cache_path):
            print(f"📂 Loading cached {filename}")
            return pd.read_csv(cache_path)
        
        print(f"🌐 Downloading {filename} from Google Drive...")
        response = requests.get(url)
        response.raise_for_status()
        
        with open(cache_path, 'wb') as f:
            f.write(response.content)
        
        print(f"✅ Saved to {cache_path}")
        return pd.read_csv(cache_path)
    
    def load_embeddings(self, path):
        """Загрузка эмбеддингов"""
        print(f"🔄 Loading embeddings from {path}")
        with open(path, 'rb') as f:
            saved_data = pickle.load(f)
            self.wine_embeddings = saved_data['embeddings']
            self.wine_descriptions = saved_data['descriptions']
        print(f"✅ Loaded {len(self.wine_embeddings)} embeddings")
    
    def load_embeddings_from_url(self):
        """Загрузка эмбеддингов с Google Drive"""
        cache_path = os.path.join(CACHE_DIR, 'embeddings.pkl')
        
        if os.path.exists(cache_path):
            self.load_embeddings(cache_path)
            return
        
        print("🌐 Downloading embeddings from Google Drive...")
        response = requests.get(EMBEDDINGS_URL)
        response.raise_for_status()
        
        with open(cache_path, 'wb') as f:
            f.write(response.content)
        
        self.load_embeddings(cache_path)
    
    def create_wine_descriptions(self):
        """Создание текстовых описаний вин"""
        if self.df is None:
            return []
        
        descriptions = []
        for _, row in self.df.iterrows():
            desc_parts = []
            
            # Основная информация
            for field in ['country', 'province', 'region_1', 'variety', 'winery', 'title']:
                if pd.notna(row.get(field)) and row.get(field):
                    desc_parts.append(f"{field}: {row[field]}")
            
            # Характеристики
            if pd.notna(row.get('description')) and row.get('description'):
                desc_parts.append(f"Description: {row['description']}")
            
            # Цена
            if pd.notna(row.get('price')) and row.get('price'):
                desc_parts.append(f"Price: ${row['price']}")
            
            description = ". ".join(desc_parts)
            descriptions.append(description)
        
        return descriptions
    
    def retrieve(self, query, top_k=20):
        """Векторный поиск"""
        if self.model is None or self.wine_embeddings is None:
            return []
        
        query_embedding = self.model.encode([query])[0]
        similarities = cosine_similarity([query_embedding], self.wine_embeddings)[0]
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] < SIMILARITY_THRESHOLD:
                continue
            
            wine_data = self.df.iloc[idx].to_dict()
            wine_data['id'] = idx
            wine_data['similarity_score'] = float(similarities[idx])
            results.append(wine_data)
        
        return results
    
    def search_by_filters(self, query=None, variety=None, country=None, max_price=None, top_k=10):
        """Поиск с фильтрами"""
        # Сначала семантический поиск
        if query:
            results = self.retrieve(query, top_k=50)
        else:
            results = [self.df.iloc[i].to_dict() for i in range(min(100, len(self.df)))]
            for i, result in enumerate(results):
                result['id'] = i
                result['similarity_score'] = 0.5
        
        # Применяем фильтры
        filtered = []
        for wine in results:
            if variety and pd.notna(wine.get('variety')):
                if variety.lower() not in str(wine['variety']).lower():
                    continue
            
            if country and pd.notna(wine.get('country')):
                if country.lower() not in str(wine['country']).lower():
                    continue
            
            if max_price and pd.notna(wine.get('price')):
                if wine['price'] > max_price:
                    continue
            
            filtered.append(wine)
        
        return filtered[:top_k]

def initialize_recommender():
    """Инициализация рекоммендера"""
    print("🚀 Initializing Wine Recommender...")
    
    # Проверяем локальные файлы
    csv_path = LOCAL_CSV_PATH if os.path.exists(LOCAL_CSV_PATH) else None
    embeddings_path = LOCAL_EMBEDDINGS_PATH if os.path.exists(LOCAL_EMBEDDINGS_PATH) else None
    
    recommender = WineRecommender(
        csv_path=csv_path,
        embeddings_path=embeddings_path,
        model_name=EMBEDDING_MODEL
    )
    
    # Если нет эмбеддингов, создаем их
    if recommender.wine_embeddings is None and recommender.df is not None:
        print("🔄 Creating embeddings...")
        recommender.wine_descriptions = recommender.create_wine_descriptions()
        recommender.wine_embeddings = recommender.model.encode(
            recommender.wine_descriptions,
            show_progress_bar=True
        )
        
        # Сохраняем эмбеддинги
        cache_path = os.path.join(CACHE_DIR, 'embeddings.pkl')
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'embeddings': recommender.wine_embeddings,
                'descriptions': recommender.wine_descriptions
            }, f)
        print(f"✅ Saved embeddings to {cache_path}")
    
    print(f"✅ Recommender initialized with {len(recommender.df) if recommender.df else 0} wines")
    return recommender

def get_recommendations(recommender, query="", variety=None, country=None, max_price=None, top_k=20):
    """Получение рекомендаций"""
    if not recommender or recommender.df is None:
        return [], "Recommender not available"
    
    try:
        # Поиск с фильтрами
        results = recommender.search_by_filters(
            query=query,
            variety=variety,
            country=country,
            max_price=max_price,
            top_k=top_k
        )
        
        # Генерируем простой комментарий LLM
        if results:
            llm_comment = generate_simple_llm_comment(query, results[:3])
        else:
            llm_comment = "К сожалению, по вашему запросу ничего не найдено. Попробуйте изменить параметры поиска."
        
        return results, llm_comment
    
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        return [], f"Ошибка при поиске: {str(e)}"

def generate_simple_llm_comment(query, top_wines):
    """Генерация простого комментария без LLM"""
    if not top_wines:
        return ""
    
    comment_parts = []
    
    # Информация о запросе
    if query:
        comment_parts.append(f"По вашему запросу '{query}'")
    
    # Информация о топовых винах
    if len(top_wines) >= 3:
        varieties = set()
        countries = set()
        prices = []
        
        for wine in top_wines[:3]:
            if wine.get('variety'):
                varieties.add(str(wine['variety']))
            if wine.get('country'):
                countries.add(str(wine['country']))
            if wine.get('price'):
                prices.append(float(wine['price']))
        
        if varieties:
            comment_parts.append(f"нашлись вина сортов: {', '.join(list(varieties)[:3])}")
        
        if countries:
            comment_parts.append(f"из стран: {', '.join(list(countries)[:3])}")
        
        if prices:
            avg_price = sum(prices) / len(prices)
            comment_parts.append(f"средняя цена: ${avg_price:.2f}")
    
    comment = " ".join(comment_parts) + ". Рекомендуем обратить внимание на предложенные варианты!"
    return comment
