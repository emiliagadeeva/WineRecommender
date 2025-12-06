from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os
from api import initialize_recommender, get_recommendations
from config import CSV_URL, EMBEDDINGS_URL, CACHE_DIR

app = Flask(__name__, static_folder='static', template_folder='.')
CORS(app)

# Инициализация рекоммендера
recommender = None
wine_data = None

@app.before_first_request
def initialize():
    global recommender, wine_data
    try:
        recommender = initialize_recommender()
        if hasattr(recommender, 'df'):
            wine_data = recommender.df
        print("✅ Recommender initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing recommender: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/wines/list')
def get_wine_list():
    """Получить список всех вин для выбора"""
    try:
        if wine_data is None:
            return jsonify({"error": "Wine data not loaded"}), 500
        
        # Создаем упрощенный список вин
        wines = []
        for idx, row in wine_data.iterrows():
            wine = {
                'id': idx,
                'name': str(row.get('title', f'Wine {idx}')),
                'variety': str(row.get('variety', 'Unknown')),
                'country': str(row.get('country', 'Unknown')),
                'price': float(row.get('price', 0)) if pd.notna(row.get('price')) else 0,
                'rating': float(row.get('points', 85)) if pd.notna(row.get('points')) else 85,
                'description': str(row.get('description', '')) if pd.notna(row.get('description')) else ''
            }
            wines.append(wine)
        
        return jsonify(wines)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/wines/unique-values')
def get_unique_values():
    """Получить уникальные значения для фильтров"""
    try:
        if wine_data is None:
            return jsonify({"error": "Wine data not loaded"}), 500
        
        # Уникальные страны
        countries = []
        if 'country' in wine_data.columns:
            countries = sorted([c for c in wine_data['country'].dropna().unique() if str(c) != 'nan'])
        
        # Уникальные сорта винограда
        varieties = []
        if 'variety' in wine_data.columns:
            varieties = sorted([v for v in wine_data['variety'].dropna().unique() if str(v) != 'nan'])
        
        # Диапазон цен
        min_price = 0
        max_price = 1000
        if 'price' in wine_data.columns:
            valid_prices = wine_data['price'].dropna()
            if len(valid_prices) > 0:
                min_price = float(valid_prices.min())
                max_price = float(valid_prices.max())
        
        return jsonify({
            'countries': countries,
            'varieties': varieties,
            'price_range': {
                'min': min_price,
                'max': max_price
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/recommend/filtered', methods=['POST'])
def recommend_filtered():
    """Рекомендации с фильтрами"""
    try:
        data = request.json
        query = data.get('query', '')
        filters = data.get('filters', {})
        
        if not recommender:
            return jsonify({"error": "Recommender not initialized"}), 500
        
        # Получаем рекомендации
        recommendations, llm_comment = get_recommendations(
            recommender=recommender,
            query=query,
            variety=filters.get('variety'),
            country=filters.get('country'),
            max_price=filters.get('max_price'),
            top_k=20
        )
        
        # Форматируем результат для фронтенда
        formatted_recs = []
        for wine in recommendations:
            formatted_wine = {
                'id': wine.get('id', wine.get('title', '')),
                'name': str(wine.get('title', 'Unknown Wine')),
                'variety': str(wine.get('variety', 'Unknown')),
                'country': str(wine.get('country', 'Unknown')),
                'region': str(wine.get('region_1', wine.get('province', 'Unknown'))),
                'price': float(wine.get('price', 0)) if pd.notna(wine.get('price')) else 0,
                'rating': float(wine.get('points', 85)) if pd.notna(wine.get('points')) else 85,
                'description': str(wine.get('description', '')) if pd.notna(wine.get('description')) else '',
                'winery': str(wine.get('winery', '')) if pd.notna(wine.get('winery')) else '',
                'similarity_score': float(wine.get('similarity_score', 0.5))
            }
            formatted_recs.append(formatted_wine)
        
        return jsonify({
            'recommendations': formatted_recs,
            'llm_comment': llm_comment or "Вот рекомендации на основе вашего запроса и фильтров."
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/recommend/simple', methods=['POST'])
def recommend_simple():
    """Простой поиск по запросу"""
    try:
        data = request.json
        query = data.get('query', '')
        
        if not recommender:
            return jsonify({"error": "Recommender not initialized"}), 500
        
        recommendations, llm_comment = get_recommendations(
            recommender=recommender,
            query=query,
            top_k=20
        )
        
        formatted_recs = []
        for wine in recommendations:
            formatted_wine = {
                'id': wine.get('id', wine.get('title', '')),
                'name': str(wine.get('title', 'Unknown Wine')),
                'variety': str(wine.get('variety', 'Unknown')),
                'country': str(wine.get('country', 'Unknown')),
                'region': str(wine.get('region_1', wine.get('province', 'Unknown'))),
                'price': float(wine.get('price', 0)) if pd.notna(wine.get('price')) else 0,
                'rating': float(wine.get('points', 85)) if pd.notna(wine.get('points')) else 85,
                'description': str(wine.get('description', '')) if pd.notna(wine.get('description')) else '',
                'winery': str(wine.get('winery', '')) if pd.notna(wine.get('winery')) else '',
                'similarity_score': float(wine.get('similarity_score', 0.5))
            }
            formatted_recs.append(formatted_wine)
        
        return jsonify({
            'recommendations': formatted_recs,
            'llm_comment': llm_comment or "Вот рекомендации на основе вашего запроса."
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/recommend/taste', methods=['POST'])
def recommend_by_taste():
    """Рекомендации на основе выбранных вин"""
    try:
        data = request.json
        selected_ids = data.get('selected_wines', [])
        
        if not recommender or wine_data is None:
            return jsonify({"error": "Recommender not initialized"}), 500
        
        # Получаем выбранные вина
        selected_wines = []
        for idx in selected_ids:
            if 0 <= idx < len(wine_data):
                wine = wine_data.iloc[idx].to_dict()
                wine['id'] = idx
                selected_wines.append(wine)
        
        if not selected_wines:
            return jsonify({"error": "No wines selected"}), 400
        
        # Анализ предпочтений
        preference_analysis = analyze_preferences(selected_wines)
        
        # Генерируем запрос на основе предпочтений
        query = generate_preference_query(selected_wines)
        
        # Получаем рекомендации
        recommendations, llm_comment = get_recommendations(
            recommender=recommender,
            query=query,
            top_k=20
        )
        
        # Форматируем результат
        formatted_recs = []
        for wine in recommendations:
            # Пропускаем вина, которые уже выбраны
            if any(w['title'] == wine.get('title') for w in selected_wines):
                continue
                
            formatted_wine = {
                'id': wine.get('id', wine.get('title', '')),
                'name': str(wine.get('title', 'Unknown Wine')),
                'variety': str(wine.get('variety', 'Unknown')),
                'country': str(wine.get('country', 'Unknown')),
                'region': str(wine.get('region_1', wine.get('province', 'Unknown'))),
                'price': float(wine.get('price', 0)) if pd.notna(wine.get('price')) else 0,
                'rating': float(wine.get('points', 85)) if pd.notna(wine.get('points')) else 85,
                'description': str(wine.get('description', '')) if pd.notna(wine.get('description')) else '',
                'winery': str(wine.get('winery', '')) if pd.notna(wine.get('winery')) else '',
                'similarity_score': float(wine.get('similarity_score', 0.5))
            }
            formatted_recs.append(formatted_wine)
        
        return jsonify({
            'recommendations': formatted_recs[:10],  # Ограничиваем 10 рекомендациями
            'llm_comment': llm_comment or f"На основе ваших предпочтений в {len(selected_wines)} винах, вот подходящие варианты.",
            'preference_analysis': preference_analysis
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def analyze_preferences(selected_wines):
    """Анализ предпочтений пользователя"""
    if not selected_wines:
        return {}
    
    # Любимые сорта
    varieties = {}
    for wine in selected_wines:
        variety = wine.get('variety')
        if variety and str(variety) != 'nan':
            varieties[variety] = varieties.get(variety, 0) + 1
    
    favorite_varieties = sorted(
        [{'variety': k, 'count': v} for k, v in varieties.items()],
        key=lambda x: x['count'],
        reverse=True
    )[:5]
    
    # Предпочитаемые страны
    countries = {}
    for wine in selected_wines:
        country = wine.get('country')
        if country and str(country) != 'nan':
            countries[country] = countries.get(country, 0) + 1
    
    preferred_countries = sorted(
        [{'country': k, 'count': v} for k, v in countries.items()],
        key=lambda x: x['count'],
        reverse=True
    )[:5]
    
    # Анализ цен
    prices = []
    for wine in selected_wines:
        price = wine.get('price')
        if price and pd.notna(price):
            prices.append(float(price))
    
    average_price = np.mean(prices) if prices else 0
    min_price = min(prices) if prices else 0
    max_price = max(prices) if prices else 0
    
    return {
        'favorite_varieties': favorite_varieties,
        'preferred_countries': preferred_countries,
        'average_price': round(average_price, 2),
        'price_range': {
            'min': round(min_price, 2),
            'max': round(max_price, 2)
        },
        'total_selected': len(selected_wines)
    }

def generate_preference_query(selected_wines):
    """Генерация текстового запроса на основе предпочтений"""
    if not selected_wines:
        return "quality wine"
    
    # Собираем ключевые слова
    keywords = []
    
    # Самые частые сорта
    varieties = {}
    for wine in selected_wines:
        variety = wine.get('variety')
        if variety:
            varieties[variety] = varieties.get(variety, 0) + 1
    
    top_varieties = sorted(varieties.items(), key=lambda x: x[1], reverse=True)[:2]
    if top_varieties:
        keywords.append(f"{top_varieties[0][0]} wine")
    
    # Страны
    countries = {}
    for wine in selected_wines:
        country = wine.get('country')
        if country:
            countries[country] = countries.get(country, 0) + 1
    
    top_countries = sorted(countries.items(), key=lambda x: x[1], reverse=True)[:1]
    if top_countries:
        keywords.append(f"from {top_countries[0][0]}")
    
    # Стиль (на основе описания)
    style_keywords = ['rich', 'full-bodied', 'light', 'fruity', 'dry', 'sweet']
    description_text = ' '.join([str(w.get('description', '')) for w in selected_wines]).lower()
    
    found_styles = [style for style in style_keywords if style in description_text]
    if found_styles:
        keywords.append(found_styles[0])
    
    return ' '.join(keywords) if keywords else "quality wine"

@app.route('/api/wine/ai-comment', methods=['POST'])
def get_ai_comment():
    """Генерация AI комментария для вина"""
    try:
        data = request.json
        wine = data.get('wine', {})
        
        # Генерируем простой комментарий
        comment_parts = []
        
        if wine.get('variety'):
            comment_parts.append(f"Это прекрасное вино сорта {wine['variety']}.")
        
        if wine.get('country'):
            comment_parts.append(f"Произведено в {wine['country']}.")
        
        if wine.get('price') and wine['price'] > 50:
            comment_parts.append("Отличное соотношение цены и качества.")
        elif wine.get('price'):
            comment_parts.append("Доступный вариант с хорошими характеристиками.")
        
        comment = ' '.join(comment_parts) if comment_parts else "Интересное вино, достойное внимания."
        
        return jsonify({'comment': comment})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "wines_loaded": wine_data is not None})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
