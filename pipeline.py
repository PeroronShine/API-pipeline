import pandas as pd
import requests
import uuid
import json
import ssl
import os
import urllib3
from dotenv import load_dotenv

# --- НАСТРОЙКИ ---

# Загружаем переменные из файла .env
load_dotenv()

# Получаем ключ из .env файла
API_KEY = os.getenv("GIGACHAT_API_KEY")

# Проверка: есть ли ключ?
if not API_KEY:
    raise ValueError("❌ ОШИБКА: Не найден API Key. Убедитесь, что создали файл .env и вставили туда ключ.")

OUTPUT_FILE = "results.csv"

# --- УДАЛЕНИЕ ПРЕДУПРЕЖДЕНИЙ (Чтобы консоль была чистой) ---
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- ФУНКЦИЯ ОБХОДА SSL ОШИБОК (Для Wi-Fi университета) ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def get_token():
    """Получает токен доступа к GigaChat API"""
    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    payload = "scope=GIGACHAT_API_PERS"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
        "RqUID": str(uuid.uuid4()),
        "Authorization": f"Basic {API_KEY}"
    }
    # verify=False отключает проверку сертификатов (для работы в корпоративных сетях)
    response = requests.post(url, data=payload, headers=headers, verify=False)
    if response.status_code == 200:
        return response.json().get("access_token")
    else:
        print(f"Ошибка получения токена: {response.text}")
        return None

def analyze_sentiment(text):
    """Отправляет текст в LLM и получает JSON с анализом"""
    token = get_token()
    if not token:
        return None

    url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
        "RqUID": str(uuid.uuid4())
    }
    
    prompt = f"""
    Проанализируй тональность следующего отзыва на русском языке.
    Ответь ТОЛЬКО валидным JSON объектом без пояснений.
    Формат: {{"sentiment": "positive/negative/neutral", "reason": "краткое объяснение"}}.
    
    Отзыв: "{text}"
    """

    body = {
        "model": "GigaChat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5
    }

    response = requests.post(url, json=body, headers=headers, verify=False)
    
    if response.status_code == 200:
        result = response.json()['choices'][0]['message']['content']
        try:
            # Если нейросеть ответила блоком кода ```json ... ```, чистим его
            if "```" in result:
                result = result.split("```")[1]
                if "json" in result:
                    result = result.replace("json", "", 1)
            return json.loads(result.strip())
        except:
            print(f"Ошибка парсинга JSON: {result}")
            return {"sentiment": "error", "reason": result}
    else:
        print(f"Ошибка API: {response.status_code}")
        return {"sentiment": "error", "reason": response.text}

# --- ОСНОВНОЙ ПИПЛАЙН ---
def main():
    print("1. Читаю входные данные (reviews.csv)...")
    # Считываем данные (указали encoding для поддержки кириллицы)
    df = pd.read_csv("reviews.csv", encoding='cp1251')
    
    results = []
    
    print("2. Отправляю данные в LLM (GigaChat API)...")
    for index, row in df.iterrows():
        review_text = row['text']
        print(f"   - Обработка отзыва #{index + 1}...")
        
        ai_response = analyze_sentiment(review_text)
        
        if ai_response:
            results.append({
                "id": row['id'],
                "original_text": review_text,
                "llm_sentiment": ai_response.get("sentiment", "unknown"),
                "llm_reason": ai_response.get("reason", "")
            })
        else:
            print("   ⚠️ Пропуск из-за ошибки.")

    print("3. Сохраняю результат...")
    results_df = pd.DataFrame(results)
    # encoding='utf-8-sig' нужен, чтобы Excel корректно открывал русские буквы
    results_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ Готово! Результат сохранен в файл: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()