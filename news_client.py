import requests
import os
import time
from dotenv import load_dotenv

load_dotenv()

class NewsClient:
    def __init__(self):
        self.api_key = os.getenv("GNEWS_API_KEY")
        self.base_url = "https://gnews.io/api/v4/search"
    
    def get_news(self, topic="artificial intelligence", max_articles=15):
        params = {
            "q": topic,
            "lang": "en",
            "max": max_articles,
            "in": "title,description",
            "sortby": "publishedAt",
            "apikey": self.api_key
        }
        
        retries = 3
        for attempt in range(retries):
            try:
                response = requests.get(self.base_url, params=params, timeout=15)
                response.raise_for_status()
                return response.json().get("articles", [])
            
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    wait_time = 2 ** (attempt + 1)
                    time.sleep(wait_time)
                    continue
                print(f"HTTP Error: {str(e)}")
                return []
            
            except Exception as e:
                print(f"API Error: {str(e)}")
                if attempt == retries - 1:
                    raise
                time.sleep(1)
        
        return []