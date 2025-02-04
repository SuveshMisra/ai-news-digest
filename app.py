import streamlit as st
import time
import torch
import logging
from transformers import pipeline
from news_client import NewsClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure page first
st.set_page_config(
    page_title="News Analyzer Pro",
    page_icon="📰",
    layout="centered"
)

class NewsProcessor:
    def __init__(self):
        self.news_client = NewsClient()
        self.models = self._load_models()
    
    def _load_models(self):
        device = 0 if torch.cuda.is_available() else -1
        return {
            "summarizer": pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=device,
                torch_dtype=torch.float16 if device == 0 else torch.float32
            ),
            "sentiment": pipeline("sentiment-analysis"),
            "translator": pipeline("translation", model="facebook/mbart-large-50-many-to-many-mmt"),
            "qa": pipeline("question-answering")
        }

    def process_article(self, article, target_lang=None):
        try:
            content = f"{article['title']}. {article.get('description', '')}"
            
            results = {
                "summary": self.models["summarizer"](
                    content,
                    max_length=150,
                    min_length=50,
                    truncation=True
                )[0]["summary_text"],
                "sentiment": self.models["sentiment"](content[:512])[0]
            }
            
            if target_lang:
                results["translation"] = self.models["translator"](
                    results["summary"][:500],
                    src_lang="en_XX",
                    tgt_lang=target_lang
                )[0]["translation_text"]
            
            return results, None
            
        except Exception as e:
            return None, str(e)

@st.cache_resource
def load_processor():
    return NewsProcessor()

processor = load_processor()

# User input for topic
with st.sidebar:
    st.header("News Settings ⚙️")
    news_topic = st.text_input("Enter news topic to analyze", "artificial intelligence")
    max_articles = st.slider("Articles to fetch", 5, 30, 10)
    translate_to = st.selectbox("Translate summaries to", 
                              ["None", "Spanish", "French", "German", "Polish"])
    
    languages = {
        "Spanish": "es_XX", 
        "French": "fr_XX",
        "German": "de_DE",
        "Polish": "pl_PL"
    }
    target_lang = languages[translate_to] if translate_to != "None" else None

st.title(f"{news_topic.capitalize()} News Analyzer 📰")
st.markdown("Get AI-powered insights on any news topic")

if st.button("🔄 Analyze Latest News"):
    start_time = time.time()
    
    with st.spinner(f"Gathering {news_topic} news..."):
        try:
            articles = processor.news_client.get_news(news_topic, max_articles)
            
            if not articles:
                st.warning("No articles found. Try a different topic or try again later.")
                st.stop()

            progress_bar = st.progress(0)
            
            for i, article in enumerate(articles):
                analysis, error = processor.process_article(article, target_lang)
                
                if error:
                    st.error(f"Error processing article: {error}")
                    continue

                with st.container():
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.subheader(article["title"])
                        st.caption(f"**{article['source']['name']}** · {article['publishedAt']}")
                    
                    with col2:
                        st.markdown(f"""
                        <div style="font-size:0.9em; color:#666; margin-top:0.8rem">
                            <b>Sentiment</b><br>
                            {analysis['sentiment']['label']}  
                            <small>({analysis['sentiment']['score']:.2f})</small>
                        </div>
                        """, unsafe_allow_html=True)

                    with st.expander("View Analysis"):
                        st.markdown(f"**Summary**: {analysis['summary']}")
                        
                        if target_lang:
                            st.markdown(f"**Translated ({translate_to})**: {analysis['translation']}")

                    st.markdown(f"[Read Full Article ↗]({article['url']})")
                    st.divider()

                progress_bar.progress((i + 1) / len(articles))

            st.success(f"✅ Analyzed {len(articles)} {news_topic} articles in {time.time()-start_time:.1f}s")

        except Exception as e:
            st.error(f"Critical error: {str(e)}")