import streamlit as st
import time
import torch
import logging
from transformers import pipeline
from news_client import GNewsClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure page first
st.set_page_config(
    page_title="AI News Digest",
    page_icon="🤖",
    layout="centered"
)

class GPUSummarizer:
    def __init__(self):
        self.device = self._get_device()
        self.model_name = "facebook/bart-large-cnn"
        self.summarizer = self._load_model()
        
    def _get_device(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.set_device(0)
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
                return 0
            except Exception as e:
                logger.error(f"GPU Error: {str(e)}")
                return -1
        logger.warning("Using CPU")
        return -1
    
    def _load_model(self):
        try:
            return pipeline(
                "summarization",
                model=self.model_name,
                device=self.device,
                torch_dtype=torch.float16 if self.device == 0 else torch.float32,
                batch_size=4,  # Optimized for RTX 3080's 10GB VRAM
                num_beams=4
            )
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise

class NewsProcessor:
    def __init__(self):
        self.news_client = GNewsClient()
        self.summarizer = GPUSummarizer().summarizer
        
    def _log_gpu_memory(self):
        if torch.cuda.is_available():
            logger.info(
                f"GPU Memory - Allocated: {torch.cuda.memory_allocated()/1024**3:.2f}GB, "
                f"Reserved: {torch.cuda.memory_reserved()/1024**3:.2f}GB"
            )
    
    def process_articles(self, max_articles=15, summary_length=100):
        try:
            articles = self.news_client.get_ai_news(max_articles)
            if not articles:
                return None, "No articles found"
                
            input_texts = [
                f"{article['title']}. {article.get('description', '')}"
                for article in articles
            ]
            
            self._log_gpu_memory()
            summaries = self.summarizer(
                input_texts,
                max_length=summary_length,
                min_length=max(30, summary_length//2),
                truncation=True
            )
            self._log_gpu_memory()
            
            return [
                (article, summary['summary_text'])
                for article, summary in zip(articles, summaries)
            ], None
            
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                return None, "VRAM overload! Reduce batch_size/summary_length"
            return None, f"GPU Error: {str(e)}"
        except Exception as e:
            return None, f"Processing error: {str(e)}"

# Streamlit UI Components
@st.cache_resource
def load_processor():
    return NewsProcessor()

processor = load_processor()

with st.sidebar:
    st.header("Settings ⚙️")
    max_articles = st.slider("Articles to fetch", 5, 30, 15)
    summary_length = st.slider("Summary length (words)", 30, 150, 100)
    if torch.cuda.is_available():
        st.success(f"GPU Active: {torch.cuda.get_device_name(0)}")
    else:
        st.warning("Using CPU")

st.title("AI News Digest 🤖")
st.markdown("Daily AI news summaries powered by GNews API")

if st.button("🔄 Fetch and Summarize Latest AI News"):
    start_time = time.time()
    
    with st.spinner("Processing..."):
        try:
            results, error = processor.process_articles(max_articles, summary_length)
            
            if error:
                st.error(error)
                st.stop()
                
            for article, summary in results:
                with st.container():
                    st.subheader(article["title"])
                    st.caption(f"**{article['source']['name']}** · {article['publishedAt']}")
                    st.markdown(f"📝 **Summary**: {summary}")
                    st.markdown(f"[Read Full Article ↗]({article['url']})")
                    st.divider()
            
            st.success(f"✅ Processed {len(results)} articles in {time.time()-start_time:.1f}s")
        
        except Exception as e:
            st.error(f"Critical error: {str(e)}")