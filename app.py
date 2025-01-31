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
    page_title="AI News Digest Pro",
    page_icon="🤖",
    layout="centered"
)

class AIProcessor:
    def __init__(self):
        self.device = self._get_device()
        self.models = self._load_models()
    
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
    
    def _load_models(self):
        try:
            return {
                "summarizer": pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    device=self.device,
                    torch_dtype=torch.float16 if self.device == 0 else torch.float32,
                    batch_size=4
                ),
                "sentiment": pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=self.device
                ),
                "ner": pipeline(
                    "ner",
                    model="dslim/bert-base-NER",
                    device=self.device,
                    grouped_entities=True
                ),
                "translator": pipeline(
                    "translation",
                    model="facebook/mbart-large-50-many-to-many-mmt",
                    device=self.device,
                    src_lang="en_XX",
                    max_length=512
                ),
                "qa": pipeline(
                    "question-answering",
                    model="deepset/roberta-base-squad2",
                    device=self.device
                )
            }
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise

class EnhancedNewsProcessor:
    def __init__(self):
        self.news_client = GNewsClient()
        self.ai = AIProcessor()
    
    def _log_resources(self):
        if torch.cuda.is_available():
            logger.info(
                f"GPU Memory - Allocated: {torch.cuda.memory_allocated()/1024**3:.2f}GB, "
                f"Reserved: {torch.cuda.memory_reserved()/1024**3:.2f}GB"
            )
    
    def process_article(self, article, target_lang=None):
        try:
            content = f"{article['title']}. {article.get('description', '')}"
            
            results = {
                "summary": self.ai.models["summarizer"](
                    content,
                    max_length=150,
                    min_length=50,
                    truncation=True
                )[0]["summary_text"],
                "sentiment": self.ai.models["sentiment"](content[:512])[0],
                "entities": self._format_entities(
                    self.ai.models["ner"](content[:512])
                )
            }
            
            if target_lang:
                results["translation"] = self.ai.models["translator"](
                    results["summary"][:500],
                    tgt_lang=target_lang,
                    max_length=150
                )[0]["translation_text"]
            
            return results, None
            
        except Exception as e:
            return None, str(e)
    
    def _format_entities(self, entities):
        return {
            entity['word']: entity['entity_group']
            for entity in entities
            if entity['entity_group'] in ['ORG', 'PER', 'LOC']
        }

@st.cache_resource
def load_processor():
    return EnhancedNewsProcessor()

processor = load_processor()

with st.sidebar:
    st.header("AI Settings ⚙️")
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

st.title("AI News Digest Pro 🤖")
st.markdown("Advanced news analysis with multiple AI capabilities")

if st.button("🔄 Analyze Latest AI News"):
    st.session_state.articles = processor.news_client.get_ai_news(max_articles)

if "articles" in st.session_state:
    articles = st.session_state.articles
    
    if not articles:
        st.warning("No articles found. Try again later.")
        st.stop()

    start_time = time.time()
    progress_bar = st.progress(0)
    
    for i, article in enumerate(articles):
        analysis, error = processor.process_article(article, target_lang)
        
        if error:
            st.error(f"Error processing article: {error}")
            continue

        with st.container():
            # Header Section
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

            # AI Analysis
            with st.expander("View AI Analysis"):
                st.markdown(f"**Summary**: {analysis['summary']}")
                
                if target_lang:
                    st.markdown(f"**Translated ({translate_to})**: {analysis['translation']}")
                    
                if analysis['entities']:
                    st.markdown("**Key Entities**:")
                    for entity, group in analysis['entities'].items():
                        st.code(f"{entity} ({group})")

            # Question Answering Form
            with st.form(key=f"qa_form_{article['url']}"):
                user_question = st.text_input(
                    "Ask about this article:", 
                    key=f"qa_input_{article['url']}",
                    placeholder="What technology does this mention?",
                    help="Ask specific questions about the article content"
                )
                
                submitted = st.form_submit_button("Get Answer")
                
                if submitted and user_question:
                    try:
                        context = f"{article['title']}. {article.get('description', '')} {article.get('content', '')}"[:4096]
                        
                        answer = processor.ai.models["qa"](
                            question=user_question,
                            context=context,
                            max_answer_len=100,
                            handle_impossible_answer=True
                        )
                        
                        if answer['score'] > 0.01:
                            st.session_state[f"answer_{article['url']}"] = {
                                'question': user_question,
                                'answer': answer
                            }
                        else:
                            st.warning("The article doesn't contain a clear answer to this question")
                            
                    except Exception as e:
                        st.error(f"Couldn't process question: {str(e)}")

            # Display stored answer if exists
            if f"answer_{article['url']}" in st.session_state:
                answer_data = st.session_state[f"answer_{article['url']}"]
                st.markdown(f"""
                    **Question**: {answer_data['question']}  
                    **Answer**: {answer_data['answer']['answer']}  
                    <small>(Confidence: {answer_data['answer']['score']:.2f})</small>
                """, unsafe_allow_html=True)

            st.markdown(f"[Read Full Article ↗]({article['url']})")
            st.divider()
        
        progress_bar.progress((i + 1) / len(articles))

    st.success(f"✅ Analyzed {len(articles)} articles in {time.time()-start_time:.1f}s")