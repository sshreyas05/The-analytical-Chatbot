import os
import streamlit as st
import yfinance as yf
import faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer
from newspaper import Article
from yfinance.exceptions import YFRateLimitError

# ================= CONFIG =================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHAT_MODEL = "llama-3.3-70b-versatile"

VECTOR_DIM = 384
TOP_K = 5

# ================= PAGE SETUP =================
st.set_page_config(
    page_title="Stock RAG Chatbot",
    page_icon="📊",
    layout="wide"
)

st.title("📊 WHY-aware Stock RAG Chatbot")
st.caption("Explains WHAT happened to a stock and WHY (using price + news)")

# ================= DATA LOADERS =================
@st.cache_data(ttl=3600)  # cache for 1 hour
def load_stock_data(ticker, period="6mo"):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)

        documents = []
        for date, row in hist.iterrows():
            documents.append(
                f"Date: {date.date()}, "
                f"Open: {row['Open']:.2f}, "
                f"High: {row['High']:.2f}, "
                f"Low: {row['Low']:.2f}, "
                f"Close: {row['Close']:.2f}, "
                f"Volume: {int(row['Volume'])}"
            )
        return documents

    except YFRateLimitError:
        return None


def load_stock_news(ticker, max_articles=10):
    urls = [f"https://finance.yahoo.com/quote/{ticker}/news"]
    articles = []

    for url in urls:
        try:
            article = Article(url)
            article.download()
            article.parse()
            text = article.text.strip()

            if len(text) > 300:
                articles.append(f"News Article:\n{text[:2000]}")
        except:
            pass

    return articles[:max_articles]


# ================= EMBEDDINGS =================
@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL)

embed_model = load_embedder()

def embed_texts(texts):
    return embed_model.encode(texts, convert_to_numpy=True)


# ================= VECTOR STORE =================
class VectorStore:
    def __init__(self):
        self.index = faiss.IndexFlatL2(VECTOR_DIM)
        self.documents = []

    def add(self, embeddings, docs):
        self.index.add(embeddings.astype("float32"))
        self.documents.extend(docs)

    def search(self, query_embedding, k):
        _, I = self.index.search(
            query_embedding.reshape(1, -1).astype("float32"), k
        )
        return [self.documents[i] for i in I[0]]


# ================= RAG CHATBOT =================
client = Groq(api_key=GROQ_API_KEY)

class RAGChatbot:
    def __init__(self, price_store, news_store):
        self.price_store = price_store
        self.news_store = news_store

    def answer(self, question):
        q_emb = embed_texts([question])[0]

        price_ctx = self.price_store.search(q_emb, TOP_K)
        news_ctx = self.news_store.search(q_emb, TOP_K)

        context = (
            "=== PRICE DATA ===\n" + "\n".join(price_ctx) +
            "\n\n=== NEWS DATA ===\n" + "\n".join(news_ctx)
        )

        prompt = f"""
You are a financial analyst.

Rules:
- Use PRICE DATA to describe WHAT happened
- Use NEWS DATA to explain WHY it happened
- If the reason is unclear, say so explicitly
- Do NOT hallucinate causes

Context:
{context}

Question:
{question}
"""

        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )

        return response.choices[0].message.content


# ================= SIDEBAR =================
st.sidebar.header("⚙️ Settings")

ticker = st.sidebar.text_input(
    "Stock Ticker",
    value="AAPL",
    help="Enter a valid Yahoo Finance ticker (e.g. AAPL, TSLA)"
).upper().strip()

load_button = st.sidebar.button("📥 Load Stock Data")

# ================= STATE =================
if "bot" not in st.session_state:
    st.session_state.bot = None

# ================= LOAD DATA =================
if load_button:
    with st.spinner("Fetching price data..."):
     price_docs = load_stock_data(ticker)

if price_docs is None:
    st.warning(
        "⚠️ Yahoo Finance rate limit hit.\n\n"
        "Please wait a few minutes and try again.\n"
        "This is a Yahoo-side limitation, not an app error."
    )
    st.stop()

if not price_docs:
    st.error("❌ No price data found. Check the ticker symbol.")
    st.stop()

else:
        with st.spinner("Fetching news..."):
            news_docs = load_stock_news(ticker)

        with st.spinner("Embedding & indexing..."):
            price_store = VectorStore()
            news_store = VectorStore()

            price_store.add(embed_texts(price_docs), price_docs)

            if news_docs:
                news_store.add(embed_texts(news_docs), news_docs)
            else:
                news_store.add(
                    embed_texts(["No relevant news found."]),
                    ["No relevant news found."]
                )

            st.session_state.bot = RAGChatbot(price_store, news_store)

        st.success(f"✅ Data loaded for {ticker}")

# ================= CHAT =================
if st.session_state.bot:
    st.subheader("💬 Ask a Question")

    question = st.text_input(
        "Your question",
        placeholder="Why did the stock go down last week?"
    )

    if st.button("Ask"):
        if question.strip():
            with st.spinner("Analyzing..."):
                answer = st.session_state.bot.answer(question)
            st.markdown("### 🧠 Answer")
            st.write(answer)
        else:
            st.warning("Please enter a question.")
else:
    st.info("👈 Enter a ticker and click **Load Stock Data** to begin.")
