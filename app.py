import os
import streamlit as st
import requests
from datetime import datetime, timedelta
import faiss
import numpy as np
import pandas as pd
from groq import Groq
from sentence_transformers import SentenceTransformer
from newspaper import Article


# ================= CONFIG =================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHAT_MODEL = "llama-3.3-70b-versatile"

VECTOR_DIM = 384
TOP_K = 3

# ================= PAGE SETUP =================
st.set_page_config(
    page_title="WHY-aware Stock RAG Chatbot",
    page_icon="📊",
    layout="wide"
)

st.title("📊 WHY-aware Stock RAG Chatbot")
st.caption("Explains WHAT happened to a stock and WHY (using price + news)")

# ================= SESSION STATE =================
if "bot" not in st.session_state:
    st.session_state.bot = None

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

# ================= DATA LOADERS =================
FINNHUB_API_KEY = os.getenv("d5nq9ehr01qma2b5n4mgd5nq9ehr01qma2b5n4n0")

@st.cache_data(ttl=900)  # cache for 15 minutes
@st.cache_data(ttl=300)
def load_stock_data(ticker):
    """
    Uses Finnhub QUOTE endpoint (cloud-safe).
    Builds a minimal price context instead of full candles.
    """
    url = (
        "https://finnhub.io/api/v1/quote"
        f"?symbol={ticker}&token={FINNHUB_API_KEY}"
    )

    r = requests.get(url, timeout=10)
    data = r.json()

    if "c" not in data or data["c"] == 0:
        return None

    documents = [
        f"Current Price: {data['c']}",
        f"Open: {data['o']}",
        f"High: {data['h']}",
        f"Low: {data['l']}",
        f"Previous Close: {data['pc']}"
    ]

    return documents

    for i in range(len(data["t"])):
        date = datetime.utcfromtimestamp(data["t"][i]).date()
        documents.append(
            f"Date: {date}, "
            f"Open: {data['o'][i]:.2f}, "
            f"High: {data['h'][i]:.2f}, "
            f"Low: {data['l'][i]:.2f}, "
            f"Close: {data['c'][i]:.2f}, "
            f"Volume: {int(data['v'][i])}"
        )

    return documents


@st.cache_data(ttl=900)
def load_price_dataframe(ticker):
    end = datetime.utcnow()
    start = end - timedelta(days=40)

    url = (
        "https://finnhub.io/api/v1/stock/candle"
        f"?symbol={ticker}"
        f"&resolution=D"
        f"&from={int(start.timestamp())}"
        f"&to={int(end.timestamp())}"
        f"&token={FINNHUB_API_KEY}"
    )

    r = requests.get(url, timeout=10)
    data = r.json()

    if data.get("s") != "ok":
        return None

    df = pd.DataFrame({
        "Date": [datetime.utcfromtimestamp(t).date() for t in data["t"]],
        "Close": data["c"]
    })

    return df

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

@st.cache_data(ttl=900)  # refresh every 15 minutes
def load_stock_news(ticker, max_articles=3):
    """
    Fetch near real-time company news from Finnhub
    """
    today = datetime.utcnow().date()
    past = today - timedelta(days=7)

    url = (
        "https://finnhub.io/api/v1/company-news"
        f"?symbol={ticker}"
        f"&from={past}"
        f"&to={today}"
        f"&token={FINNHUB_API_KEY}"
    )

    try:
        r = requests.get(url, timeout=10)
        news = r.json()

        if not isinstance(news, list):
            return []

        docs = []
        for article in news[:max_articles]:
            docs.append(
                f"Headline: {article.get('headline')}\n"
                f"Source: {article.get('source')}\n"
                f"Summary: {article.get('summary')}"
            )

        return docs

    except Exception:
        return []


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
        )[:4000]  # hard safety cap

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

# ================= LOAD DATA =================
if load_button:
    with st.spinner("Fetching price data..."):
        price_docs = load_stock_data(ticker)

    if price_docs is None:
        st.error(
            "❌ Unable to fetch live price data.\n\n"
            "Please check the ticker symbol or try again later."
        )
        st.stop()

    if not price_docs:
        st.error("❌ No price data found. Check the ticker symbol.")
        st.stop()

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

    # -------- PRICE CHART (LAST 30 DAYS) --------
    st.subheader("📈 Price movement (last 30 days)")
    price_df = load_price_dataframe(ticker)

    if price_df is not None and not price_df.empty:
        st.line_chart(
            price_df.set_index("Date")["Close"],
            height=300
        )
    else:
        st.info("No chart data available.")


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

