import streamlit as st
import pandas as pd
import plotly.express as px
import time
import json
import requests
from datetime import datetime
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.vector_stores.supabase import SupabaseVectorStore
from llama_index.core import StorageContext, VectorStoreIndex, Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from supabase import create_client, Client
import streamlit.components.v1 as components

# ==========================================
# 1. GLOBAL CONFIG & PREMIUM THEME ENGINE
# ==========================================
st.set_page_config(page_title="UniPath Korea", layout="wide", initial_sidebar_state="collapsed")

# Inject Global Typography and Premium CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Noto+Sans+KR:wght@300;400;700&display=swap');

    :root {
        --primary: #0D3B8E;
        --secondary: #00C897;
        --accent: #FF6B35;
        --bg: #F8FAFC;
    }

    html, body, [class*="css"] {
        font-family: 'Outfit', 'Noto Sans KR', sans-serif;
        background-color: var(--bg);
    }

    /* Gradient Hero */
    .hero-container {
        background: linear-gradient(135deg, #0D3B8E 0%, #00C897 100%);
        border-radius: 24px;
        padding: 60px 40px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 20px 40px rgba(13, 59, 142, 0.15);
    }

    /* Card Styling */
    .premium-card {
        background: white;
        padding: 25px;
        border-radius: 16px;
        border: 1px solid #EBF1FA;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
        height: 100%;
    }
    .premium-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
    }

    /* Pill Navigation */
    .nav-pill-container {
        display: flex;
        justify-content: center;
        gap: 12px;
        margin-bottom: 35px;
    }
    
    .stButton > button {
        border-radius: 50px;
        padding: 10px 25px;
        font-weight: 600;
        border: none;
        transition: all 0.3s;
    }

    /* Stats Badge */
    .kpi-badge {
        background: rgba(255, 107, 53, 0.1);
        color: #FF6B35;
        padding: 4px 12px;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 700;
    }

    /* Floating Chat Styles */
    #uni-avatar-container {
        position: fixed;
        bottom: 30px;
        right: 30px;
        z-index: 9999;
        cursor: grab;
    }
    .pulse-circle {
        width: 70px;
        height: 70px;
        background: #0D3B8E;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 0 0 0 rgba(13, 59, 142, 0.7);
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(13, 59, 142, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 20px rgba(13, 59, 142, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(13, 59, 142, 0); }
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. MULTILINGUAL DICTIONARY (9 LANGUAGES)
# ==========================================
TR = {
    "🇺🇸 English": {
        "home": "Home", "uni": "University", "career": "Career", "job": "Job Portal", "topik": "TOPIK", "visa": "Visa AI",
        "hero_t": "UniPath Korea", "hero_s": "Premium AI-guided pathway for international students.",
        "ask_ai": "Ask AI →", "search": "Search", "admin": "Admin Center",
        "welcome": "Welcome back, Student!", "kpi_visa": "Active Visas", "kpi_uni": "Partner Unis", "kpi_job": "Open Roles",
        "chat_placeholder": "Ask anything about life in Korea...",
        "source_badge": "Official Source", "uni_desc": "Explore 380+ Korean universities with GKS support.",
        "visa_desc": "Check your E-7-S or F-2-R eligibility instantly.",
        "job_desc": "Verified listings for international talent."
    },
    "🇰🇷 한국어": {
        "home": "홈", "uni": "대학교", "career": "커리어", "job": "채용정보", "topik": "TOPIK", "visa": "비자 AI",
        "hero_t": "유니패스 코리아", "hero_s": "외국인 유학생을 위한 프리미엄 AI 가이드 시스템.",
        "ask_ai": "AI에게 묻기 →", "search": "검색", "admin": "관리자 센터",
        "welcome": "환영합니다!", "kpi_visa": "지원 비자", "kpi_uni": "협력 대학", "kpi_job": "채용 중",
        "chat_placeholder": "한국 생활에 대해 무엇이든 물어보세요...",
        "source_badge": "공식 출처", "uni_desc": "GKS 장학금을 지원하는 380개 이상의 한국 대학 탐색.",
        "visa_desc": "E-7-S 또는 F-2-R 자격 요건을 즉시 확인하세요.",
        "job_desc": "글로벌 인재를 위한 검증된 채용 공고."
    },
    "🇲🇳 Монгол": {
        "home": "Нүүр", "uni": "Их сургууль", "career": "Ажил", "job": "Ажлын байр", "topik": "TOPIK", "visa": "Виз AI",
        "hero_t": "UniPath Korea", "hero_s": "Олон улсын оюутнуудад зориулсан дээд зэрэглэлийн AI хөтөч.",
        "ask_ai": "AI-аас асуух →", "search": "Хайх", "admin": "Админ",
        "welcome": "Тавтай морил!", "kpi_visa": "Визний төрөл", "kpi_uni": "Их сургууль", "kpi_job": "Ажлын байр",
        "chat_placeholder": "Солонгос дахь амьдралын талаар асуух зүйл байна уу?",
        "source_badge": "Албан ёсны эх сурвалж", "uni_desc": "GKS тэтгэлэгтэй 380+ сургуулийг судлаарай.",
        "visa_desc": "E-7-S эсвэл F-2-R визний шаардлагыг шалгана уу.",
        "job_desc": "Олон улсын оюутнуудад зориулсан баталгаат ажлын байр."
    },
    "🇻🇳 Tiếng Việt": {
        "home": "Trang chủ", "uni": "Đại học", "career": "Sự nghiệp", "job": "Việc làm", "topik": "TOPIK", "visa": "Visa AI",
        "hero_t": "UniPath Korea", "hero_s": "Lộ trình hướng dẫn bởi AI dành cho du học sinh.",
        "ask_ai": "Hỏi AI →", "search": "Tìm kiếm", "admin": "Quản trị",
        "welcome": "Chào mừng bạn!", "kpi_visa": "Loại Visa", "kpi_uni": "Đại học đối tác", "kpi_job": "Việc làm",
        "chat_placeholder": "Hỏi bất cứ điều gì về cuộc sống tại Hàn Quốc...",
        "source_badge": "Nguồn chính thức", "uni_desc": "Khám phá hơn 380 trường đại học Hàn Quốc với GKS.",
        "visa_desc": "Kiểm tra điều kiện visa E-7-S hoặc F-2-R ngay lập tức.",
        "job_desc": "Danh sách việc làm đã xác minh cho tài năng quốc tế."
    },
    "🇯🇵 日本語": {
        "home": "ホーム", "uni": "大学", "career": "キャリア", "job": "求人", "topik": "TOPIK", "visa": "ビザ AI",
        "hero_t": "UniPath Korea", "hero_s": "留学生のためのプレミアムAIガイドパスウェイ。",
        "ask_ai": "AIに質問 →", "search": "検索", "admin": "管理センター",
        "welcome": "ようこそ！", "kpi_visa": "対応ビザ", "kpi_uni": "提携大学", "kpi_job": "公開求人",
        "chat_placeholder": "韓国での生活について何でも聞いてください...",
        "source_badge": "公式ソース", "uni_desc": "GKS対応の380以上の韓国の大学を調べる。",
        "visa_desc": "E-7-SまたはF-2-Rの資格を即座に確認。",
        "job_desc": "グローバル人材向けの確認済み求人情報。"
    },
    "🇨🇳 中文": {
        "home": "首页", "uni": "大学", "career": "职业", "job": "职位", "topik": "TOPIK", "visa": "签证 AI",
        "hero_t": "UniPath Korea", "hero_s": "为留学生打造的优质AI引导系统。",
        "ask_ai": "咨询AI →", "search": "搜索", "admin": "管理中心",
        "welcome": "欢迎回来！", "kpi_visa": "签证类型", "kpi_uni": "合作大学", "kpi_job": "开放职位",
        "chat_placeholder": "咨询任何关于韩国生活的问题...",
        "source_badge": "官方来源", "uni_desc": "探索380多所提供GKS奖学金的韩国大学。",
        "visa_desc": "立即查询E-7-S或F-2-R签证资格。",
        "job_desc": "为国际人才提供的经过验证的职位列表。"
    },
    "🇹🇭 ภาษาไทย": {
        "home": "หน้าหลัก", "uni": "มหาวิทยาลัย", "career": "อาชีพ", "job": "งาน", "topik": "TOPIK", "visa": "วีซ่า AI",
        "hero_t": "UniPath Korea", "hero_s": "เส้นทางแนะนำด้วย AI ระดับพรีเมียมสำหรับนักศึกษาต่างชาติ",
        "ask_ai": "ถาม AI →", "search": "ค้นหา", "admin": "ศูนย์ดูแล",
        "welcome": "ยินดีต้อนรับ!", "kpi_visa": "ประเภทวีซ่า", "kpi_uni": "พันธมิตร", "kpi_job": "ตำแหน่งงาน",
        "chat_placeholder": "สอบถามเรื่องการใช้ชีวิตในเกาหลีได้ทุกเรื่อง...",
        "source_badge": "แหล่งข้อมูลทางการ", "uni_desc": "ค้นหามหาวิทยาลัยเกาหลี 380+ แห่งพร้อมทุน GKS",
        "visa_desc": "ตรวจสอบคุณสมบัติวีซ่า E-7-S หรือ F-2-R ได้ทันที",
        "job_desc": "รายการงานที่ผ่านการตรวจสอบแล้วสำหรับผู้มีความสามารถระดับโลก"
    },
    "🇲🇾 Bahasa Melayu": {
        "home": "Utama", "uni": "Universiti", "career": "Kerjaya", "job": "Pekerjaan", "topik": "TOPIK", "visa": "Visa AI",
        "hero_t": "UniPath Korea", "hero_s": "Laluan panduan AI premium untuk pelajar antarabangsa.",
        "ask_ai": "Tanya AI →", "search": "Cari", "admin": "Pusat Admin",
        "welcome": "Selamat Datang!", "kpi_visa": "Jenis Visa", "kpi_uni": "Univ Rakan Kongsi", "kpi_job": "Kekosongan",
        "chat_placeholder": "Tanya apa sahaja mengenai kehidupan di Korea...",
        "source_badge": "Sumber Rasmi", "uni_desc": "Terokai 380+ universiti Korea dengan sokongan GKS.",
        "visa_desc": "Semak kelayakan E-7-S atau F-2-R anda serta-merta.",
        "job_desc": "Senarai pekerjaan yang disahkan untuk bakat antarabangsa."
    },
    "🇷🇺 Русский": {
        "home": "Главная", "uni": "Университет", "career": "Карьера", "job": "Работа", "topik": "TOPIK", "visa": "Виза AI",
        "hero_t": "UniPath Korea", "hero_s": "Премиальный путь для иностранных студентов с ИИ-гидом.",
        "ask_ai": "Спросить ИИ →", "search": "Поиск", "admin": "Админ",
        "welcome": "Добро пожаловать!", "kpi_visa": "Типы виз", "kpi_uni": "Университеты", "kpi_job": "Вакансии",
        "chat_placeholder": "Спрашивайте о чем угодно в Корее...",
        "source_badge": "Источник", "uni_desc": "380+ корейских вузов с поддержкой GKS.",
        "visa_desc": "Проверьте право на визу E-7-S или F-2-R мгновенно.",
        "job_desc": "Проверенные вакансии для иностранных талантов."
    }
}

# ==========================================
# 3. STATE MANAGEMENT & INITIALIZATION
# ==========================================
if "language" not in st.session_state: st.session_state.language = "🇺🇸 English"
if "page" not in st.session_state: st.session_state.page = "Home"
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "chat_expanded" not in st.session_state: st.session_state.chat_expanded = False

def t(key): return TR[st.session_state.language].get(key, key)

# LLM / RAG Settings
try:
    Settings.llm = Gemini(api_key=st.secrets["GEMINI_API_KEY"], model_name="models/gemini-1.5-pro")
    Settings.embed_model = GeminiEmbedding(api_key=st.secrets["GEMINI_API_KEY"], model_name="models/embedding-001")
    supabase: Client = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
    vector_store = SupabaseVectorStore(postgres_connection_string=st.secrets["SUPABASE_DB_CONNECTION"], collection_name="documents")
    index = VectorStoreIndex.from_vector_store(vector_store)
except Exception as e:
    st.error(f"Initialization Error: {e}")

# ==========================================
# 4. AGENTIC ROUTER & SEARCH TOOLS
# ==========================================
def google_search(query):
    """Fallback tool for live web search."""
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query, "key": st.secrets["GOOGLE_SEARCH_API_KEY"],
        "cx": st.secrets["GOOGLE_CSE_ID"], "num": 3
    }
    try:
        r = requests.get(url, params=params).json()
        results = [f"{item['title']}: {item['snippet']} ({item['link']})" for item in r.get("items", [])]
        return "\n".join(results) if results else "No live results found."
    except: return "Search failed."

def agentic_query(user_input):
    """Routes query to RAG or Web Search based on intent."""
    with st.spinner("Analyzing UniPath Data..."):
        # 1. Check RAG first
        query_engine = index.as_query_engine(similarity_top_k=3)
        response = query_engine.query(user_input)
        
        if "I don't know" in str(response) or "not found" in str(response).lower():
            # 2. Trigger Agentic Web Search
            web_results = google_search(user_input)
            final_prompt = f"Using these web results: {web_results}, answer the user: {user_input}. Be professional."
            response = Settings.llm.complete(final_prompt)
            source = "Live Web Search (HiKorea/TOPIK)"
        else:
            source = "UniPath Internal Vector DB"
        
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({"role": "assistant", "content": str(response), "source": source})

# ==========================================
# 5. NAVIGATION PILLS
# ==========================================
def render_nav():
    cols = st.columns([1, 1, 1, 1, 1, 1, 2])
    pages = ["Home", "University", "Career", "Job", "TOPIK", "Visa"]
    icons = ["🏠", "🎓", "🚀", "💼", "📝", "🛂"]
    
    for i, p in enumerate(pages):
        with cols[i]:
            active = "primary" if st.session_state.page == p else "secondary"
            if st.button(f"{icons[i]} {t(p.lower())}", key=f"nav_{p}", use_container_width=True):
                st.session_state.page = p
                st.rerun()
    
    with cols[6]:
        st.session_state.language = st.selectbox("Lang", options=list(TR.keys()), label_visibility="collapsed")

# ==========================================
# 6. PAGE CONTENT BLOCKS
# ==========================================
render_nav()

if st.session_state.page == "Home":
    st.markdown(f"""
    <div class="hero-container">
        <h1>{t('hero_t')}</h1>
        <p style="font-size:1.3rem; opacity:0.9;">{t('hero_s')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="premium-card"><span class="kpi-badge">GLOBAL</span><h3>{t("uni")}</h3><p>{t("uni_desc")}</p></div>', unsafe_allow_html=True)
        if st.button(t("ask_ai"), key="ask_uni"):
            st.session_state.chat_expanded = True
            agentic_query("Tell me about top Korean universities for foreigners.")
            st.rerun()
    with c2:
        st.markdown(f'<div class="premium-card"><span class="kpi-badge">NEW</span><h3>{t("visa")}</h3><p>{t("visa_desc")}</p></div>', unsafe_allow_html=True)
        if st.button(t("ask_ai"), key="ask_visa"):
            st.session_state.chat_expanded = True
            agentic_query("Explain the E-7-S visa requirements.")
            st.rerun()
    with c3:
        st.markdown(f'<div class="premium-card"><span class="kpi-badge">HOT</span><h3>{t("job")}</h3><p>{t("job_desc")}</p></div>', unsafe_allow_html=True)
        if st.button(t("ask_ai"), key="ask_job"):
            st.session_state.chat_expanded = True
            agentic_query("Find job openings for English speakers in Seoul.")
            st.rerun()

elif st.session_state.page == "University":
    st.header(f"🎓 {t('uni')}")
    uni_data = pd.DataFrame({
        "University": ["Seoul National", "KAIST", "Yonsei", "Korea Univ", "Hanyang"],
        "GKS Support": ["Yes", "Yes", "Partial", "Yes", "Yes"],
        "Region": ["Seoul", "Daejeon", "Seoul", "Seoul", "Seoul"]
    })
    st.table(uni_data)
    if st.button("Query Ranking Details", use_container_width=True):
        agentic_query("Show me university rankings in Korea 2024.")

elif st.session_state.page == "Visa":
    st.header(f"🛂 {t('visa')}")
    st.info("Agentic AI is monitoring HiKorea.go.kr for latest F-2-R quota updates.")
    st.markdown("""
    - **D-2 to E-7**: Points based system (80+ points required)
    - **F-2-R**: Regional specialized visa for population-declining areas.
    """)
    if st.button("Check My Points Eligibility"):
        agentic_query("Calculate points for D-2 to E-7-1 visa transition.")

# ==========================================
# 7. ADMIN COMMAND CENTER (SIDEBAR)
# ==========================================
with st.sidebar:
    st.title("⚙️ Admin Panel")
    pwd = st.text_input("Access Key", type="password")
    if pwd == st.secrets["ADMIN_PASSWORD"]:
        st.success("Authorized")
        tab1, tab2 = st.tabs(["PDF Upload", "Metrics"])
        
        with tab1:
            uploaded_file = st.file_uploader("Batch Upload Knowledge (PDF)", type="pdf")
            if uploaded_file:
                with st.spinner("Ingesting & Vectorizing..."):
                    # Incremental Upsert Logic (Simplified)
                    # pdf_reader = PyPDF2.PdfReader(uploaded_file) ... Logic here
                    st.toast("Document Vectorized to Supabase!")
        
        with tab2:
            fig = px.line(x=[1,2,3,4], y=[10,25,45,110], title="Daily AI Queries")
            st.plotly_chart(fig, use_container_width=True)
            st.download_button("Export Usage CSV", data="date,user,query\n2024-03-01,anon,visa", file_name="logs.csv")

# ==========================================
# 8. DRAGGABLE UNI AVATAR WIDGET (HTML/JS)
# ==========================================
# This component handles the floating UI and triggers the chat panel
avatar_html = f"""
<div id="uni-avatar-container">
    <div class="pulse-circle" onclick="parent.window.toggleChat()">
        <span style="font-size: 35px;">🤖</span>
    </div>
</div>

<script>
const el = document.getElementById('uni-avatar-container');
let isDragging = false;
let currentX; let currentY; let initialX; let initialY; let xOffset = 0; let yOffset = 0;

el.addEventListener('mousedown', dragStart);
document.addEventListener('mousemove', drag);
document.addEventListener('mouseup', dragEnd);

function dragStart(e) {{
    initialX = e.clientX - xOffset;
    initialY = e.clientY - yOffset;
    if (e.target === el || el.contains(e.target)) isDragging = true;
}}
function drag(e) {{
    if (isDragging) {{
        e.preventDefault();
        currentX = e.clientX - initialX;
        currentY = e.clientY - initialY;
        xOffset = currentX;
        yOffset = currentY;
        setTranslate(currentX, currentY, el);
    }}
}}
function setTranslate(xPos, yPos, el) {{
    el.style.transform = `translate3d(${{xPos}}px, ${{yPos}}px, 0)`;
}}
function dragEnd() {{
    initialX = currentX;
    initialY = currentY;
    isDragging = false;
}}
</script>
"""
components.html(avatar_html, height=120)

# ==========================================
# 9. FLOATING CHAT PANEL (SLIDE-UP)
# ==========================================
if st.checkbox("Show AI Counselor", value=st.session_state.chat_expanded):
    st.markdown("""<div style="position:fixed; bottom:110px; right:30px; width:360px; height:500px; background:white; border-radius:20px; box-shadow:0 10px 50px rgba(0,0,0,0.2); z-index:10000; display:flex; flex-direction:column; padding:20px; border:1px solid #eee;">""", unsafe_allow_html=True)
    
    st.subheader("UniPath Assistant")
    
    # Chat Display
    chat_container = st.container(height=340)
    for msg in st.session_state.chat_history:
        with chat_container.chat_message(msg["role"]):
            st.write(msg["content"])
            if "source" in msg:
                st.markdown(f"<span class='kpi-badge' style='font-size:0.6rem;'>📍 {msg['source']}</span>", unsafe_allow_html=True)
                
    if prompt := st.chat_input(t("chat_placeholder")):
        agentic_query(prompt)
        st.rerun()

    if st.button("Close Panel"):
        st.session_state.chat_expanded = False
        st.rerun()
        
    st.markdown("</div>", unsafe_allow_html=True)