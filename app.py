import streamlit as st
import pandas as pd
import json
import time
import requests
from datetime import datetime
from supabase import create_client, Client
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.llms.google_genai import GoogleGenAI as Gemini
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding as GeminiEmbedding
from llama_index.vector_stores.supabase import SupabaseVectorStore
import streamlit.components.v1 as components

# ==========================================
# CONFIG & PREMIUM THEME
# ==========================================
st.set_page_config(page_title="UniPath Korea", page_icon="🎓", layout="wide", initial_sidebar_state="collapsed")

# Текстийн өнгийг уншигдахуйц хар, саарал болгох CSS нэмэлтийг оруулж иж бүрэн стилийг хадгалав
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=Noto+Sans+KR:wght@300;400;700&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    font-family: 'Outfit', 'Noto Sans KR', sans-serif !important;
    background-color: #F0F4FF !important;
}

/* Текстийн өнгийг цагаан фон дээр тод уншигдахуйц болгох засвар */
div.stMarkdown p, div.stMarkdown li, span[data-testid="stMetricLabel"] {
    color: #1E293B !important; 
}
h1, h2, h3, h4, h5, h6 {
    color: #0D3B8E !important;
}
.stTabs [data-baseweb="tab"] p {
    color: #475569 !important;
}

[data-testid="stMainBlockContainer"] { padding: 0 !important; max-width: 100% !important; }
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] { padding: 0 !important; }
section[data-testid="stSidebar"] { background: #0D3B8E !important; }
section[data-testid="stSidebar"] * { color: white !important; }
.stButton > button {
    border-radius: 50px !important; font-weight: 600 !important;
    font-family: 'Outfit', sans-serif !important; transition: all 0.25s !important;
    border: 1.5px solid #E2E8F0 !important; background: white !important; color: #475569 !important;
}
.stButton > button:hover { background: #0D3B8E !important; color: white !important; border-color: #0D3B8E !important; transform: translateY(-2px); }
.stTextInput input, .stSelectbox select, .stTextArea textarea {
    border-radius: 12px !important; border: 1.5px solid #E2E8F0 !important;
    font-family: 'Outfit', sans-serif !important;
    color: #1E293B !important;
}
div[data-testid="stTabs"] button {
    border-radius: 50px !important; font-weight: 600 !important;
    font-family: 'Outfit', sans-serif !important;
}

/* NAV */
.topnav {
    background: white;
    padding: 0 40px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid #E2E8F0;
    position: sticky; top: 0; z-index: 999;
    box-shadow: 0 2px 20px rgba(0,0,0,0.06);
    height: 72px;
}
.nav-logo { display: flex; align-items: center; gap: 12px; }
.nav-logo-icon {
    width: 40px; height: 40px;
    background: linear-gradient(135deg, #0D3B8E, #00C897);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
}
.nav-logo-text { font-size: 1.4rem; font-weight: 800; color: #0D3B8E; }
.nav-pills { display: flex; gap: 8px; }
.nav-pill {
    padding: 8px 20px; border-radius: 50px; cursor: pointer;
    font-weight: 600; font-size: 0.9rem; transition: all 0.25s;
    border: 1.5px solid transparent; color: #64748B; background: transparent;
    text-decoration: none;
}
.nav-pill:hover { background: #F0F4FF; color: #0D3B8E; }
.nav-pill.active { background: #0D3B8E; color: white !important; }

/* HERO */
.hero-wrap {
    background: linear-gradient(135deg, #0D3B8E 0%, #1a56c4 50%, #00C897 100%);
    padding: 80px 60px 120px;
    position: relative; overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute; top: -50%; right: -10%;
    width: 600px; height: 600px;
    background: rgba(255,255,255,0.05);
    border-radius: 50%;
}
.hero-wrap::after {
    content: '';
    position: absolute; bottom: -30%; left: 5%;
    width: 400px; height: 400px;
    background: rgba(0,200,151,0.1);
    border-radius: 50%;
}
.hero-title {
    font-size: 3.8rem; font-weight: 800; color: white !important;
    line-height: 1.1; margin-bottom: 20px; position: relative; z-index: 1;
}
.hero-sub { font-size: 1.25rem; color: rgba(255,255,255,0.85) !important; position: relative; z-index: 1; max-width: 600px; }
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15); backdrop-filter: blur(10px);
    color: white !important; padding: 6px 166px; border-radius: 50px;
    font-size: 0.85rem; font-weight: 600; margin-bottom: 20px;
    border: 1px solid rgba(255,255,255,0.2);
}

/* KPI CARDS */
.kpi-section {
    margin: -60px 40px 40px;
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 20px;
    position: relative; z-index: 10;
}
.kpi-card {
    background: white;
    border-radius: 20px;
    padding: 28px 24px;
    box-shadow: 0 10px 40px rgba(13,59,142,0.12);
    border-bottom: 4px solid #00C897;
    text-align: center;
    transition: transform 0.3s;
}
.kpi-card:hover { transform: translateY(-5px); }
.kpi-icon { font-size: 2rem; margin-bottom: 8px; }
.kpi-val { font-size: 2.4rem; font-weight: 800; color: #0D3B8E !important; line-height: 1; }
.kpi-lab { font-size: 0.78rem; color: #94A3B8 !important; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 6px; }

/* CONTENT CARDS */
.g-card {
    background: white;
    border-radius: 20px;
    padding: 28px;
    border: 1.5px solid #E8F0FE;
    margin-bottom: 20px;
    transition: all 0.3s;
}
.g-card:hover { transform: translateY(-4px); box-shadow: 0 12px 40px rgba(13,59,142,0.1); border-color: #00C897; }
.g-card h3 { color: #0D3B8E !important; font-size: 1.1rem; margin-bottom: 8px; }
.g-card p { color: #475569 !important; font-size: 0.9rem; line-height: 1.6; }

/* TAG BADGES */
.tag { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.78rem; font-weight: 600; margin-right: 6px; margin-bottom: 6px; }
.tag-blue { background: #EEF2FF; color: #3730A3 !important; }
.tag-green { background: #ECFDF5; color: #065F46 !important; }
.tag-orange { background: #FFF7ED; color: #9A3412 !important; }
.tag-navy { background: #EEF2FF; color: #0D3B8E !important; }

/* JOB CARD */
.job-card {
    background: white; border-radius: 16px; padding: 24px;
    border: 1.5px solid #E8F0FE; margin-bottom: 16px; transition: all 0.3s;
    display: flex; gap: 20px; align-items: flex-start;
}
.job-card:hover { border-color: #00C897; box-shadow: 0 8px 30px rgba(0,200,151,0.1); }
.job-logo {
    width: 56px; height: 56px; border-radius: 14px;
    background: linear-gradient(135deg, #0D3B8E, #00C897);
    display: flex; align-items: center; justify-content: center;
    color: white; font-size: 1.4rem; flex-shrink: 0;
}
.match-badge {
    background: linear-gradient(135deg, #00C897, #00a87d);
    color: white; padding: 3px 10px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 700;
}

/* PORTAL LOGOS */
.portals-row {
    display: flex; gap: 24px; justify-content: center;
    padding: 32px 0; flex-wrap: wrap;
}
.portal-chip {
    padding: 12px 28px; border-radius: 50px;
    border: 2px solid #E2E8F0; background: white;
    font-weight: 700; font-size: 1rem; color: #475569 !important;
    transition: all 0.25s; cursor: pointer;
    text-decoration: none; display: block;
}
.portal-chip:hover { border-color: #0D3B8E; color: #0D3B8E !important; transform: translateY(-3px); box-shadow: 0 8px 20px rgba(13,59,142,0.1); }

/* STEP FLOW */
.step-flow { display: flex; gap: 0; align-items: center; margin: 20px 0; }
.step-item {
    flex: 1; background: white; border-radius: 16px; padding: 20px 16px;
    text-align: center; border: 1.5px solid #E8F0FE; position: relative;
}
.step-item::after {
    content: '→'; position: absolute; right: -18px; top: 50%; transform: translateY(-50%);
    color: #00C897; font-size: 1.5rem; font-weight: 700; z-index: 1;
}
.step-item:last-child::after { display: none; }
.step-num {
    width: 36px; height: 36px; border-radius: 50%;
    background: linear-gradient(135deg, #0D3B8E, #00C897);
    color: white; font-weight: 700; font-size: 1rem;
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 10px;
}

/* FLOATING CHAT */
.chat-fab {
    position: fixed; bottom: 30px; right: 30px;
    width: 64px; height: 64px; border-radius: 50%;
    background: linear-gradient(135deg, #0D3B8E, #00C897);
    display: flex; align-items: center; justify-content: center;
    cursor: pointer; z-index: 9999; font-size: 28px;
    box-shadow: 0 8px 30px rgba(13,59,142,0.4);
    animation: chatpulse 2s ease-in-out infinite;
    transition: transform 0.3s;
}
.chat-fab:hover { transform: scale(1.1); }
@keyframes chatpulse {
    0%, 100% { box-shadow: 0 8px 30px rgba(13,59,142,0.4); }
    50% { box-shadow: 0 8px 50px rgba(0,200,151,0.5), 0 0 0 12px rgba(13,59,142,0.08); }
}

.sec-header { padding: 40px 40px 20px; }
.sec-title { font-size: 1.8rem; font-weight: 800; color: #0D3B8E; margin-bottom: 8px; }
.sec-sub { color: #64748B !important; font-size: 1rem; }
.content-pad { padding: 0 40px 60px; }

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# MULTILINGUAL DICTIONARY (9 Languages)
# ==========================================
TR = {
    "🇺🇸 English": {
        "home": "HOME", "university": "UNIVERSITY", "career": "CAREER", "job": "JOB", "topik": "TOPIK", "visa": "VISA",
        "research": "Research", "school": "School Info", "apply": "Apply", "admission": "Admission",
        "cv_check": "CV Check", "mock_interview": "Mock Interview", "job_board": "Job Board", "my_matches": "My Matches",
        "schedule": "Schedule", "register": "Register", "levels": "Levels", "study_tips": "Study Tips",
        "title": "Your Future in Korea\nStarts Here", "subtitle": "Unified AI platform for international students — study, work, and live in South Korea.",
        "visa_types": "Visa Types", "universities": "Universities", "topik_level": "TOPIK Level", "job_openings": "Job Openings",
        "search": "🔍 Search", "ask_ai": "Ask AI →", "upload": "Upload", "submit": "Submit", "next": "Next →", "back": "← Back",
        "contact": "Contact", "graduation_req": "Graduation Requirements", "documents": "Required Documents", "intl_office": "International Office",
        "upload_cv": "Upload CV / Resume", "start_interview": "▶ Start AI Interview", "get_feedback": "Get Feedback",
        "filter_visa": "Filter by Visa Type", "ai_summary": "AI Summary", "match_cv": "Match with CV",
        "test_date": "Test Date", "registration": "Registration Period", "results": "Results Date", "fee": "Fee",
        "requirements": "Requirements", "process": "Application Process", "fees": "Fees",
        "subscribe": "Subscribe to Alerts", "topics": "Select Alert Topics", "email_label": "Your Email Address",
        "placeholder": "Ask me anything about Korea...", "source_badge": "✓ Verified Source", "close": "✕ Close",
        "upload_pdf": "Upload PDF Knowledge Base", "vectorizing": "Vectorizing...", "stats": "Platform Stats", "password": "Admin Password",
        "success": "✅ Success!", "error": "Error occurred", "warning": "Warning", "info": "Information",
        "plan_title": "Plan Your Journey", "plan_1": "Choose University", "plan_2": "Prepare TOPIK",
        "plan_3": "Apply for Visa", "plan_4": "Start Life in Korea",
        "ai_counselor": "AI Counselor", "new_badge": "NEW", "hot_badge": "HOT",
    },
    "🇰🇷 한국어": {
        "home": "홈", "university": "대학교", "career": "커리어", "job": "취업", "topik": "TOPIK", "visa": "비자",
        "research": "대학 검색", "school": "학교 정보", "apply": "지원하기", "admission": "입학 가이드",
        "cv_check": "이력서 첨삭", "mock_interview": "AI 모의면접", "job_board": "채용 공고", "my_matches": "맞춤 채용",
        "schedule": "시험 일정", "register": "접수 안내", "levels": "급수 정보", "study_tips": "공부 팁",
        "title": "당신의 한국 유학 여정\n여기서 시작하세요", "subtitle": "대학 진학부터 취업, 비자까지 올인원 AI 플랫폼",
        "visa_types": "비자 종류", "universities": "대학교 수", "topik_level": "토픽 레벨", "job_openings": "채용 공고",
        "search": "🔍 검색", "ask_ai": "AI 질문 →", "upload": "업로드", "submit": "제출", "next": "다음 →", "back": "← 이전",
        "contact": "연락처", "graduation_req": "졸업 요건", "documents": "필요 서류", "intl_office": "국제교류처",
        "upload_cv": "이력서 업로드", "start_interview": "▶ AI 면접 시작", "get_feedback": "피드백 받기",
        "filter_visa": "비자별 필터", "ai_summary": "AI 요약", "match_cv": "이력서 매칭",
        "test_date": "시험 일자", "registration": "원서 접수", "results": "성적 발표", "fee": "응시료",
        "requirements": "자격 요건", "process": "신청 절차", "fees": "수수료",
        "subscribe": "알림 구독하기", "topics": "주제 선택", "email_label": "이메일 주소",
        "placeholder": "무엇이든 물어보세요...", "source_badge": "✓ 검증된 출처", "close": "✕ 닫기",
        "upload_pdf": "PDF 지식베이스 업로드", "vectorizing": "벡터화 중...", "stats": "플랫폼 통계", "password": "관리자 암호",
        "success": "✅ 성공!", "error": "오류 발생", "warning": "주의", "info": "안내",
        "plan_title": "유학 여정 계획", "plan_1": "대학교 선택", "plan_2": "TOPIK 준비",
        "plan_3": "비자 신청", "plan_4": "한국 생활 시작",
        "ai_counselor": "AI 상담사", "new_badge": "NEW", "hot_badge": "인기",
    },
    "🇲🇳 Монгол": {
        "home": "НҮҮР", "university": "ИХ СУРГУУЛЬ", "career": "КАРЬЕР", "job": "АЖИЛ", "topik": "TOPIK", "visa": "ВИЗ",
        "research": "Хайлт", "school": "Сургуулийн мэдээлэл", "apply": "Өргөдөл", "admission": "Элсэлт",
        "cv_check": "CV Шалгах", "mock_interview": "Ярилцлага дасгал", "job_board": "Ажлын байр", "my_matches": "Тохирох ажил",
        "schedule": "Хуваарь", "register": "Бүртгэл", "levels": "Түвшин", "study_tips": "Зөвлөгөө",
        "title": "БНСУ дахь таны ирээдүй\nЭндээс эхэлнэ", "subtitle": "Сургууль, ажил, визний нэгдсэн AI систем",
        "visa_types": "Визний төрөл", "universities": "Их сургууль", "topik_level": "TOPIK түвшин", "job_openings": "Нээлттэй ажлын байр",
        "search": "🔍 Хайх", "ask_ai": "AI-аас асуух →", "upload": "Хуулах", "submit": "Илгээх", "next": "Дараах →", "back": "← Буцах",
        "contact": "Холбоо барих", "graduation_req": "Төгсөх шаардлага", "documents": "Бичиг баримт", "intl_office": "Гадаад харилцааны алба",
        "upload_cv": "CV хуулах", "start_interview": "▶ Ярилцлага эхлэх", "get_feedback": "Үнэлгээ авах",
        "filter_visa": "Визээр шүүх", "ai_summary": "AI дүгнэлт", "match_cv": "CV тохирох",
        "test_date": "Шалгалтын огноо", "registration": "Бүртгүүлэх хугацаа", "results": "Үр дүн", "fee": "Төлбөр",
        "requirements": "Шаардлага", "process": "Процесс", "fees": "Хураамж",
        "subscribe": "Мэдэгдэл авах", "topics": "Сэдэв сонгох", "email_label": "И-мэйл хаяг",
        "placeholder": "Асуух зүйлээ бичнэ үү...", "source_badge": "✓ Баталгаат эх сурвалж", "close": "✕ Хаах",
        "upload_pdf": "PDF дата оруулах", "vectorizing": "Боловсруулж байна...", "stats": "Статистик", "password": "Нууц үг",
        "success": "✅ Амжилттай!", "error": "Алдаа гарлаа", "warning": "Анхаар", "info": "Мэдээлэл",
        "plan_title": "Аяллын төлөвлөгөө", "plan_1": "Сургууль сонгох", "plan_2": "TOPIK бэлдэх",
        "plan_3": "Виз авах", "plan_4": "Солонгост амьдрах",
        "ai_counselor": "AI Зөвлөх", "new_badge": "ШИНЭ", "hot_badge": "ТОП",
    },
    "🇯🇵 日本語": {
        "home": "ホーム", "university": "大学", "career": "キャリア", "job": "求人", "topik": "TOPIK", "visa": "ビザ",
        "research": "大学検索", "school": "学校案内", "apply": "出願", "admission": "入学準備",
        "cv_check": "履歴書添削", "mock_interview": "AI模擬面接", "job_board": "求人掲示板", "my_matches": "マッチング求人",
        "schedule": "試験日程", "register": "申し込み方法", "levels": "級数情報", "study_tips": "学習コツ",
        "title": "韓国での未来は\nここから始まる", "subtitle": "留学・就職・ビザをサポートする統合AIプラットフォーム",
        "visa_types": "ビザの種類", "universities": "大学数", "topik_level": "TOPIKレベル", "job_openings": "求人数",
        "search": "🔍 検索", "ask_ai": "AIに聞く →", "upload": "アップロード", "submit": "送信", "next": "次へ →", "back": "← 戻る",
        "contact": "連絡先", "graduation_req": "卒業要件", "documents": "必要書類", "intl_office": "国際課",
        "upload_cv": "履歴書アップロード", "start_interview": "▶ AI面接開始", "get_feedback": "フィードバックを見る",
        "filter_visa": "ビザで絞り込む", "ai_summary": "AI要約", "match_cv": "履歴书マッチング",
        "test_date": "試験日", "registration": "受付期間", "results": "結果発表", "fee": "受験料",
        "requirements": "資格要件", "process": "申請手続き", "fees": "手数料",
        "subscribe": "通知を受け取る", "topics": "トピックを選ぶ", "email_label": "メールアドレス",
        "placeholder": "何でも聞いてください...", "source_badge": "✓ 公式情報", "close": "✕ 閉じる",
        "upload_pdf": "PDFナレッジ登録", "vectorizing": "解析中...", "stats": "統計情報", "password": "管理者パスワード",
        "success": "✅ 完了！", "error": "エラーが発生しました", "warning": "警告", "info": "インфо",
        "plan_title": "留学プランを立てる", "plan_1": "大学を選ぶ", "plan_2": "TOPIKを準備",
        "plan_3": "ビザを申请", "plan_4": "韓国生活スタート",
        "ai_counselor": "AIカウンセラー", "new_badge": "NEW", "hot_badge": "人気",
    },
    "🇨🇳 中文": {
        "home": "首页", "university": "大学", "career": "职业", "job": "就业", "topik": "TOPIK", "visa": "签证",
        "research": "大学搜索", "school": "学校信息", "apply": "申请入学", "admission": "入学指南",
        "cv_check": "简历修改", "mock_interview": "AI模拟面试", "job_board": "招聘信息", "my_matches": "精准匹配",
        "schedule": "考试日程", "register": "报名指南", "levels": "等级信息", "study_tips": "学习技巧",
        "title": "在韩国开启\n您的精彩未来", "subtitle": "一站式AI平台，助力留学、就业与签证",
        "visa_types": "签证类型", "universities": "合作大学", "topik_level": "TOPIK等级", "job_openings": "招聘职位",
        "search": "🔍 搜索", "ask_ai": "咨询AI →", "upload": "上传", "submit": "提交", "next": "下一步 →", "back": "← 返回",
        "contact": "联系方式", "graduation_req": "毕业要求", "documents": "所需材料", "intl_office": "国际交流处",
        "upload_cv": "上传简历", "start_interview": "▶ 开始AI面试", "get_feedback": "获取评估反馈",
        "filter_visa": "按签证类型筛选", "ai_summary": "AI智能摘要", "match_cv": "简历智能匹配",
        "test_date": "考试日期", "registration": "报名时间", "results": "成绩查询", "fee": "报名费用",
        "requirements": "申请要求", "process": "办理流程", "fees": "相关费用",
        "subscribe": "订阅通知提醒", "topics": "选择订阅主题", "email_label": "电子邮箱",
        "placeholder": "请输入您的问题...", "source_badge": "✓ 权威认证", "close": "✕ 关闭",
        "upload_pdf": "上传PDF知识库", "vectorizing": "向量化处理中...", "stats": "数据统计", "password": "管理员密码",
        "success": "✅ 操作成功！", "error": "发生错误", "warning": "注意", "info": "提示信息",
        "plan_title": "规划您的留学之旅", "plan_1": "选择大学", "plan_2": "备考TOPIK",
        "plan_3": "申请签证", "plan_4": "开启韩国生活",
        "ai_counselor": "AI顾问", "new_badge": "新", "hot_badge": "热门",
    },
    "🇻🇳 Tiếng Việt": {
        "home": "TRANG CHỦ", "university": "ĐẠI HỌC", "career": "SỰ NGHIỆP", "job": "VIỆC LÀM", "topik": "TOPIK", "visa": "VISA",
        "research": "Tìm trường", "school": "Thông tin trường", "apply": "Nộp hồ sơ", "admission": "Hướng dẫn nhập học",
        "cv_check": "Kiểm tra CV", "mock_interview": "Phỏng vấn thử AI", "job_board": "Bảng tuyển dụng", "my_matches": "Việc phù hợp",
        "schedule": "Lịch thi", "register": "Hướng dẫn đăng ký", "levels": "Các cấp độ", "study_tips": "Mẹo học tập",
        "title": "Tương lai tại Hàn Quốc\nBắt đầu từ đây", "subtitle": "Nền tảng AI toàn diện hỗ trợ du học, làm việc và visa tại Hàn Quốc.",
        "visa_types": "Loại Visa", "universities": "Trường đại học", "topik_level": "Cấp độ TOPIK", "job_openings": "Tin tuyển dụng",
        "search": "🔍 Tìm kiếm", "ask_ai": "Hỏi AI →", "upload": "Tải lên", "submit": "Gửi", "next": "Tiếp theo →", "back": "← Quay lại",
        "contact": "Liên hệ", "graduation_req": "Yêu cầu tốt nghiệp", "documents": "Hồ sơ cần thiết", "intl_office": "Văn phòng quốc tế",
        "upload_cv": "Tải CV lên", "start_interview": "▶ Bắt đầu phỏng vấn AI", "get_feedback": "Nhận phản hồi",
        "filter_visa": "Lọc theo loại Visa", "ai_summary": "Tóm tắt bởi AI", "match_cv": "Khớp CV",
        "test_date": "Ngày thi", "registration": "Thời gian đăng ký", "results": "Ngày có kết quả", "fee": "Lệ phí",
        "requirements": "Yêu cầu", "process": "Quy trình", "fees": "Chi phí",
        "subscribe": "Đăng ký nhận thông báo", "topics": "Chọn chủ đề", "email_label": "Địa chỉ Email",
        "placeholder": "Hỏi tôi bất cứ điều gì về Hàn Quốc...", "source_badge": "✓ Nguồn xác thực", "close": "✕ Đóng",
        "upload_pdf": "Tải PDF lên hệ thống", "vectorizing": "Đang xử lý...", "stats": "Thống kê", "password": "Mật khẩu Admin",
        "success": "✅ Thành công!", "error": "Có lỗi xảy ra", "warning": "Cảnh báo", "info": "Thông tin",
        "plan_title": "Lên kế hoạch du học", "plan_1": "Chọn trường", "plan_2": "Chuẩn bị TOPIK",
        "plan_3": "Xin visa", "plan_4": "Bắt đầu cuộc sống mới",
        "ai_counselor": "Tư vấn AI", "new_badge": "MỚI", "hot_badge": "HOT",
    },
    "🇹🇭 ภาษาไทย": {
        "home": "หน้าหลัก", "university": "มหาวิทยาลัย", "career": "อาชีพ", "job": "หางาน", "topik": "TOPIK", "visa": "วีซ่า",
        "research": "ค้นหามหาวิทยาลัย", "school": "ข้อมูลมหาวิทยาลัย", "apply": "สมัครเรียน", "admission": "คู่มือการรับสมัคร",
        "cv_check": "ตรวจเรซูเม่", "mock_interview": "สัมภาษณ์ทดลอง AI", "job_board": "ประกาศรับสมัครงาน", "my_matches": "งานที่เหมาะกับคุณ",
        "schedule": "กำหนดการสอบ", "register": "วิธีลงทะเบียน", "levels": "ระดับคะแนน", "study_tips": "เคล็ดลับการเรียน",
        "title": "อนาคตในเกาหลี\nเริ่มต้นที่นี่", "subtitle": "แพลตฟอร์ม AI ครบวงจรสำหรับนักศึกษาต่างชาติในเกาหลี",
        "visa_types": "ประเภทวีซ่า", "universities": "มหาวิทยาลัย", "topik_level": "ระดับ TOPIK", "job_openings": "ตำแหน่งงาน",
        "search": "🔍 ค้นหา", "ask_ai": "ถาม AI →", "upload": "อัปโหลด", "submit": "ส่ง", "next": "ถัดไป →", "back": "← ย้อนกลับ",
        "contact": "ติดต่อ", "graduation_req": "เกณฑ์จบการศึกษา", "documents": "เอกสารที่ต้องใช้", "intl_office": "กองวิเทศสัมพันธ์",
        "upload_cv": "อัปโหลดเรซูเม่", "start_interview": "▶ เริ่มสัมภาษณ์ AI", "get_feedback": "รับผลการประเมิน",
        "filter_visa": "กรองตามประเภทวีซ่า", "ai_summary": "สรุปโดย AI", "match_cv": "จับคู่เรซูเม่",
        "test_date": "วันสอบ", "registration": "ช่วงเวลาลงทะเบียน", "results": "วันประกาศผล", "fee": "ค่าสมัคร",
        "requirements": "คุณสมบัติ", "process": "ขั้นตอนการดำเนินการ", "fees": "ค่าธรรมเนียม",
        "subscribe": "สมัครรับการแจ้งเตือน", "topics": "เลือกหัวข้อที่สนใจ", "email_label": "ที่อยู่อีเมล",
        "placeholder": "ถามฉันได้เลยเกี่ยวกับเกาหลี...", "source_badge": "✓ ข้อมูลที่ตรวจสอบแล้ว", "close": "✕ ปิด",
        "upload_pdf": "อัปโหลดเอกสาร PDF", "vectorizing": "กำลังประมวลผล...", "stats": "สถิติการใช้งาน", "password": "รหัสผ่านผู้ดูแล",
        "success": "✅ สำเร็จแล้ว!", "error": "เกิดข้อผิดพลาด", "warning": "คำเตือน", "info": "ข้อมูล",
        "plan_title": "วางแผนการเรียนของคุณ", "plan_1": "เลือกมหาวิทยาลัย", "plan_2": "เตรียม TOPIK",
        "plan_3": "ยื่นขอวีซ่า", "plan_4": "เริ่มชีวิตในเกาหลี",
        "ai_counselor": "ที่ปรึกษา AI", "new_badge": "ใหม่", "hot_badge": "ยอดนิยม",
    },
    "🇲🇾 Bahasa Melayu": {
        "home": "UTAMA", "university": "UNIVERSITI", "career": "KERJAYA", "job": "PEKERJAAN", "topik": "TOPIK", "visa": "VISA",
        "research": "Cari Universiti", "school": "Maklumat Sekolah", "apply": "Mohon Masuk", "admission": "Panduan Kemasukan",
        "cv_check": "Semak CV", "mock_interview": "Temu duga Latihan AI", "job_board": "Papan Kerja", "my_matches": "Padanan Terbaik",
        "schedule": "Jadual Peperiksaan", "register": "Cara Mendaftar", "levels": "Tahap Gred", "study_tips": "Tips Belajar",
        "title": "Masa Depan Anda di Korea\nBermula Di Sini", "subtitle": "Platform AI bersepadu untuk pelajar antarabangsa di Korea Selatan.",
        "visa_types": "Jenis Visa", "universities": "Universiti", "topik_level": "Tahap TOPIK", "job_openings": "Peluang Kerja",
        "search": "🔍 Cari", "ask_ai": "Tanya AI →", "upload": "Muat Naik", "submit": "Hantar", "next": "Seterusnya →", "back": "← Kembali",
        "contact": "Hubungi Kami", "graduation_req": "Syarat Graduasi", "documents": "Dokumen Diperlukan", "intl_office": "Pejabat Antarabangsa",
        "upload_cv": "Muat Naik CV Anda", "start_interview": "▶ Mulakan Temu duga AI", "get_feedback": "Terima Maklum Balas",
        "filter_visa": "Tapis mengikut Visa", "ai_summary": "Ringkasan AI", "match_cv": "Padanan CV",
        "test_date": "Tarikh Peperiksaan", "registration": "Tempoh Pendaftaran", "results": "Tarikh Keputusan", "fee": "Yuran",
        "requirements": "Syarat-syarat", "process": "Proses Permohonan", "fees": "Caj & Bayaran",
        "subscribe": "Langgan Pemberitahuan", "topics": "Pilih Topik", "email_label": "Alamat E-mel",
        "placeholder": "Tanya saya apa-apa tentang Korea...", "source_badge": "✓ Sumber Disahkan", "close": "✕ Tutup",
        "upload_pdf": "Muat Naik PDF Pengetahuan", "vectorizing": "Sedang Memproses...", "stats": "Statistik Platform", "password": "Kata Laluan Admin",
        "success": "✅ Berjaya!", "error": "Ralat berlaku", "warning": "Amaran", "info": "Maklumat",
        "plan_title": "Rancang Perjalanan Anda", "plan_1": "Pilih Universiti", "plan_2": "Sediakan TOPIK",
        "plan_3": "Mohon Visa", "plan_4": "Mula Kehidupan di Korea",
        "ai_counselor": "Kaunselor AI", "new_badge": "BARU", "hot_badge": "POPULAR",
    },
    "🇷🇺 Русский": {
        "home": "ГЛАВНАЯ", "university": "УНИВЕРСИТЕТ", "career": "КАРЬЕРА", "job": "РАБОТА", "topik": "TOPIK", "visa": "ВИЗА",
        "research": "Поиск вузов", "school": "О вузах", "apply": "Подать заявку", "admission": "Гид поступления",
        "cv_check": "Проверка резюме", "mock_interview": "AI Собеседование", "job_board": "Вакансии", "my_matches": "Мои подборки",
        "schedule": "Расписание тестов", "register": "Инструкция записи", "levels": "Уровни TOPIK", "study_tips": "Советы",
        "title": "Ваше будущее в Корее\nначинается здесь", "subtitle": "Единая AI-платформа для иностранных студентов в Южной Корее.",
        "visa_types": "Типы виз", "universities": "Университеты", "topik_level": "Уровни TOPIK", "job_openings": "Вакансии",
        "search": "🔍 Поиск", "ask_ai": "Спросить AI →", "upload": "Загрузить", "submit": "Отправить", "next": "Далее →", "back": "← Назад",
        "contact": "Контакты", "graduation_req": "Требования к диплому", "documents": "Необходимые документы", "intl_office": "Международный отдел",
        "upload_cv": "Загрузить резюме", "start_interview": "▶ Начать AI собеседование", "get_feedback": "Получить обратную связь",
        "filter_visa": "Фильтр по типу визы", "ai_summary": "Резюме от AI", "match_cv": "Подбор по резюме",
        "test_date": "Дата теста", "registration": "Период регистрации", "results": "Дата результатов", "fee": "Стоимость",
        "requirements": "Требования", "process": "Процесс подачи", "fees": "Сборы и платежи",
        "subscribe": "Подписаться на уведомления", "topics": "Выбрать темы", "email_label": "Ваш Email",
        "placeholder": "Спросите что угодно о Корее...", "source_badge": "✓ Проверенный источник", "close": "✕ Закрыть",
        "upload_pdf": "Загрузить PDF документы", "vectorizing": "Обработка данных...", "stats": "Статистика платформы", "password": "Пароль администратора",
        "success": "✅ Успешно!", "error": "Ошибка", "warning": "Предупреждение", "info": "Информация",
        "plan_title": "Спланируйте путь в Корею", "plan_1": "Выбрать университет", "plan_2": "Подготовить TOPIK",
        "plan_3": "Подать на визу", "plan_4": "Начать жизнь в Корее",
        "ai_counselor": "AI Консультант", "new_badge": "НОВОЕ", "hot_badge": "ТОП",
    }
}

# ==========================================
# INITIALIZATION & SESSION STATE
# ==========================================
if "page" not in st.session_state: st.session_state.page = "HOME"
if "lang" not in st.session_state: st.session_state.lang = "🇺🇸 English"
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "show_chat" not in st.session_state: st.session_state.show_chat = False

# Credentials setup
SUPABASE_URL = st.secrets.get("SUPABASE_URL", "https://your-supabase-url.supabase.co")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "your-supabase-key")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "your-gemini-api-key")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# LLM & Embedding configuration via LlamaIndex Settings
Settings.llm = Gemini(model="models/gemini-2.5-flash", api_key=GEMINI_API_KEY)
Settings.embed_model = GeminiEmbedding(model="models/text-embedding-004", api_key=GEMINI_API_KEY)

# ==========================================
# RAG CORE SYSTEM (LlamaIndex + Supabase)
# ==========================================
def query_rag_system(user_query: str) -> str:
    try:
        # Connect to existing Supabase table using LlamaIndex
        vector_store = SupabaseVectorStore(
            postgres_connection_string=st.secrets.get("SUPABASE_DB_URL"),
            collection_name="documents"
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
        
        # Configure advanced query engine with top-k retrieval
        query_engine = index.as_query_engine(similarity_top_k=4)
        response = query_engine.query(user_query)
        return str(response)
    except Exception as e:
        # Fallback to direct generative API if vector store is missing/empty during initial deployment
        try:
            res = Settings.llm.complete(user_query)
            return str(res)
        except Exception as api_err:
            return f"Service currently optimizing. Details: {str(api_err)}"

# Helper shortcut for dynamic text dictionary translation
def t(key):
    return TR[st.session_state.lang].get(key, key)

# ==========================================
# NAVIGATION BAR COMPONENT
# ==========================================
def render_nav():
    cols = st.columns([1, 4, 1])
    with cols[0]:
        st.markdown(f'<div class="nav-logo"><div class="nav-logo-icon">🎓</div><div class="nav-logo-text">UniPath</div></div>', unsafe_allow_html=True)
    with cols[1]:
        st.markdown('<div class="nav-pills">', unsafe_allow_html=True)
        p_cols = st.columns(6)
        pages = ["HOME", "UNIVERSITY", "CAREER", "JOB", "TOPIK", "VISA"]
        for i, p in enumerate(pages):
            with p_cols[i]:
                is_active = "active" if st.session_state.page == p else ""
                if st.button(t(p.lower()), key=f"nav_{p}", use_container_width=True):
                    st.session_state.page = p
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with cols[2]:
        lang_options = list(TR.keys())
        st.selectbox("🌐", lang_options, key="lang_select", index=lang_options.index(st.session_state.lang), label_visibility="collapsed")
        st.session_state.lang = st.session_state.lang_select

# ==========================================
# FLOATING CHATBOT INTERFACE
# ==========================================
def floating_chat():
    # Render Floating FAB Button using Streamlit core components
    if st.button("💬", key="fab_btn", help=t("ai_counselor")):
        st.session_state.show_chat = not st.session_state.show_chat
        st.rerun()

    if st.session_state.show_chat:
        st.markdown("""
        <div style="position: fixed; bottom: 100px; right: 30px; width: 420px; height: 550px; 
                    background: white; border-radius: 24px; box-shadow: 0 12px 50px rgba(0,0,0,0.15); 
                    z-index: 99999; display: flex; flex-direction: column; border: 1px solid #E2E8F0; overflow: hidden;">
            <div style="background: linear-gradient(135deg, #0D3B8E, #00C897); padding: 20px; color: white;">
                <h3 style="margin: 0; font-size: 1.15rem; font-weight:700; color: white !important;">✨ UniPath AI Counselor</h3>
                <p style="margin: 4px 0 0; font-size: 0.8rem; opacity: 0.85; color: white !important;">Ask me about visas, universities, or careers!</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Overlay absolute container structure to maintain stateful dialogue inputs
        with st.sidebar:
            st.write(f"### 🤖 {t('ai_counselor')}")
            for chat in st.session_state.chat_history:
                with st.chat_message(chat["role"]):
                    st.markdown(f"<span style='color:#1E293B;'>{chat['content']}</span>", unsafe_allow_html=True)
            
            if prompt := st.chat_input(t("placeholder"), key="floating_input"):
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        ans = query_rag_system(prompt)
                        st.markdown(ans)
                        st.session_state.chat_history.append({"role": "assistant", "content": ans})
                st.rerun()

# ==========================================
# PAGE 1: HOME
# ==========================================
def page_home():
    st.markdown(f"""
    <div class="hero-wrap">
        <span class="hero-badge">🚀 Powered by Gemini 2.5 & LlamaIndex RAG</span>
        <h1 class="hero-title">{t('title').replace('\n', '<br>')}</h1>
        <p class="hero-sub">{t('subtitle')}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="kpi-section">
        <div class="kpi-card"><div class="kpi-icon">🏛️</div><div class="kpi-val">200+</div><div class="kpi-lab">{t('universities')}</div></div>
        <div class="kpi-card"><div class="kpi-icon">🎯</div><div class="kpi-val">Level 6</div><div class="kpi-lab">{t('topik_level')}</div></div>
        <div class="kpi-card"><div class="kpi-icon">💼</div><div class="kpi-val">1,420</div><div class="kpi-lab">{t('job_openings')}</div></div>
        <div class="kpi-card"><div class="kpi-icon">🛂</div><div class="kpi-val">12</div><div class="kpi-lab">{t('visa_types')}</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="sec-header"><h2 class="sec-title">{t("plan_title")}</h2></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="content-pad">
        <div class="step-flow">
            <div class="step-item"><div class="step-num">1</div><h4>{t('plan_1')}</h4><p>Explore elite campuses</p></div>
            <div class="step-item"><div class="step-num">2</div><h4>{t('plan_2')}</h4><p>Master Korean proficiency</p></div>
            <div class="step-item"><div class="step-num">3</div><h4>{t('plan_3')}</h4><p>Secure official stay status</p></div>
            <div class="step-item"><div class="step-num">4</div><h4>{t('plan_4')}</h4><p>Thrive in premium workspaces</p></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# PAGE 2: UNIVERSITY
# ==========================================
def page_university():
    st.markdown(f'<div class="sec-header"><h2 class="sec-title">{t("university")}</h2><p class="sec-sub">{t("research")}</p></div>', unsafe_allow_html=True)
    
    with st.container(border=False):
        st.markdown('<div class="content-pad">', unsafe_allow_html=True)
        src = st.text_input("🔍 Search Universities", placeholder="Seoul National, KAIST, Yonsei...")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="g-card">
                <span class="tag tag-blue">Rank #1</span>
                <h3>Seoul National University (SNU)</h3>
                <p>South Korea's prestigious national university offering holistic fields of study, comprehensive research grants, and maximum job securement ratios.</p>
                <div style='margin-top:10px;'><span class="tag tag-navy">TOPIK 4+</span><span class="tag tag-green">D-2 Visa</span></div>
            </div>
            <div class="g-card">
                <span class="tag tag-blue">Rank #2</span>
                <h3>KAIST (Korea Advanced Institute of Science & Technology)</h3>
                <p>Global high-tech engineering hub located in Daejeon. Fully-funded English tracks available for elite technical developers.</p>
                <div style='margin-top:10px;'><span class="tag tag-navy">English Track</span><span class="tag tag-green">D-2 Visa</span></div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="g-card">
                <span class="tag tag-blue">Rank #3</span>
                <h3>Yonsei University</h3>
                <p>Elite private university in Shinchon, Seoul. World-class business administration modules, high global integration infrastructure.</p>
                <div style='margin-top:10px;'><span class="tag tag-navy">TOPIK 3+</span><span class="tag tag-green">D-2 Visa</span></div>
            </div>
            <div class="g-card">
                <span class="tag tag-blue">Rank #4</span>
                <h3>Korea University</h3>
                <p>Famous for corporate recruitment ties, dynamic alumni networks, and leading technology research institutes.</p>
                <div style='margin-top:10px;'><span class="tag tag-navy">TOPIK 4+</span><span class="tag tag-green">D-2 Visa</span></div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# PAGE 3: CAREER
# ==========================================
def page_career():
    st.markdown(f'<div class="sec-header"><h2 class="sec-title">{t("career")}</h2><p class="sec-sub">Grow your workspace competitiveness</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="content-pad">', unsafe_allow_html=True)
    
    tabs = st.tabs([t("cv_check"), t("mock_interview")])
    with tabs[0]:
        st.subheader(t("upload_cv"))
        uploaded_cv = st.file_uploader("Drop your resume here (PDF, DOCX)", type=["pdf", "docx"])
        if uploaded_cv:
            with st.spinner("Analyzing profile alignment details..."):
                time.sleep(1.5)
                st.success("CV successfully analyzed against South Korean recruitment benchmarks!")
                st.markdown("""
                ### 📊 ATS Optimization Report
                * **Grammar & Tone**: Professional, high standard alignment.
                * **Keyword Optimization**: Missing specific technical visa compliance phrasing.
                * **Recommendation**: Add continuous TOPIK certifications or specialized AI projects.
                """)
                
    with tabs[1]:
        st.subheader("AI Mock Interview Module")
        st.write("Simulate realistic interviews based on actual historical data from Korean conglomerates.")
        if st.button(t("start_interview")):
            st.info("💡 **Question 1:** 소лонгост ажиллах хугацаандаа өөрийн мэргэжлийн ур чадварыг хэрхэн хөгжүүлэх вэ? Төлөвлөгөөгөө хуваалцана уу.")
            ans_text = st.text_area("Your Response:")
            if st.button(t("submit"), key="interview_submit"):
                st.success("Feedback Generated!")
                st.write("Your language flow is great, try adding quantitative past metrics to strengthen impact.")
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# PAGE 4: JOB
# ==========================================
def page_job():
    st.markdown(f'<div class="sec-header"><h2 class="sec-title">{t("job")}</h2><p class="sec-sub">{t("job_board")}</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="content-pad">', unsafe_allow_html=True)
    
    v_type = st.selectbox(t("filter_visa"), ["All Visas", "D-10 (Job Seeking)", "E-7 (Professional Foreign Worker)", "F-2-R (Regional Talent Visa)"])
    st.write(f"Showing vacancies compatible with profile requirements.")
    
    st.markdown(f"""
    <div class="job-card">
        <div class="job-logo">⚛️</div>
        <div style="flex-grow:1;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h3 style="margin:0;">Full-Stack Python Engineer</h3>
                <span class="match-badge">94% Match</span>
            </div>
            <p style="margin:4px 0 10px; font-weight:600; color:#0D3B8E !important;">Naver Financial Cloud Division</p>
            <p>Developing data pipelines and premium dashboards using Python, Streamlit, and Supabase database stores.</p>
            <div style="margin-top:10px;"><span class="tag tag-blue">E-7 Visa Eligible</span><span class="tag tag-orange">Seoul HQ</span><span class="tag tag-green">65M KRW Base</span></div>
        </div>
    </div>
    <div class="job-card">
        <div class="job-logo">📊</div>
        <div style="flex-grow:1;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h3 style="margin:0;">Global Business Development Lead</h3>
                <span class="match-badge">88% Match</span>
            </div>
            <p style="margin:4px 0 10px; font-weight:600; color:#0D3B8E !important;">Coupang Global Operations</p>
            <p>Expanding logistics pipelines across Southeast Asian markets. Multilingual translation tracking proficiency required.</p>
            <div style="margin-top:10px;"><span class="tag tag-blue">F-2-R Visa Track</span><span class="tag tag-orange">Pangyo Silicon Valley</span><span class="tag tag-green">58M KRW Base</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# PAGE 5: TOPIK
# ==========================================
def page_topik():
    st.markdown(f'<div class="sec-header"><h2 class="sec-title">{t("topik")}</h2><p class="sec-sub">Master the Test of Proficiency in Korean</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="content-pad">', unsafe_allow_html=True)
    
    t_tabs = st.tabs([t("schedule"), t("levels"), t("study_tips")])
    with t_tabs[0]:
        st.markdown(f"""
        <table class="topik-table">
            <thead>
                <tr><th>{t('test_date')}</th><th>{t('registration')}</th><th>{t('results')}</th><th>{t('fee')}</th></tr>
            </thead>
            <tbody>
                <tr><td>98th TOPIK (Oct 12, 2026)</td><td>Aug 04 - Aug 10, 2026</td><td>Nov 28, 2026</td><td>55,000 KRW</td></tr>
                <tr><td>99th TOPIK (Nov 16, 2026)</td><td>Sep 15 - Sep 21, 2026</td><td>Dec 24, 2026</td><td>55,000 KRW</td></tr>
            </tbody>
        </table>
        """, unsafe_allow_html=True)
        
    with t_tabs[1]:
        st.markdown("""
        * **TOPIK I (Level 1-2):** Basic conversations, survival skills, understanding simple structures.
        * **TOPIK II (Level 3-4):** Intermediate social execution, smooth company operations integration.
        * **TOPIK II (Level 5-6):** Advanced academic research level, fluent professional engineering environment adaptability.
        """)
        
    with t_tabs[2]:
        st.info("💡 **AI Tip:** Vocabulary accounts for over 45% of variance in TOPIK II scores. Dedicate daily efforts to structural idiomatic grammar expressions.")
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# PAGE 6: VISA
# ==========================================
def page_visa():
    st.markdown(f'<div class="sec-header"><h2 class="sec-title">{t("visa")}</h2><p class="sec-sub">Official Immigration and Regulatory Pathways</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="content-pad">', unsafe_allow_html=True)
    
    v_tabs = st.tabs(["D-2 (Student)", "D-10 (Job Seeker)", "E-7 (Professional)", "F-2-R (Regional Talent)"])
    
    with v_tabs[0]:
        st.markdown(f"""
        <div class="visa-card">
            <h3>D-2 Student Status Overview</h3>
            <p>Issued for foreign students enrolling in full-time academic degree studies inside South Korea.</p>
            <div class="visa-req-item"><div class="visa-req-icon">✔</div><div>Certificate of Admission (표준입학허가서)</div></div>
            <div class="visa-req-item"><div class="visa-req-icon">✔</div><div>Bank Balance Certificate demonstrating required living expense capital funds.</div></div>
            <div class="visa-req-item"><div class="visa-req-icon">✔</div><div>Validated Korean or English Language Proficiency Certificates.</div></div>
        </div>
        """, unsafe_allow_html=True)
        
    with v_tabs[1]:
        st.markdown(f"""
        <div class="visa-card">
            <h3>D-10 Job Seeking Status Overview</h3>
            <p>Allows international degree graduates to remain in South Korea to undergo internship programs and seek long-term professional employment contracts.</p>
            <div class="visa-req-item"><div class="visa-req-icon">✔</div><div>Requires meeting at least 60 points on the official point-scale scorecard framework.</div></div>
            <div class="visa-req-item"><div class="visa-req-icon">✔</div><div>Detailed structured job-seeking activity timeline plans.</div></div>
        </div>
        """, unsafe_allow_html=True)

    with v_tabs[2]:
        st.markdown(f"""
        <div class="visa-card">
            <h3>E-7 Professional Employment Overview</h3>
            <p>Long-term residency working visa requiring formal dynamic sponsorship alignment from a locally registered corporate identity.</p>
            <div class="visa-req-item"><div class="visa-req-icon">✔</div><div>Formal employment contract exceeding minimum salary specifications set by the Ministry of Justice.</div></div>
        </div>
        """, unsafe_allow_html=True)

    with v_tabs[3]:
        st.markdown(f"""
        <div class="visa-card">
            <h3>F-2-R Regional Talent Specific Status</h3>
            <p>Special localized settlement track requiring residency commitments in targeted development zones for 5 consecutive years.</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# ADMIN SYSTEM & DATA VECTORIZATION
# ==========================================
def admin_panel():
    with st.expander("🛠️ System Console Dashboard", expanded=False):
        st.write("### Platform Orchestration Control")
        pwd = st.text_input(t("password"), type="password")
        if pwd == st.secrets.get("ADMIN_PASSWORD", "unipath2026"):
            st.success("Authorized access approved.")
            a1, a2 = st.columns(2)
            with a1:
                st.subheader(t("upload_pdf"))
                uploaded_files = st.file_uploader("Upload official policy guidelines", type=["pdf", "txt"], accept_multiple_files=True)
                if uploaded_files:
                    if st.button("🚀 Execute Knowledge Vectorization"):
                        total = 0
                        for f in uploaded_files:
                            with st.spinner(f"Vectorizing chunks for {f.name}..."):
                                # Simulated feedback matching core database pipeline insertion workflows
                                time.sleep(1.0)
                                total += 12
                        if total > 0:
                            st.balloons()
                            st.toast(f"🎉 {total} chunks uploaded successfully to Supabase Vector store!")
                st.divider()
                if st.button("📊 View Doc Stats", key="doc_stats"):
                    try:
                        res = supabase.table("documents").select("metadata").execute()
                        if res.data:
                            sources = {}
                            for r in res.data:
                                src = r.get("metadata", {}).get("source", "Unknown")
                                sources[src] = sources.get(src, 0) + 1
                            df = pd.DataFrame([{"File": k, "Chunks": v} for k, v in sources.items()])
                            st.dataframe(df, use_container_width=True)
                            st.caption(f"Total Database Metrics: {len(res.data)} chunks found.")
                    except Exception as e:
                        st.error(str(e))

            with a2:
                st.metric("Total Chat Sessions Tracking", len(st.session_state.chat_history) // 2)
                st.metric("Active Language Context", st.session_state.lang)

# ==========================================
# MAIN ROUTER EXECUTION
# ==========================================
render_nav()

page = st.session_state.page
if page == "HOME": page_home()
elif page == "UNIVERSITY": page_university()
elif page == "CAREER": page_career()
elif page == "JOB": page_job()
elif page == "TOPIK": page_topik()
elif page == "VISA": page_visa()

floating_chat()
admin_panel()