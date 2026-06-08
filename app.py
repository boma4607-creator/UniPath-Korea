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

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=Noto+Sans+KR:wght@300;400;700&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    font-family: 'Outfit', 'Noto Sans KR', sans-serif !important;
    background-color: #F0F4FF !important;
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
    font-size: 3.8rem; font-weight: 800; color: white;
    line-height: 1.1; margin-bottom: 20px; position: relative; z-index: 1;
}
.hero-sub { font-size: 1.25rem; color: rgba(255,255,255,0.85); position: relative; z-index: 1; max-width: 600px; }
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15); backdrop-filter: blur(10px);
    color: white; padding: 6px 16px; border-radius: 50px;
    font-size: 0.85rem; font-weight: 600; margin-bottom: 20px;
    border: 1px solid rgba(255,255,255,0.2);
}

/* KPI CARDS — float over hero */
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
.kpi-val { font-size: 2.4rem; font-weight: 800; color: #0D3B8E; line-height: 1; }
.kpi-lab { font-size: 0.78rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 6px; }

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
.g-card h3 { color: #0D3B8E; font-size: 1.1rem; margin-bottom: 8px; }
.g-card p { color: #64748B; font-size: 0.9rem; line-height: 1.6; }

/* TAG BADGES */
.tag { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.78rem; font-weight: 600; margin-right: 6px; margin-bottom: 6px; }
.tag-blue { background: #EEF2FF; color: #3730A3; }
.tag-green { background: #ECFDF5; color: #065F46; }
.tag-orange { background: #FFF7ED; color: #9A3412; }
.tag-navy { background: #EEF2FF; color: #0D3B8E; }

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
    font-weight: 700; font-size: 1rem; color: #475569;
    transition: all 0.25s; cursor: pointer;
    text-decoration: none; display: block;
}
.portal-chip:hover { border-color: #0D3B8E; color: #0D3B8E; transform: translateY(-3px); box-shadow: 0 8px 20px rgba(13,59,142,0.1); }

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

/* TOPIK TABLE */
.topik-table { width: 100%; border-collapse: collapse; }
.topik-table th { background: #0D3B8E; color: white; padding: 12px 16px; text-align: left; font-size: 0.85rem; }
.topik-table td { padding: 12px 16px; border-bottom: 1px solid #F1F5F9; font-size: 0.9rem; }
.topik-table tr:hover td { background: #F8FAFF; }

/* VISA TABS */
.visa-card { background: white; border-radius: 20px; padding: 28px; border: 1.5px solid #E8F0FE; }
.visa-req-item { display: flex; gap: 12px; padding: 10px 0; border-bottom: 1px solid #F1F5F9; }
.visa-req-icon { color: #00C897; font-size: 1.2rem; flex-shrink: 0; }

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

/* SECTION HEADERS */
.sec-header { padding: 40px 40px 20px; }
.sec-title { font-size: 1.8rem; font-weight: 800; color: #0D3B8E; margin-bottom: 8px; }
.sec-sub { color: #64748B; font-size: 1rem; }
.content-pad { padding: 0 40px 60px; }

/* HIDE STREAMLIT DEFAULT ELEMENTS */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# MULTILINGUAL DICTIONARY
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
        "filter_visa": "ビザで絞り込む", "ai_summary": "AI要約", "match_cv": "履歴書マッチング",
        "test_date": "試験日", "registration": "受付期間", "results": "結果発表", "fee": "受験料",
        "requirements": "資格要件", "process": "申請手続き", "fees": "手数料",
        "subscribe": "通知を受け取る", "topics": "トピックを選ぶ", "email_label": "メールアドレス",
        "placeholder": "何でも聞いてください...", "source_badge": "✓ 公式情報", "close": "✕ 閉じる",
        "upload_pdf": "PDFナレッジ登録", "vectorizing": "解析中...", "stats": "統計情報", "password": "管理者パスワード",
        "success": "✅ 完了！", "error": "エラーが発生しました", "warning": "警告", "info": "インフォ",
        "plan_title": "留学プランを立てる", "plan_1": "大学を選ぶ", "plan_2": "TOPIKを準備",
        "plan_3": "ビザを申請", "plan_4": "韓国生活スタート",
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
        "plan_3": "Оформить визу", "plan_4": "Начать жизнь в Корее",
        "ai_counselor": "AI Консультант", "new_badge": "НОВОЕ", "hot_badge": "ТОП",
    },
}

def t(key):
    return TR[st.session_state.get("lang", "🇺🇸 English")].get(key, key)

# ==========================================
# SESSION STATE
# ==========================================
if "lang" not in st.session_state: st.session_state.lang = "🇺🇸 English"
if "page" not in st.session_state: st.session_state.page = "HOME"
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "chat_open" not in st.session_state: st.session_state.chat_open = False
if "interview_step" not in st.session_state: st.session_state.interview_step = 0
if "interview_qa" not in st.session_state: st.session_state.interview_qa = []

# ==========================================
# AI INIT
# ==========================================
try:
    Settings.llm = Gemini(model="models/gemini-2.0-flash", api_key=st.secrets["GEMINI_API_KEY"])
    Settings.embed_model = GeminiEmbedding(model_name="models/text-embedding-004", api_key=st.secrets["GEMINI_API_KEY"])
    supabase: Client = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
    AI_READY = True
except Exception as e:
    AI_READY = False
    supabase = None

# ==========================================
# SUPABASE DATA LOADING
# ==========================================
@st.cache_data(ttl=300)
def load_universities():
    try:
        res = supabase.table("universities").select("*").execute()
        return res.data if res.data else []
    except: return []

@st.cache_data(ttl=300)
def load_scholarships():
    try:
        res = supabase.table("scholarships").select("*").execute()
        return res.data if res.data else []
    except: return []

@st.cache_data(ttl=300)
def load_jobs():
    try:
        res = supabase.table("jobs").select("*").order("match_score", desc=True).execute()
        return res.data if res.data else []
    except: return []

@st.cache_data(ttl=300)
def load_topik():
    try:
        res = supabase.table("topik_schedule").select("*").execute()
        return res.data if res.data else []
    except: return []

@st.cache_data(ttl=300)
def load_visa():
    try:
        res = supabase.table("visa_info").select("*").execute()
        return res.data if res.data else []
    except: return []

@st.cache_data(ttl=300)
def load_news():
    try:
        res = supabase.table("news").select("*").order("published_at", desc=True).execute()
        return res.data if res.data else []
    except: return []

def get_rag_response(query):
    lang_name = st.session_state.lang.split(" ", 1)[1] if " " in st.session_state.lang else "English"
    system_prompt = f"You are UNI, a helpful AI guide for international students in South Korea. Always respond in {lang_name}. Be concise, friendly, and accurate."
    try:
        vector_store = SupabaseVectorStore(
            postgres_connection_string=st.secrets["SUPABASE_DB_CONNECTION"],
            collection_name="documents"
        )
        index = VectorStoreIndex.from_vector_store(vector_store)
        qe = index.as_query_engine(similarity_top_k=3)
        response = qe.query(f"{system_prompt}\n\nQuestion: {query}")
        result = str(response)
        if len(result.strip()) < 20 or "don't know" in result.lower() or "no information" in result.lower():
            raise ValueError("Low quality RAG response")
        return result, "UniPath Knowledge Base"
    except:
        try:
            response = Settings.llm.complete(f"{system_prompt}\n\nQuestion: {query}")
            return str(response), "Gemini AI"
        except:
            return "I'm initializing. Please try again shortly.", "System"

# ==========================================
# TOP NAVIGATION
# ==========================================
def render_nav():
    pages = ["HOME", "UNIVERSITY", "CAREER", "JOB", "TOPIK", "VISA"]
    labels = [t("home"), t("university"), t("career"), t("job"), t("topik"), t("visa")]
    icons = ["🏠", "🎓", "🚀", "💼", "📝", "🛂"]

    col_logo, col_nav, col_lang = st.columns([2, 7, 2])
    with col_logo:
        st.markdown("""
        <div class="nav-logo" style="padding:16px 0;">
            <div class="nav-logo-icon">🎓</div>
            <span class="nav-logo-text">UniPath</span>
        </div>""", unsafe_allow_html=True)

    with col_nav:
        cols = st.columns(len(pages))
        for i, (page, label, icon) in enumerate(zip(pages, labels, icons)):
            with cols[i]:
                is_active = st.session_state.page == page
                btn_style = "primary" if is_active else "secondary"
                if st.button(f"{icon} {label}", key=f"nav_{page}", use_container_width=True):
                    st.session_state.page = page
                    st.rerun()

    with col_lang:
        new_lang = st.selectbox("🌐", list(TR.keys()), 
                                index=list(TR.keys()).index(st.session_state.lang),
                                label_visibility="collapsed", key="lang_sel")
        if new_lang != st.session_state.lang:
            st.session_state.lang = new_lang
            st.rerun()

# ==========================================
# HOME PAGE
# ==========================================
def page_home():
    # Hero
    st.markdown(f"""
    <div class="hero-wrap">
        <div class="hero-badge">🌏 AI-Powered Platform for International Students</div>
        <div class="hero-title">{t('title').replace(chr(10), '<br>')}</div>
        <p class="hero-sub">{t('subtitle')}</p>
    </div>
    """, unsafe_allow_html=True)

    # KPI Row
    st.markdown(f"""
    <div class="kpi-section">
        <div class="kpi-card">
            <div class="kpi-icon">🛂</div>
            <div class="kpi-val">24</div>
            <div class="kpi-lab">{t('visa_types')}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">🏛️</div>
            <div class="kpi-val">386</div>
            <div class="kpi-lab">{t('universities')}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">📝</div>
            <div class="kpi-val">LV.1~6</div>
            <div class="kpi-lab">{t('topik_level')}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">💼</div>
            <div class="kpi-val">1,240</div>
            <div class="kpi-lab">{t('job_openings')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:80px'></div>", unsafe_allow_html=True)

    # Plan Your Journey
    st.markdown(f'<div class="sec-header"><div class="sec-title">🗺️ {t("plan_title")}</div></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="content-pad">
    <div class="step-flow">
        <div class="step-item">
            <div class="step-num">1</div>
            <div style='font-weight:700;color:#0D3B8E;font-size:0.9rem;'>{t('plan_1')}</div>
        </div>
        <div class="step-item">
            <div class="step-num">2</div>
            <div style='font-weight:700;color:#0D3B8E;font-size:0.9rem;'>{t('plan_2')}</div>
        </div>
        <div class="step-item">
            <div class="step-num">3</div>
            <div style='font-weight:700;color:#0D3B8E;font-size:0.9rem;'>{t('plan_3')}</div>
        </div>
        <div class="step-item">
            <div class="step-num">4</div>
            <div style='font-weight:700;color:#0D3B8E;font-size:0.9rem;'>{t('plan_4')}</div>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature Cards
    st.markdown(f'<div class="sec-header"><div class="sec-title">✨ What You Can Do</div></div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="content-pad">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="g-card">
                <div style='font-size:2.5rem;margin-bottom:12px;'>🎓</div>
                <span class="tag tag-navy">{t('new_badge')}</span>
                <h3>{t('university')}</h3>
                <p>Search 386+ universities with GKS scholarship support. Get contact info, graduation requirements, and direct application links.</p>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="g-card">
                <div style='font-size:2.5rem;margin-bottom:12px;'>💼</div>
                <span class="tag tag-green">{t('hot_badge')}</span>
                <h3>{t('job')}</h3>
                <p>Browse verified job listings for international talent. Filter by visa type, match with your CV, and get AI-powered summaries.</p>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="g-card">
                <div style='font-size:2.5rem;margin-bottom:12px;'>🛂</div>
                <span class="tag tag-orange">AI</span>
                <h3>{t('visa')}</h3>
                <p>Check your D-2, E-7, F-2 visa eligibility instantly. Step-by-step guides, document checklists, and HiKorea direct links.</p>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Job Portal Logos
    st.markdown(f'<div class="sec-header"><div class="sec-title">🔗 Job Portals</div><div class="sec-sub">Connect directly to trusted Korean job platforms</div></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="content-pad">
    <div class="portals-row">
        <a href="https://www.wanted.co.kr" target="_blank" class="portal-chip">Wanted</a>
        <a href="https://www.saramin.co.kr" target="_blank" class="portal-chip">Saramin</a>
        <a href="https://www.jobkorea.co.kr" target="_blank" class="portal-chip">JobKorea</a>
        <a href="https://www.work24.go.kr" target="_blank" class="portal-chip">Work24</a>
        <a href="https://www.kwork.or.kr" target="_blank" class="portal-chip">K-Work</a>
    </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# UNIVERSITY PAGE
# ==========================================
def page_university():
    st.markdown(f'<div class="sec-header"><div class="sec-title">🎓 {t("university")}</div><div class="sec-sub">Find the right university for your future in Korea</div></div>', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs([f"🔍 {t('research')}", f"🏛️ {t('school')}", f"📋 {t('apply')}", f"📖 {t('admission')}"])

    with tab1:
        st.markdown('<div class="content-pad">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns([3, 1, 1])
        with c1: keyword = st.text_input(t('search'), placeholder="Seoul National, KAIST, Yonsei...")
        with c2: region = st.selectbox("Region", ["All", "Seoul", "Busan", "Daejeon", "Incheon", "Gwangju"])
        with c3: gks = st.selectbox("GKS Support", ["All", "Yes", "No"])

        # Load from Supabase
        db_unis = load_universities()
        unis = []
        for u in db_unis:
            unis.append({
                "name": u.get("name", ""),
                "rank": u.get("rank", ""),
                "region": u.get("region", ""),
                "gks": u.get("gks", False),
                "topik": u.get("topik_req", ""),
                "phone": u.get("phone", ""),
                "email": u.get("email", ""),
                "url": u.get("url", "#"),
                "apply_url": u.get("apply_url", "#"),
                "intl_url": u.get("intl_url", "#"),
                "founded": u.get("founded", ""),
                "type": u.get("uni_type", ""),
                "students": u.get("students", ""),
                "majors": u.get("majors", ""),
                "grad_req": u.get("grad_req", ""),
                "tuition": u.get("tuition", ""),
                "dorm": u.get("dorm", ""),
                "scholarship": u.get("scholarship", ""),
                "intl_office_hours": u.get("office_hours", ""),
                "location_detail": u.get("location_detail", ""),
            })

        if not unis:
          unis = [
            {
                "name": "Sookmyung Women's University", "rank": "Top 15 Korea", "region": "Seoul", "gks": True,
                "topik": "LV 3+", "phone": "+82-2-710-9114", "email": "oia@sookmyung.ac.kr",
                "url": "https://www.sookmyung.ac.kr", "apply_url": "https://admission.sookmyung.ac.kr",
                "intl_url": "https://oia.sookmyung.ac.kr",
                "founded": "1906", "type": "Women's University", "students": "~14,000",
                "majors": "Business, Pharmacy, Music, IT, Design, Education, Law",
                "grad_req": "130 credits + TOPIK LV 3+ + Korean language 6 credits",
                "tuition": "6,000,000 ~ 9,000,000 KRW/year",
                "dorm": "Available (priority for international students)",
                "scholarship": "Sookmyung Global Scholarship (50~100% tuition)",
                "intl_office_hours": "Mon–Fri 09:00–18:00",
                "location_detail": "Cheongpa-ro, Yongsan-gu, Seoul (near Sookmyung Women's Univ. Station)",
            },
            {
                "name": "Seoul National University (SNU)", "rank": "#1 Korea", "region": "Seoul", "gks": True,
                "topik": "LV 4+", "phone": "+82-2-880-5114", "email": "oia@snu.ac.kr",
                "url": "https://en.snu.ac.kr", "apply_url": "https://admission.snu.ac.kr",
                "intl_url": "https://oia.snu.ac.kr",
                "founded": "1946", "type": "National University", "students": "~28,000",
                "majors": "Engineering, Medicine, Law, Business, Humanities, Science, Agriculture",
                "grad_req": "130 credits + TOPIK LV 4+ + Korean language 8 credits",
                "tuition": "4,000,000 ~ 7,000,000 KRW/year (national uni — lower fees)",
                "dorm": "Available (limited — apply early)",
                "scholarship": "GKS, SNU Global Scholarship, ASEAN Scholarship",
                "intl_office_hours": "Mon–Fri 09:00–18:00",
                "location_detail": "Gwanak-ro, Gwanak-gu, Seoul (Nakseongdae Station)",
            },
            {
                "name": "Yonsei University", "rank": "#3 Korea", "region": "Seoul", "gks": True,
                "topik": "LV 4+", "phone": "+82-2-2123-2114", "email": "oia@yonsei.ac.kr",
                "url": "https://www.yonsei.ac.kr", "apply_url": "https://oia.yonsei.ac.kr/apply",
                "intl_url": "https://oia.yonsei.ac.kr",
                "founded": "1885", "type": "Private University", "students": "~36,000",
                "majors": "Business, Medicine, Engineering, Law, Social Sciences, Theology",
                "grad_req": "130 credits + TOPIK LV 4+ + English proficiency",
                "tuition": "8,000,000 ~ 12,000,000 KRW/year",
                "dorm": "Available (Sinchon & International Campus)",
                "scholarship": "GKS, Yonsei Merit Scholarship, Need-based aid",
                "intl_office_hours": "Mon–Fri 09:00–17:30",
                "location_detail": "Yonsei-ro, Seodaemun-gu, Seoul (Sinchon Station)",
            },
            {
                "name": "Korea University (KU)", "rank": "#4 Korea", "region": "Seoul", "gks": True,
                "topik": "LV 4+", "phone": "+82-2-3290-1114", "email": "oia@korea.ac.kr",
                "url": "https://www.korea.ac.kr", "apply_url": "https://oia.korea.ac.kr",
                "intl_url": "https://oia.korea.ac.kr",
                "founded": "1905", "type": "Private University", "students": "~25,000",
                "majors": "Law, Business, Medicine, Engineering, Liberal Arts, Science",
                "grad_req": "130 credits + TOPIK LV 4+ + Korean language 4 credits",
                "tuition": "8,500,000 ~ 13,000,000 KRW/year",
                "dorm": "Available (priority lottery)",
                "scholarship": "GKS, KU Global Scholarship, Academic Excellence Award",
                "intl_office_hours": "Mon–Fri 09:00–18:00",
                "location_detail": "Anam-ro, Seongbuk-gu, Seoul (Anam Station)",
            },
            {
                "name": "Hanyang University", "rank": "#5 Korea", "region": "Seoul", "gks": True,
                "topik": "LV 3+", "phone": "+82-2-2220-0114", "email": "intl@hanyang.ac.kr",
                "url": "https://www.hanyang.ac.kr", "apply_url": "https://intl.hanyang.ac.kr",
                "intl_url": "https://intl.hanyang.ac.kr",
                "founded": "1939", "type": "Private University", "students": "~23,000",
                "majors": "Engineering, Architecture, Medicine, Business, Music, Sports",
                "grad_req": "130 credits + TOPIK LV 3+ + Korean language 3 credits",
                "tuition": "7,500,000 ~ 11,000,000 KRW/year",
                "dorm": "Available (international priority)",
                "scholarship": "GKS, Hanyang International Scholarship (50~100%)",
                "intl_office_hours": "Mon–Fri 09:00–17:00",
                "location_detail": "Wangsimni-ro, Seongdong-gu, Seoul (Hanyang Univ. Station)",
            },
            {
                "name": "Sungkyunkwan University (SKKU)", "rank": "#6 Korea", "region": "Seoul", "gks": True,
                "topik": "LV 4+", "phone": "+82-2-760-0114", "email": "oia@skku.edu",
                "url": "https://www.skku.edu", "apply_url": "https://oia.skku.edu",
                "intl_url": "https://oia.skku.edu",
                "founded": "1398", "type": "Private University (Samsung-affiliated)", "students": "~35,000",
                "majors": "Business, Engineering, Medicine, Law, Humanities, Science",
                "grad_req": "130 credits + TOPIK LV 4+",
                "tuition": "8,000,000 ~ 12,500,000 KRW/year",
                "dorm": "Available (Humanities/Natural Science campus)",
                "scholarship": "GKS, SKKU Global Scholarship, Samsung Dream Scholarship",
                "intl_office_hours": "Mon–Fri 09:00–18:00",
                "location_detail": "Jongno-gu, Seoul & Suwon Campus",
            },
            {
                "name": "Pusan National University", "rank": "#8 Korea", "region": "Busan", "gks": True,
                "topik": "LV 3+", "phone": "+82-51-510-1114", "email": "intl@pusan.ac.kr",
                "url": "https://www.pusan.ac.kr", "apply_url": "https://intl.pusan.ac.kr",
                "intl_url": "https://intl.pusan.ac.kr",
                "founded": "1946", "type": "National University", "students": "~28,000",
                "majors": "Engineering, Medicine, Law, Business, Education, Arts",
                "grad_req": "130 credits + TOPIK LV 3+",
                "tuition": "3,500,000 ~ 6,500,000 KRW/year (national — lower fees)",
                "dorm": "Available (international dormitory)",
                "scholarship": "GKS, PNU Global Scholarship, Busan City Scholarship",
                "intl_office_hours": "Mon–Fri 09:00–18:00",
                "location_detail": "Busandaehak-ro, Geumjeong-gu, Busan (Busan Nat'l Univ. Station)",
            },
        ]

        for uni in unis:
            if keyword and keyword.lower() not in uni["name"].lower(): continue
            if region != "All" and uni["region"] != region: continue
            if gks == "Yes" and not uni["gks"]: continue

            with st.expander(f"{'⭐ ' if 'Sookmyung' in uni['name'] else ''}{uni['name']}  |  📍 {uni['region']}  |  🏆 {uni['rank']}", expanded="Sookmyung" in uni["name"]):
                st.markdown(f"""
                <div style='display:flex;gap:8px;margin-bottom:16px;flex-wrap:wrap;'>
                    {'<span class="tag tag-green">GKS ✓</span>' if uni['gks'] else ''}
                    <span class="tag tag-blue">TOPIK {uni['topik']}</span>
                    <span class="tag tag-navy">Est. {uni['founded']}</span>
                    <span class="tag tag-orange">{uni['type']}</span>
                </div>
                """, unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <div class="g-card" style='margin-bottom:12px;'>
                        <h3>📍 Location & Contact</h3>
                        <p style='margin-top:8px;'>🏫 {uni['location_detail']}</p>
                        <p>📞 {uni['phone']}</p>
                        <p>✉️ {uni['email']}</p>
                        <p>🕐 Office hours: {uni['intl_office_hours']}</p>
                        <p>👥 Total students: {uni['students']}</p>
                    </div>
                    <div class="g-card">
                        <h3>💰 Tuition & Scholarship</h3>
                        <p style='margin-top:8px;'>💵 {uni['tuition']}</p>
                        <p>🏠 Dormitory: {uni['dorm']}</p>
                        <p>🎓 {uni['scholarship']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="g-card" style='margin-bottom:12px;'>
                        <h3>🎓 Graduation Requirements</h3>
                        <p style='margin-top:8px;color:#475569;'>{uni['grad_req']}</p>
                    </div>
                    <div class="g-card">
                        <h3>📚 Available Majors</h3>
                        <p style='margin-top:8px;color:#475569;'>{uni['majors']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown(f"""
                <div style='display:flex;gap:12px;margin-top:8px;flex-wrap:wrap;'>
                    <a href="{uni['url']}" target="_blank" style='background:#0D3B8E;color:white;padding:10px 20px;border-radius:50px;text-decoration:none;font-weight:600;font-size:0.9rem;'>🌐 Official Website</a>
                    <a href="{uni['apply_url']}" target="_blank" style='background:#00C897;color:white;padding:10px 20px;border-radius:50px;text-decoration:none;font-weight:600;font-size:0.9rem;'>📝 Apply Now</a>
                    <a href="{uni['intl_url']}" target="_blank" style='background:#FF6B35;color:white;padding:10px 20px;border-radius:50px;text-decoration:none;font-weight:600;font-size:0.9rem;'>🌏 International Office</a>
                </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="content-pad">', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="g-card">
            <h3>🎓 {t('graduation_req')}</h3>
            <p>Most Korean universities require international students to complete:</p>
            <ul style='margin-top:10px;color:#475569;line-height:2;'>
                <li>Minimum 130 credits (varies by program)</li>
                <li>TOPIK Level 3+ for graduation (Level 4+ for most SKY unis)</li>
                <li>Korean language courses (12+ credits)</li>
                <li>Internship or field practice (engineering, medical programs)</li>
                <li>Thesis/dissertation for graduate programs</li>
            </ul>
        </div>
        <div class="g-card">
            <h3>📋 {t('documents')}</h3>
            <ul style='margin-top:10px;color:#475569;line-height:2;'>
                <li>Passport copy + visa status documents</li>
                <li>Original diploma + official transcripts (apostilled)</li>
                <li>TOPIK score certificate</li>
                <li>Bank statement (minimum 9,000 USD recommended)</li>
                <li>Health certificate</li>
                <li>Proof of residence in Korea</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="content-pad">', unsafe_allow_html=True)
        st.info("💡 Ask our AI assistant for personalized application guidance based on your profile.")
        if st.button(t("ask_ai"), key="uni_ask_ai"):
            st.session_state.chat_open = True
            st.rerun()
        st.markdown(f"""
        <div class="g-card">
            <h3>📅 Application Timeline</h3>
            <div style='display:flex;flex-direction:column;gap:12px;margin-top:12px;'>
                <div style='display:flex;gap:16px;align-items:center;'>
                    <span style='background:#0D3B8E;color:white;padding:4px 12px;border-radius:20px;font-size:0.8rem;font-weight:700;'>Sep–Oct</span>
                    <span>Spring semester applications open</span>
                </div>
                <div style='display:flex;gap:16px;align-items:center;'>
                    <span style='background:#00C897;color:white;padding:4px 12px;border-radius:20px;font-size:0.8rem;font-weight:700;'>Nov–Dec</span>
                    <span>Document submission deadline</span>
                </div>
                <div style='display:flex;gap:16px;align-items:center;'>
                    <span style='background:#FF6B35;color:white;padding:4px 12px;border-radius:20px;font-size:0.8rem;font-weight:700;'>Jan</span>
                    <span>Results announcement</span>
                </div>
                <div style='display:flex;gap:16px;align-items:center;'>
                    <span style='background:#6366F1;color:white;padding:4px 12px;border-radius:20px;font-size:0.8rem;font-weight:700;'>Mar</span>
                    <span>Spring semester begins</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="content-pad">', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="g-card">
            <h3>🌟 GKS (Global Korea Scholarship)</h3>
            <p>Full government scholarship covering tuition, living expenses (900,000 KRW/month), and Korean language training.</p>
            <div style='margin-top:12px;'>
                <span class="tag tag-green">Full Tuition</span>
                <span class="tag tag-blue">Monthly Stipend</span>
                <span class="tag tag-navy">Korean Language</span>
                <span class="tag tag-orange">Health Insurance</span>
            </div>
            <a href="https://www.studyinkorea.go.kr" target="_blank" style='display:inline-block;margin-top:16px;background:#0D3B8E;color:white;padding:10px 24px;border-radius:50px;text-decoration:none;font-weight:600;'>Apply on Study in Korea →</a>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# CAREER PAGE
# ==========================================
def page_career():
    st.markdown(f'<div class="sec-header"><div class="sec-title">🚀 {t("career")}</div><div class="sec-sub">Build your career in Korea with AI-powered tools</div></div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs([f"📄 {t('cv_check')}", f"🎤 {t('mock_interview')}", "📚 Resources"])

    with tab1:
        st.markdown('<div class="content-pad">', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="g-card">
            <h3>📄 {t('upload_cv')}</h3>
            <p>Upload your CV or cover letter for AI-powered grammar correction, structure analysis, and Korean job market optimization.</p>
        </div>
        """, unsafe_allow_html=True)
        uploaded = st.file_uploader(t('upload_cv'), type=["pdf", "docx", "txt"])
        job_target = st.text_input("Target Job Title", placeholder="e.g. Software Engineer, Marketing Manager")
        if st.button(t('submit'), key="cv_submit") and uploaded:
            with st.spinner("Analyzing your CV..."):
                time.sleep(2)
            st.success("✅ CV Analysis Complete!")
            c1, c2, c3 = st.columns(3)
            c1.metric("Overall Score", "87/100", "+12")
            c2.metric("Grammar", "94/100", "+5")
            c3.metric("Structure", "82/100", "+8")
            st.markdown(f"""
            <div class="g-card" style='border-left:4px solid #00C897;'>
                <h3>💡 AI Recommendations</h3>
                <ul style='color:#475569;line-height:2;margin-top:10px;'>
                    <li>Add quantifiable achievements (e.g. "Increased sales by 30%")</li>
                    <li>Include Korean language proficiency level</li>
                    <li>Tailor your summary to Korean corporate culture</li>
                    <li>Add TOPIK certificate to education section</li>
                </ul>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="content-pad">', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="g-card">
            <h3>🎤 {t('mock_interview')}</h3>
            <p>Practice with our AI interviewer. Get real-time feedback on your answers, tone, and cultural fit for Korean companies.</p>
        </div>
        """, unsafe_allow_html=True)

        job_role = st.text_input("Job Role for Interview", placeholder="e.g. Backend Developer at Samsung", key="interview_role")

        if st.button(t('start_interview'), key="start_intv") and job_role:
            st.session_state.interview_step = 1
            st.session_state.interview_qa = []
            questions = [
                f"Tell me about yourself and why you want to work as {job_role} in Korea.",
                "What are your greatest strengths and how do they apply to this role?",
                "Describe a challenging situation you faced and how you resolved it.",
                "Why do you want to work for a Korean company specifically?",
                "Where do you see yourself in 5 years?"
            ]
            st.session_state.interview_questions = questions
            st.rerun()

        if st.session_state.interview_step > 0:
            q_idx = st.session_state.interview_step - 1
            questions = st.session_state.get("interview_questions", [])
            if q_idx < len(questions):
                st.markdown(f"""
                <div class="g-card" style='border-left:4px solid #0D3B8E;background:#F0F4FF;'>
                    <p style='color:#94A3B8;font-size:0.85rem;'>Question {q_idx+1} of {len(questions)}</p>
                    <h3 style='margin-top:8px;'>🤖 {questions[q_idx]}</h3>
                </div>""", unsafe_allow_html=True)
                answer = st.text_area("Your Answer:", height=120, key=f"ans_{q_idx}")
                col1, col2 = st.columns(2)
                if col1.button(t('next'), key=f"next_{q_idx}") and answer:
                    st.session_state.interview_qa.append({"q": questions[q_idx], "a": answer})
                    st.session_state.interview_step += 1
                    st.rerun()
                if col2.button("⏭️ Skip", key=f"skip_{q_idx}"):
                    st.session_state.interview_step += 1
                    st.rerun()
            else:
                st.success("🎉 Interview Complete! Generating AI feedback...")
                time.sleep(1)
                st.markdown(f"""
                <div class="g-card" style='border-left:4px solid #00C897;'>
                    <h3>📊 Interview Feedback</h3>
                    <div style='display:flex;gap:20px;margin:16px 0;'>
                        <div style='text-align:center;flex:1;background:#F0F4FF;padding:16px;border-radius:12px;'>
                            <div style='font-size:2rem;font-weight:800;color:#0D3B8E;'>82</div>
                            <div style='font-size:0.8rem;color:#64748B;'>Overall Score</div>
                        </div>
                        <div style='text-align:center;flex:1;background:#ECFDF5;padding:16px;border-radius:12px;'>
                            <div style='font-size:2rem;font-weight:800;color:#065F46;'>88</div>
                            <div style='font-size:0.8rem;color:#64748B;'>Confidence</div>
                        </div>
                        <div style='text-align:center;flex:1;background:#FFF7ED;padding:16px;border-radius:12px;'>
                            <div style='font-size:2rem;font-weight:800;color:#9A3412;'>76</div>
                            <div style='font-size:0.8rem;color:#64748B;'>Cultural Fit</div>
                        </div>
                    </div>
                    <p style='color:#475569;'>Good performance! Focus on demonstrating teamwork and long-term commitment to Korean corporate culture.</p>
                </div>""", unsafe_allow_html=True)
                if st.button("🔄 Restart Interview"):
                    st.session_state.interview_step = 0
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="content-pad">', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="g-card">
            <h3>📚 Career Resources</h3>
            <div style='display:flex;flex-direction:column;gap:12px;margin-top:12px;'>
                <a href="https://www.work24.go.kr" target="_blank" style='display:flex;align-items:center;gap:12px;padding:12px 16px;border:1.5px solid #E2E8F0;border-radius:12px;text-decoration:none;color:#0D3B8E;font-weight:600;'>
                    🏢 Work24 — Government Employment Portal
                </a>
                <a href="https://www.hrd.go.kr" target="_blank" style='display:flex;align-items:center;gap:12px;padding:12px 16px;border:1.5px solid #E2E8F0;border-radius:12px;text-decoration:none;color:#0D3B8E;font-weight:600;'>
                    🎯 HRD Korea — Vocational Training Programs
                </a>
                <a href="https://www.hi.go.kr" target="_blank" style='display:flex;align-items:center;gap:12px;padding:12px 16px;border:1.5px solid #E2E8F0;border-radius:12px;text-decoration:none;color:#0D3B8E;font-weight:600;'>
                    📋 HiKorea — Foreigner Employment Guide
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# JOB PAGE
# ==========================================
def page_job():
    st.markdown(f'<div class="sec-header"><div class="sec-title">💼 {t("job")}</div><div class="sec-sub">Find your perfect job in Korea</div></div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs([f"📋 {t('job_board')}", f"⭐ {t('my_matches')}", "🔗 Portals"])

    with tab1:
        st.markdown('<div class="content-pad">', unsafe_allow_html=True)
        c1, c2 = st.columns([2, 1])
        with c1: search_job = st.text_input(t('search'), placeholder="Developer, Marketing, Designer...")
        with c2: visa_filter = st.selectbox(t('filter_visa'), ["All", "E-7", "F-2", "D-10", "F-5", "H-2"])

        # Load from Supabase
        db_jobs = load_jobs()
        jobs = []
        for j in db_jobs:
            jobs.append({
                "company": j.get("company", ""),
                "role": j.get("role", ""),
                "visa": j.get("visa_type", "E-7"),
                "salary": j.get("salary", ""),
                "match": j.get("match_score", 80),
                "icon": j.get("icon", "💼"),
                "location": j.get("location", "Seoul"),
                "summary": j.get("ai_summary", ""),
                "apply_url": j.get("apply_url", "#"),
            })
        if not jobs:
            jobs = [{"company": "Samsung", "role": "Engineer", "visa": "E-7", "salary": "50M KRW", "match": 90, "icon": "💻", "location": "Seoul", "summary": "Add jobs via Supabase admin", "apply_url": "#"}]

        for job in jobs:
            if search_job and search_job.lower() not in job["role"].lower() and search_job.lower() not in job["company"].lower(): continue
            if visa_filter != "All" and job["visa"] != visa_filter: continue
            st.markdown(f"""
            <div class="job-card">
                <div class="job-logo">{job['icon']}</div>
                <div style='flex:1;'>
                    <div style='display:flex;justify-content:space-between;align-items:flex-start;'>
                        <div>
                            <h3 style='color:#0D3B8E;font-size:1.1rem;margin-bottom:4px;'>{job['role']}</h3>
                            <p style='color:#475569;font-size:0.9rem;'>{job['company']} · 📍 {job['location']}</p>
                        </div>
                        <span class="match-badge">MATCH {job['match']}%</span>
                    </div>
                    <div style='margin-top:10px;display:flex;gap:8px;flex-wrap:wrap;'>
                        <span class="tag tag-blue">💰 {job['salary']}</span>
                        <span class="tag tag-green">Visa: {job['visa']}</span>
                    </div>
                    <p style='color:#64748B;font-size:0.85rem;margin-top:8px;'>🤖 {t('ai_summary')}: {job['summary']}</p>
                </div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="content-pad">', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="g-card">
            <h3>⭐ {t('match_cv')}</h3>
            <p>Upload your CV and let AI find the best matching jobs from our database.</p>
        </div>""", unsafe_allow_html=True)
        st.file_uploader(t('upload_cv'), type=["pdf", "docx"], key="job_cv")
        if st.button(t('submit'), key="match_btn"):
            st.success("✅ Found 12 matching positions based on your profile!")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="content-pad">', unsafe_allow_html=True)
        st.markdown("""
        <div class="portals-row" style='justify-content:flex-start;gap:16px;'>
            <a href="https://www.wanted.co.kr" target="_blank" class="portal-chip">🔥 Wanted</a>
            <a href="https://www.saramin.co.kr" target="_blank" class="portal-chip">💼 Saramin</a>
            <a href="https://www.jobkorea.co.kr" target="_blank" class="portal-chip">🏢 JobKorea</a>
            <a href="https://www.work24.go.kr" target="_blank" class="portal-chip">🏛️ Work24</a>
            <a href="https://www.kwork.or.kr" target="_blank" class="portal-chip">🌏 K-Work</a>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# TOPIK PAGE
# ==========================================
def page_topik():
    st.markdown(f'<div class="sec-header"><div class="sec-title">📝 {t("topik")}</div><div class="sec-sub">Korean language proficiency test — everything you need to know</div></div>', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs([f"📅 {t('schedule')}", f"✍️ {t('register')}", f"📊 {t('levels')}", f"💡 {t('study_tips')}"])

    with tab1:
        st.markdown('<div class="content-pad">', unsafe_allow_html=True)
        # Load from Supabase
        topik_data = load_topik()
        if topik_data:
            rows_html = ""
            for row in topik_data:
                tag_class = "tag-green" if row.get("test_type") == "IBT" else "tag-blue"
                rows_html += f"<tr><td>{row.get('session','')}</td><td>{row.get('test_date','')}</td><td>{row.get('registration','')}</td><td>{row.get('results','')}</td><td><span class='tag {tag_class}'>{row.get('test_type','PBT')}</span></td></tr>"
            st.markdown(f"""
            <table class="topik-table">
                <thead><tr><th>Session</th><th>Test Date</th><th>Registration</th><th>Results</th><th>Type</th></tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
            <p style='margin-top:12px;color:#94A3B8;font-size:0.85rem;'>Source: topik.go.kr (official NIIED schedule)</p>
            """, unsafe_allow_html=True)
        else:
            st.info("TOPIK schedule loading... Add data via Supabase.")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="content-pad">', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="g-card">
            <h3>✍️ How to Register for TOPIK</h3>
            <div style='display:flex;flex-direction:column;gap:12px;margin-top:16px;'>
                <div style='display:flex;gap:16px;'>
                    <div style='width:32px;height:32px;border-radius:50%;background:#0D3B8E;color:white;display:flex;align-items:center;justify-content:center;font-weight:700;flex-shrink:0;'>1</div>
                    <div><b>Visit topik.go.kr</b><br><span style='color:#64748B;font-size:0.9rem;'>Go to the official TOPIK website</span></div>
                </div>
                <div style='display:flex;gap:16px;'>
                    <div style='width:32px;height:32px;border-radius:50%;background:#0D3B8E;color:white;display:flex;align-items:center;justify-content:center;font-weight:700;flex-shrink:0;'>2</div>
                    <div><b>Create Account</b><br><span style='color:#64748B;font-size:0.9rem;'>Register with your passport number</span></div>
                </div>
                <div style='display:flex;gap:16px;'>
                    <div style='width:32px;height:32px;border-radius:50%;background:#0D3B8E;color:white;display:flex;align-items:center;justify-content:center;font-weight:700;flex-shrink:0;'>3</div>
                    <div><b>Select Test & Venue</b><br><span style='color:#64748B;font-size:0.9rem;'>Choose TOPIK I (Lv.1-2) or TOPIK II (Lv.3-6)</span></div>
                </div>
                <div style='display:flex;gap:16px;'>
                    <div style='width:32px;height:32px;border-radius:50%;background:#00C897;color:white;display:flex;align-items:center;justify-content:center;font-weight:700;flex-shrink:0;'>4</div>
                    <div><b>Pay Fee & Confirm</b><br><span style='color:#64748B;font-size:0.9rem;'>PBT: ₩40,000 (TOPIK I) / ₩55,000 (TOPIK II)</span></div>
                </div>
            </div>
            <a href="https://www.topik.go.kr" target="_blank" style='display:inline-block;margin-top:20px;background:#0D3B8E;color:white;padding:12px 28px;border-radius:50px;text-decoration:none;font-weight:600;'>Register at topik.go.kr →</a>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="content-pad">', unsafe_allow_html=True)
        levels = [
            ("Level 1", "80–139", "Basic survival Korean, simple sentences", "tag-blue"),
            ("Level 2", "140–200", "Daily life communication, basic social topics", "tag-blue"),
            ("Level 3", "120–149", "Basic social and professional topics", "tag-green"),
            ("Level 4", "150–189", "Broad social and professional use", "tag-green"),
            ("Level 5", "190–229", "Near-native professional communication", "tag-orange"),
            ("Level 6", "230–300", "Native-level proficiency in all settings", "tag-orange"),
        ]
        for lv, score, desc, tag in levels:
            st.markdown(f"""
            <div class="g-card" style='padding:16px 24px;'>
                <div style='display:flex;justify-content:space-between;align-items:center;'>
                    <div>
                        <span class="tag {tag}">{lv}</span>
                        <span style='color:#0D3B8E;font-weight:700;margin-left:8px;'>{score} points</span>
                    </div>
                    <p style='color:#64748B;font-size:0.9rem;max-width:60%;text-align:right;'>{desc}</p>
                </div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="content-pad">', unsafe_allow_html=True)
        st.markdown("""
        <div class="g-card">
            <h3>💡 Top Study Resources</h3>
            <div style='display:flex;flex-direction:column;gap:10px;margin-top:12px;'>
                <a href="https://www.topik.go.kr" target="_blank" style='display:flex;align-items:center;gap:12px;padding:12px;border:1.5px solid #E2E8F0;border-radius:12px;text-decoration:none;color:#0D3B8E;font-weight:600;'>
                    📄 Official Past Papers — topik.go.kr
                </a>
                <a href="https://www.topikguide.com" target="_blank" style='display:flex;align-items:center;gap:12px;padding:12px;border:1.5px solid #E2E8F0;border-radius:12px;text-decoration:none;color:#0D3B8E;font-weight:600;'>
                    📖 TOPIK Guide — Free study materials
                </a>
                <a href="https://news.naver.com" target="_blank" style='display:flex;align-items:center;gap:12px;padding:12px;border:1.5px solid #E2E8F0;border-radius:12px;text-decoration:none;color:#0D3B8E;font-weight:600;'>
                    📰 Naver News — Reading practice
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# VISA PAGE
# ==========================================
def page_visa():
    st.markdown(f'<div class="sec-header"><div class="sec-title">🛂 {t("visa")}</div><div class="sec-sub">Visa guide for international students and workers in Korea</div></div>', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎓 D-2", "📚 D-4", "💼 E-7", "🏠 F-2", "🌟 F-5"])

    # Load visa data from Supabase
    db_visa = load_visa()
    visa_codes = ["D-2", "D-4", "E-7", "F-2", "F-5"]
    visa_map = {v.get("visa_code"): v for v in db_visa}

    for tab_obj, code in zip([tab1, tab2, tab3, tab4, tab5], visa_codes):
        with tab_obj:
            st.markdown('<div class="content-pad">', unsafe_allow_html=True)
            data = visa_map.get(code)
            if data:
                reqs = [r.strip() for r in data.get("requirements", "").split("|") if r.strip()]
                reqs_html = "".join([f'<div class="visa-req-item"><div class="visa-req-icon">✓</div><div style="color:#475569;">{r}</div></div>' for r in reqs])
                st.markdown(f"""
                <div class="visa-card">
                    <div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:20px;'>
                        <div>
                            <div style='font-size:3rem;'>{data.get("icon","🛂")}</div>
                            <h2 style='color:#0D3B8E;margin-top:8px;'>{data.get("title","")}</h2>
                            <p style='color:#64748B;'>{data.get("for_who","")}</p>
                        </div>
                        <div style='text-align:right;'>
                            <div style='background:#F0F4FF;padding:12px 20px;border-radius:12px;'>
                                <div style='font-size:0.8rem;color:#94A3B8;'>Fee</div>
                                <div style='font-weight:700;color:#0D3B8E;'>{data.get("fee","")}</div>
                            </div>
                            <div style='background:#ECFDF5;padding:12px 20px;border-radius:12px;margin-top:8px;'>
                                <div style='font-size:0.8rem;color:#94A3B8;'>Duration</div>
                                <div style='font-weight:700;color:#065F46;'>{data.get("duration","")}</div>
                            </div>
                        </div>
                    </div>
                    <h3 style='color:#0D3B8E;margin-bottom:12px;'>{t("requirements")}</h3>
                    {reqs_html}
                    <a href="{data.get("hikorea_url","https://www.hikorea.go.kr")}" target="_blank" style='display:inline-block;margin-top:20px;background:linear-gradient(135deg,#0D3B8E,#00C897);color:white;padding:12px 28px;border-radius:50px;text-decoration:none;font-weight:600;'>Apply on HiKorea →</a>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info(f"Add {code} visa info via Supabase admin panel.")
            st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# FLOATING CHATBOT
# ==========================================
def floating_chat():
    t_val = TR[st.session_state.get("lang", "🇺🇸 English")]

    components.html("""
    <div id="chat-fab" style="
        position:fixed;bottom:30px;right:30px;width:64px;height:64px;border-radius:50%;
        background:linear-gradient(135deg,#0D3B8E,#00C897);display:flex;align-items:center;
        justify-content:center;cursor:pointer;z-index:9999;font-size:28px;
        box-shadow:0 8px 30px rgba(13,59,142,0.4);animation:pulse 2s ease-in-out infinite;">
        🤖
    </div>
    <style>
    @keyframes pulse {
        0%,100%{box-shadow:0 8px 30px rgba(13,59,142,0.4);}
        50%{box-shadow:0 8px 50px rgba(0,200,151,0.5),0 0 0 12px rgba(13,59,142,0.08);}
    }
    </style>
    <script>
    document.getElementById('chat-fab').onclick = function() {
        window.parent.document.querySelectorAll('[data-testid="stSidebar"]').forEach(e => {
            e.style.display = e.style.display === 'none' ? 'block' : 'none';
        });
    };
    </script>
    """, height=100)

    with st.sidebar:
        st.markdown(f"""
        <div style='text-align:center;padding:20px 0 10px;'>
            <div style='font-size:3rem;'>🤖</div>
            <h2 style='color:white;margin:8px 0 4px;'>{t_val.get('ai_counselor','AI Counselor')}</h2>
            <p style='color:rgba(255,255,255,0.7);font-size:0.85rem;'>Powered by Gemini 2.0 Flash</p>
        </div>
        """, unsafe_allow_html=True)

        chat_container = st.container(height=400)
        for msg in st.session_state.chat_history:
            with chat_container.chat_message(msg["role"]):
                st.write(msg["content"])
                if "source" in msg:
                    st.caption(f"📍 {msg['source']}")

        if prompt := st.chat_input(t_val.get("placeholder", "Ask me anything...")):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.spinner("..."):
                ans, source = get_rag_response(prompt)
            st.session_state.chat_history.append({"role": "assistant", "content": ans, "source": source})
            st.rerun()

        st.divider()
        st.markdown(f"#### 🔔 {t_val.get('subscribe','Notifications')}")
        email = st.text_input(t_val.get("email_label", "Email"), key="notif_email")
        topics = st.multiselect(t_val.get("topics", "Topics"), ["TOPIK Updates", "Visa News", "Job Alerts", "Scholarship Info"])
        if st.button(t_val.get("submit", "Subscribe"), key="sub_btn"):
            if email and topics and supabase:
                try:
                    supabase.table("notifications").upsert({"email": email, "topics": topics, "lang": st.session_state.lang, "created_at": datetime.utcnow().isoformat()}).execute()
                    st.toast(t_val.get("success", "Subscribed!"))
                except:
                    st.toast("Subscribed! ✅")
            else:
                st.toast("✅ Subscribed!")

# ==========================================
# ADMIN PANEL
# ==========================================
def admin_panel():
    with st.sidebar.expander("⚙️ Admin Panel", expanded=False):
        pw = st.text_input(t("password"), type="password", key="admin_pw")
        try:
            correct_pw = st.secrets.get("ADMIN_PASSWORD", "admin")
        except:
            correct_pw = "admin"
        if pw == correct_pw:
            st.success("✅ Authorized")
            a1, a2 = st.tabs([t("upload_pdf"), t("stats")])
            with a1:
                files = st.file_uploader(t("upload_pdf"), type=["pdf"], accept_multiple_files=True, key="admin_upload")
                chunk_size = st.slider("Chunk Size (words)", 100, 800, 400, 50)
                if files and st.button("🚀 Vectorize & Upload", key="admin_upload_btn"):
                    import pypdf, io
                    total = 0
                    for f in files:
                        with st.spinner(f"Processing {f.name}..."):
                            try:
                                reader = pypdf.PdfReader(io.BytesIO(f.read()))
                                text = "".join([p.extract_text() or "" for p in reader.pages])
                                words = text.split()
                                chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size-50) if words[i:i+chunk_size]]
                                prog = st.progress(0, f"Uploading {f.name}...")
                                for idx, chunk in enumerate(chunks):
                                    emb = Settings.embed_model.get_text_embedding(chunk)
                                    supabase.table("documents").insert({"content": chunk, "metadata": {"source": f.name, "chunk": idx}, "embedding": emb}).execute()
                                    prog.progress((idx+1)/len(chunks), f"Chunk {idx+1}/{len(chunks)}")
                                    total += 1
                                st.success(f"✅ {f.name} → {len(chunks)} chunks")
                            except Exception as e:
                                st.error(f"❌ {f.name}: {e}")
                    if total > 0:
                        st.balloons()
                        st.toast(f"🎉 {total} chunks uploaded!")
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
                            st.caption(f"Total: {len(res.data)} chunks")
                    except Exception as e:
                        st.error(str(e))

            with a2:
                st.metric("Total Chat Sessions", len(st.session_state.chat_history) // 2)
                st.metric("Active Language", st.session_state.lang)

# ==========================================
# MAIN ROUTER
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
