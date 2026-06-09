# 🎓 UniPath Korea — AI Guide for International Students

> 한국에서 공부·취업·생활하려는 외국인 유학생을 위한 **AI 기반 멀티링궐(9개 언어) 가이드 플랫폼**
> Agentic RAG 챗봇 · 대학/장학금/비자/TOPIK/취업/생활 정보 · 이메일 알림 · 관리자 지식베이스

[![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-FF4B4B)](https://streamlit.io)
[![Gemini](https://img.shields.io/badge/LLM-Gemini%202.0%20Flash-4285F4)](https://ai.google.dev)
[![Supabase](https://img.shields.io/badge/DB-Supabase%20pgvector-3ECF8E)](https://supabase.com)

---

## 🔗 제출 링크 (Submission Links)

| 항목 | 링크 |
|------|------|
| 🌐 배포 사이트 (Streamlit Cloud) | `https://<your-app>.streamlit.app` |
| 💻 GitHub Repository | `https://github.com/<you>/unipath-korea` |
| 🤖 AI 프롬프트 대화 내역 | `https://...` (로그인 없이 열람 가능) |

---

## 1. 문제 정의 및 배경 (Problem & Motivation)

한국 내 외국인 유학생은 **16만 명 이상**으로 매년 증가하지만, 이들이 필요로 하는 정보는
- 정부 사이트(hikorea.go.kr, topik.go.kr, studyinkorea.go.kr)에 **흩어져 있고**,
- 대부분 **한국어 위주**라 언어 장벽이 크며,
- 비자·TOPIK·취업·생활 정보가 **분절**되어 있어 한 곳에서 흐름을 파악하기 어렵습니다.

**UniPath Korea**는 이 문제를 해결하기 위해, 유학 여정(대학 선택 → TOPIK → 비자 → 취업 → 생활)을
**9개 언어로, AI 챗봇과 함께** 한 곳에서 안내합니다.

## 2. 데이터 및 시스템 구조 (Data & Architecture)

```
┌──────────────────────────────────────────────────────────────┐
│                     Streamlit (app.py · 단일 파일)              │
│  9개 언어 UI · 7개 페이지 · 사이드바 RAG 챗봇 · 관리자 패널        │
└───────────────┬───────────────────────────┬──────────────────┘
                │                           │
        ┌───────▼────────┐         ┌────────▼─────────┐
        │  Google Gemini  │         │     Supabase      │
        │  2.0 Flash (LLM)│         │  PostgreSQL +     │
        │  text-embedding │         │  pgvector         │
        │  -004 (임베딩)   │         │                   │
        └───────┬────────┘         │  • universities   │
                │                  │  • scholarships   │
        ┌───────▼────────┐         │  • jobs           │
        │   LlamaIndex    │◄───────│  • topik_*        │
        │   RAG 파이프라인 │         │  • visa_info      │
        │  (Vector Store) │         │  • news           │
        └────────────────┘         │  • documents(RAG) │
                                   │  • users / notif. │
                                   └───────────────────┘
```

**사용 데이터 (Supabase 테이블):**
`universities`, `scholarships`, `jobs`, `topik_schedule`, `topik_info`, `topik_structure`,
`topik_faq`, `topik_centers`, `visa_info`, `news`, `documents`(PDF 지식베이스), `users`, `notifications`, `chat_history`

## 3. 개발 내용 및 핵심 기능 (Features)

| 페이지 | 기능 |
|--------|------|
| **HOME** | 히어로 + KPI · 여정 플로우 · 공식 자료 · 통계 차트(Plotly) · 뉴스 |
| **UNIVERSITY** | 대학 검색/필터 · 학교 정보 · 지원 타임라인 · GKS/장학금 |
| **CAREER** | AI 이력서 분석 · AI 모의면접(5문항·피드백) · 취업 자료 |
| **JOB** | 잡 보드 · AI 매칭(CV 업로드) · 채용 포털 |
| **TOPIK** | 시험 일정 · 접수 가이드 · 등급 · 학습 팁 · 시험장 |
| **VISA** | D-2/D-4/E-7/F-2/F-5 비자 상세 |
| **LIFE** | 주거·교통·건강·은행·안전 생활 가이드 |

**AI / LLM 활용:**
- **RAG 챗봇 (UNI)** — Supabase 벡터스토어 top-3 검색 → 품질 미달 시 Gemini 폴백, 사용자 언어로 응답
- **콘텐츠 자동 번역 엔진** — 영어 원문을 LLM으로 일괄(batch) 번역, (텍스트·언어) 단위 24h 캐싱 → 주소·가격·전화번호·고유명사는 보존
- **이력서 분석 / 모의면접** — 구조화된 JSON 출력 프롬프트
- **관리자 패널** — PDF 업로드 → 청크 분할 → 임베딩 → `documents` 적재로 지식베이스 구축

## 4. 기대 효과 및 한계 (Impact & Limitations)

**기대 효과**
- 흩어진 공공 정보를 9개 언어로 통합 → 정보 탐색 비용·언어 장벽 감소
- 유학 → 취업 → 정착의 **전체 여정**을 한 흐름으로 제공
- RAG 기반으로 **출처 있는 정확한 답변** 제공 (할루시네이션 완화)

**한계 및 개선 방향**
- 데이터의 최신성은 공식 사이트 갱신에 의존 → 정기 동기화 파이프라인 필요
- 이메일 발송은 구독 정보 저장까지 구현(SendGrid/Resend 연동은 향후 과제)
- 비로그인 비밀번호 검증 미구현 → bcrypt 해시 인증 추가 필요

---

## ⚙️ 로컬 실행 (Local Run)

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # 키 입력
streamlit run app.py
```

### Secrets (`.streamlit/secrets.toml`)
```toml
GEMINI_API_KEY = "..."
SUPABASE_URL = "https://xxxx.supabase.co"
SUPABASE_KEY = "..."
SUPABASE_DB_CONNECTION = "postgresql://...:6543/postgres"
ADMIN_PASSWORD = "..."
```

## ☁️ Streamlit Community Cloud 배포

1. 이 저장소를 GitHub에 푸시 (⚠️ `secrets.toml`은 `.gitignore`로 제외됨)
2. https://share.streamlit.io → **New app** → 저장소·브랜치·`app.py` 선택
3. **Advanced settings → Secrets**에 위 5개 키 입력
4. **Deploy** → 발급된 `*.streamlit.app` 주소를 제출

## 🛠 기술 스택

`Streamlit` · `LlamaIndex` · `Google Gemini 2.0 Flash` · `text-embedding-004`
`Supabase (PostgreSQL + pgvector)` · `Plotly` · `pypdf` · `pandas`
