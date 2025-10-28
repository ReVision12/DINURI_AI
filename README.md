# DiNuri AI — FastAPI 백엔드

**DiNuri AI**는 시니어를 위한 문서 해설 서비스입니다.  
이미지 문서 → 텍스트 추출 → 요약 → 음성 변환을 한 번에 수행합니다.

---

## 주요 기능

| Endpoint | 설명 |
|-----------|------|
| `POST /api/ocr-summarize` | 이미지 → OCR + 요약 + 태그 |
| `POST /api/ocr-tags` | 이미지 → 태그만 추출 |
| `POST /api/tts-summary` | 요약 JSON → MP3 음성 |
| `POST /api/tts` | 텍스트 → MP3 음성 |
| `GET /api/health` | 서버 상태 확인 |

---

## 환경 설정

`.env.example`을 복사해 `.env`로 변경한 뒤,  
다음 항목을 본인 환경에 맞게 수정하세요.

```bash
GOOGLE_APPLICATION_CREDENTIALS=/Users/yourname/Desktop/dinuri-ai/service-account.json
OPENAI_API_KEY=sk-proj-당신의_키
```

---

## 실행 방법 

```bash
# 1. 가상환경 생성
python3 -m venv venv

# 2. 가상환경 활성화
source venv/bin/activate   # macOS / Linux
# venv\Scripts\activate    # Windows

# 3. 패키지 설치
pip install -r requirements.txt

# 4. 서버 실행
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

---

## 테스트
- 서버 상태 확인: http://localhost:8000/api/health

- Swagger 문서: http://localhost:8000/docs

---

## 주요 파일 구조
```bash
DINURI_AI/
├── app.py              # FastAPI 서버 엔트리포인트
├── ai_doc_helper2.py   # OCR + 요약 + TTS 핵심 로직
├── requirements.txt    # 패키지 목록
├── .env.example        # 환경변수 예시 파일
└── README.md
```
