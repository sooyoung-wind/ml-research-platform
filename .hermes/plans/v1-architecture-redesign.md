# ML Research Platform v1.0 — 아키텍처 재설계

## 현재(v0.2.0) → 목표(v1.0) 갭 분석

```
┌─────────────────────────────────────────────────────────────────────┐
│                     사용자가 원하는 워크플로우                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [사용자 질문]                                                      │
│       "RAG 시스템에서 hallucination을 줄이는 최신 연구 알려줘"      │
│       ↓                                                             │
│  [1] 질문 명확성 판단 (LLM)                                        │
│       ↓ 애매하면                                                    │
│  [2] 인터랙티브 인터뷰 (3+ 라운드)                                  │
│       → 검색 전략 수립 (키워드, 소스, 범위)                         │
│       ↓                                                             │
│  [3] 다경로 논문 수집                                               │
│       → arXiv, Semantic Scholar, HuggingFace Papers,                │
│         Google Scholar, PapersWithCode, GitHub                      │
│       ↓                                                             │
│  [4] 지식 그래프 + Wiki 구축                                        │
│       → 제목/초록/참고문헌 기반                                     │
│       → 자동 관계 매핑 (방법론, 데이터셋, 태스크)                  │
│       ↓                                                             │
│  [5] 리서치 보고서 생성 (HTML 대시보드)                             │
│       → 수집 결과 + KG 시각화 + 트렌드 분석                        │
│       → 사용자가 브라우저에서 확인                                  │
│       ↓                                                             │
│  [6] 논문 선택 (인터랙티브)                                         │
│       → 사용자가 관심 논문 체크/선택                                │
│       ↓                                                             │
│  [7] DeepCode 논문 구현                                             │
│       → 선택된 논문 → PDF → 코드 자동 생성                         │
│       ↓                                                             │
│  [8] 재현성 분석 리포트                                             │
│       → 무엇이 구현됐는지 / 안 됐는지                               │
│       → 사용자가 확인                                               │
│       ↓                                                             │
│  [9] Future Work 수립                                               │
│       → 구현 결과 기반 개선사항 + 후속 연구 제안                   │
│       → 사용자 승인 후 Notion/GitHub에 저장                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 기존 모듈 재활용 매핑

| 단계 | 신규/기존 | 모듈 | 상태 |
|------|-----------|------|------|
| [1] 질문 판단 | **신규** | `agent/question_clarifier.py` | 미구현 |
| [2] 인터뷰 | 기존 개선 | `analysis/interview.py` | 기존 (Trend용, 범용화 필요) |
| [3] 다경로 수집 | 기존 확장 | `discovery/pipeline.py` + 4개 클라이언트 | 기존 (통합 필요) |
| [4] KG + Wiki | 기존 | `graph/` + `wiki/` | 구현됨 |
| [5] 보고서 | **신규** | `agent/report_dashboard.py` | reporter.py는 Notion만, HTML 대시보드 없음 |
| [6] 논문 선택 | **신규** | `agent/paper_selector.py` | 미구현 |
| [7] DeepCode | 기존 | `codegen/deepcode_runner.py` | 구현됨 |
| [8] 재현성 분석 | **신규** | `agent/reproducibility_analyzer.py` | 미구현 |
| [9] Future Work | **신규** | `agent/future_work_planner.py` | 미구현 |
| 오케스트레이터 | 기존 개선 | `orchestration/orchestrator.py` | batch 기반, interactive 전환 필요 |

## v1.0 구현 계획 (9개 Phase)

### Phase A: Agent Core (질문 → 인터뷰 → 검색전략)
- `agent/` 패키지 생성
- `QuestionClarifier`: LLM이 질문 명확성 판단 (1~5점)
- `ResearchInterviewer`: interview.py 범용화, 최대 5라운드
- `SearchStrategy`: 검색 키워드/소스/범위 자동 생성
- CLI: `ml-research ask "질문"` — 인터랙티브 세션 시작

### Phase B: Multi-Source Discovery Pipeline
- `discovery/unified_search.py`: 4개 클라이언트 통합 병렬 검색
- arXiv + Semantic Scholar + HuggingFace + PapersWithCode
- 중복 제거 + 랭킹 + relevance scoring
- 검색 전략(SearchStrategy) 기반 동적 쿼리 생성

### Phase C: Knowledge Graph + Wiki 자동 구축
- 수집 결과 → 자동 KG 빌드 (제목/초록/참고문헌)
- Wiki 임포트 자동화
- 기존 `graph/builder.py` 활용

### Phase D: Research Dashboard (HTML 보고서)
- `agent/report_dashboard.py`: Jinja2 + Tailwind HTML
- 종합 대시보드: 수집 통계 + KG 시각화 + 트렌드 + 논문 목록
- 논문 선택 인터랙티브 UI (체크박스)

### Phase E: Paper Selection + DeepCode 실행
- 대시보드에서 선택된 논문 → DeepCode 실행
- 진행 상태 실시간 표시
- 기존 `codegen/deepcode_runner.py` 활용

### Phase F: Reproducibility Analysis
- `agent/reproducibility_analyzer.py`
- 생성된 코드 vs 논문 알고리즘 비교
- 구현률 (%) + 미구현 항목 + 에러 로그 분석
- HTML 리포트 생성

### Phase G: Future Work Planner
- `agent/future_work_planner.py`
- 구현 결과 + 논문의 future work 섹션 기반
- 개선사항 + 후속 연구 방향 제안
- Notion + GitHub Issues 동기화

### Phase H: Interactive Session Manager
- `agent/session.py`: 전체 워크플로우 상태 관리
- 세션 저장/복구 (interrupt/resume)
- LangGraph 기반 상태 머신

### Phase I: CLI 통합 + Web UI
- `ml-research research "질문"` — 전체 인터랙티브 플로우
- `ml-research session list/resume` — 세션 관리
- `ml-research dashboard` — 대시보드 서버 (FastAPI)
