# GAR (GPT-based Answer Refinement)

금융 도메인 질의응답을 위한 RAG 기반 답변 생성 및 개선 파이프라인입니다.

## 프로젝트 구조

GAR/
├── task1/ # Hybrid Search 구현
│ └── run_hybrid_search.py
├── task2/ # GPT 기반 답변 생성 파이프라인
│ ├── data/
│ │ ├── raw/ # 원본 데이터
│ │ ├── processed/ # 중간 처리 결과
│ │ └── models/ # 파인튜닝된 모델
│ ├── Selection_agent.py # 관련 문서 선택
│ ├── DPO_agent.py # 초기 답변 생성
│ ├── DPO_agent_finetuning.py # 모델 파인튜닝
│ ├── Finetuned_DPO_agent.py # 파인튜닝 모델 적용
│ ├── G_Eval_agent.py # 답변 평가 및 개선
│ └── main.py # 전체 파이프라인 실행
└── requirements.txt # 필요 패키지 목록


## 파이프라인 설명

1. **Selection Agent**: 주어진 query에 대해 10개의 문서 중 관련성 높은 문서들을 선택
2. **DPO Agent**: 선택된 문서들을 바탕으로 두 개의 답변을 생성
3. **DPO Finetuning**: 생성된 답변들을 바탕으로 모델 파인튜닝 수행
4. **Finetuned DPO Agent**: 파인튜닝된 모델을 사용하여 전체 데이터셋에 대한 답변 생성
5. **G-Eval Agent**: 생성된 답변의 품질을 평가하고 필요시 개선

## 설치 방법

1. 저장소 클론
git clone https://github.com/seohyunwoo-0407/GAR.git
cd GAR


2. 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate # Linux/Mac
.venv\Scripts\activate # Windows

3. 필요 패키지 설치
pip install -r requirements.txt


## 주요 파라미터

- `top_probs`: G-Eval에서 사용할 상위 확률 개수 (기본값: 10)
- `model_name`: 사용할 GPT 모델 이름 (기본값: "gpt-4o-mini")
- `temperature`: 답변 생성시 사용할 temperature 값

## 참고사항

- 모든 중간 결과는 `task2/data/processed/` 디렉토리에 저장됩니다.
- 파인튜닝된 모델은 `task2/data/models/fine_tuned/` 디렉토리에 저장됩니다.
- 실행 중 오류 발생시 중간 저장된 결과부터 다시 시작할 수 있습니다.
