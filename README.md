# Auto-Leverage

S&P 500의 변동성과 VIX 프리미엄을 이용해 레버리지 비중을 예측하는 프로젝트입니다.

## 모델 구조

- GARCH(1,1)로 S&P 500의 조건부 분산을 추정합니다.
- VIX의 일간 분산과 GARCH 분산의 차이를 variance risk premium으로 계산합니다.
- GARCH 분산, premium, VIX, VIX 5일 변화율을 입력으로 사용합니다.
- Ridge 회귀로 목표 레버리지를 직접 예측합니다.
- 예측값에 일일 변화 제한과 EMA를 적용한 뒤 백테스트 수익률을 계산합니다.

학습된 모델은 매일 다시 학습하지 않고 `garch_regression_model.pkl`을 사용합니다.

## 주요 설정

- 목표 변동성: 연 18%
- 최소 레버리지: 0.5배
- 최대 레버리지: 1.0배
- EMA 기간: 7거래일
- 일일 레버리지 변화 제한: ±10%

설정은 `garch_train_model.py`와 학습된 모델 artifact에 저장됩니다.

## 파일 구성

| 파일 | 역할 |
| --- | --- |
| `garch_train_model.py` | 직접 레버리지 Ridge 모델 학습 및 저장 |
| `garch_model.py` | GARCH 계산, feature/target 생성, 레버리지 적용 함수 |
| `garch_auto_leverage.py` | 저장된 모델 로드, 백테스트 및 현재 레버리지 계산 |
| `garch_regression_model.pkl` | 학습된 Ridge 및 GARCH 파라미터 artifact |
| `visualize_garch.py` | SPY, SSO, GARCH AUTO 비교 차트 생성 |
| `index.html` | 현재 레버리지와 백테스트 성과 웹 페이지 |
| `leverage.json` | 최신 레버리지와 백테스트 성과 |
| `leverage-history.json` | 최대 5년치 레버리지 히스토리 |

## 실행

모델 학습:

```powershell
uv run python garch_train_model.py
```

저장된 모델로 백테스트 및 최신 레버리지 계산:

```powershell
uv run python garch_auto_leverage.py
```

시각화 생성:

```powershell
uv run python visualize_garch.py
```

## 생성 데이터

실행 스크립트는 다음 파일을 생성합니다.

- `garch_auto_leverage_backtest.csv`
- `garch_auto_leverage.png`
- `garch_auto_leverage_performance.png`
- `leverage.json`
- `leverage-history.json`

시각화 스크립트는 `visual-sp500-garch-auto.png`를 생성합니다.

## GitHub Pages 배포

GitHub Actions는 저장된 모델을 불러와 백테스트를 실행하고, 시각화와 JSON을 생성한 뒤 개인 웹사이트 저장소에 배포합니다.

매일 실행되므로 일반 실행 경로에는 모델 재학습을 포함하지 않습니다.
