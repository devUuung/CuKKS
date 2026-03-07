# CuKKS 기여 가이드

CuKKS에 기여해 주셔서 감사합니다.

이 프로젝트는 issue-first 흐름으로 운영됩니다.

1. 이슈를 생성하거나 기존 이슈 범위를 정리합니다
2. 메인테이너가 이슈를 분류하고 라벨/마일스톤을 정합니다
3. 기여자는 이슈에 연결된 PR을 엽니다
4. CI가 PR을 검증합니다
5. 메인테이너가 리뷰하고 머지한 뒤, 해당 변경을 마일스톤 릴리즈에 포함합니다

## 시작하기 전에

- 설치와 프로젝트 개요는 `README.ko.md`를 먼저 읽어주세요
- 새 이슈를 만들기 전에 기존 이슈를 확인해 주세요
- `.github/ISSUE_TEMPLATE/`의 템플릿을 사용해 주세요
- 큰 기능 변경은 바로 PR을 보내기보다 먼저 이슈에서 범위를 맞추는 것을 권장합니다

## 기여 흐름

### 1. 이슈 만들기

- 버그/회귀는 `Bug Report` 템플릿 사용
- 기능 제안/개선은 `Feature Request` 템플릿 사용
- 가능하면 최소 재현 코드나 구체적인 제안 내용을 포함해 주세요

마일스톤 지정과 triage는 메인테이너가 담당합니다. 외부 기여자는 마일스톤을 직접 만들 필요가 없습니다.

### 2. 브랜치에서 작업하기

```bash
git clone https://github.com/devUuung/CuKKS.git
cd CuKKS
git checkout -b your-branch-name
```

기존 코드 스타일과 패턴을 맞추고, 변경 범위는 가능한 한 작고 명확하게 유지해 주세요.

### 3. PR 열기 전에 로컬 확인

일반적으로 아래 정도는 먼저 확인해 주세요.

```bash
pytest -q
python -m build
```

GPU 패키징이나 릴리즈 로직을 건드렸다면 PR 본문에 무엇을 검증했는지 적어 주세요.

### 4. Pull Request 열기

- `.github/pull_request_template.md` 템플릿을 사용해 주세요
- 변경 요약을 명확히 적어 주세요
- 테스트/검증 결과를 구체적으로 적어 주세요
- 가능하면 `Closes #123` 형식으로 관련 이슈를 연결해 주세요
- 메인테이너가 PR에 마일스톤을 지정했다면 그 릴리즈 범위에 맞게 유지해 주세요

### 5. 리뷰와 머지

- 메인테이너가 코드 품질, 릴리즈 적합성, 마일스톤 범위를 함께 검토합니다
- 닫힌 마일스톤에 속한 머지된 PR은 milestone 기반 릴리즈에 포함됩니다

## 기여자 관점의 CI

PR과 `main` 브랜치 push는 `.github/workflows/build-wheels.yml`을 실행합니다.

이 워크플로우는:

- wheel 빌드에 필요한 CUDA Docker 이미지를 만듭니다
- CUDA 11.8, 12.1, 12.4, 12.8용 GPU wheel artifact를 빌드합니다
- Python 3.10-3.13에서 wheel 구조를 검증합니다
- 메인 `cukks` 패키지를 빌드합니다

일반 PR에서는 검증만 수행하며, 패키지 publish는 하지 않습니다.

자동화 구조 전체는 `docs/ci-cd.ko.md`를 참고해 주세요.

## 릴리즈 모델

CuKKS는 milestone 기반으로 릴리즈합니다.

- 이슈와 PR을 하나의 milestone 아래 묶고
- 메인테이너가 milestone을 닫은 뒤
- 수동 milestone release workflow를 실행합니다

메인테이너용 릴리즈 운영 방법은 `RELEASING.md`를 참고해 주세요.
