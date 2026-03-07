# CI/CD 개요

이 문서는 CuKKS에서 변경 검증, 패키지 배포, 릴리즈 생성이 어떻게 연결되는지 설명합니다.

## 워크플로우 구조

### 1. CI 검증: `.github/workflows/build-wheels.yml`

트리거:

- pull request
- `main` 브랜치 push
- 수동 `workflow_dispatch`

수행 내용:

- CUDA 11.8, 12.1, 12.4, 12.8용 Docker 이미지를 빌드
- 각 CUDA 패키지용 GPU wheel 빌드
- Python 3.10-3.13에서 wheel 구조 검증
- 메인 `cukks` 패키지 빌드

하지 않는 일:

- PyPI publish는 하지 않음
- GitHub release를 만들지 않음

### 2. 패키지 배포: `.github/workflows/publish-packages.yml`

트리거:

- release workflow에서 `workflow_call`
- 복구 목적의 수동 `workflow_dispatch`

안전 장치:

- 요청된 릴리즈 버전과 패키지 버전이 일치하는지 확인
- 대상 버전이 이미 PyPI에 존재하는지 확인
- GitHub OIDC Trusted Publishing과 `environment: pypi` 사용

배포 대상 패키지:

- `cukks`
- `cukks-cu118`
- `cukks-cu121`
- `cukks-cu124`
- `cukks-cu128`

### 3. 마일스톤 릴리즈 오케스트레이션: `.github/workflows/release-milestone.yml`

트리거:

- 수동 `workflow_dispatch`

입력값:

- `milestone_name`
- `target_ref`
- `draft`
- `publish_pypi`

수행 내용:

- milestone이 존재하고 닫혀 있는지 확인
- 해당 milestone의 PR이 모두 `main`에 머지됐는지 확인
- milestone 제목을 릴리즈 버전으로 사용하고, 패키지 버전을 그 값에 맞게 동기화
- milestone 안의 merged PR / closed issue 기준으로 릴리즈 노트 생성
- annotated tag 생성
- GitHub release 생성
- `publish-packages.yml` 호출

## 외부 기여자 관점

외부 기여자는 보통 PR 단계에서 CI와 상호작용합니다.

- 먼저 이슈를 열고
- 이슈에 연결된 PR을 만들고
- PR의 `Build Wheels` 상태를 확인하면 됩니다
- 마일스톤 지정과 릴리즈 시점 결정은 메인테이너가 담당합니다

## 메인테이너 관점

릴리즈 준비는 메인테이너가 담당합니다.

릴리즈 전 확인:

- milestone이 닫혀 있는지
- milestone의 PR이 모두 머지됐는지
- 모든 패키지 버전이 일관되게 올라갔는지
- PyPI Trusted Publishing 설정이 아래와 일치하는지
  - owner: `devUuung`
  - repo: `CuKKS`
  - workflow: `.github/workflows/publish-packages.yml`
  - environment: `pypi`

## 운영 원칙

- GitHub UI에서 일반 release를 수동 생성하는 것은 publish 경로가 아닙니다
- historical release 복구는 publish 수단으로 사용하지 않습니다
- release publication은 PR/`main` CI와 분리되어 있습니다
