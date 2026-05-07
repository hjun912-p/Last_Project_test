# 팀 Git 협업 규칙

> 충돌 없이 함께 작업하기 위한 최소한의 약속입니다.

---

## 하루 고정 규칙 (2가지만)

```
09:00   작업 시작 전  →  git pull   (필수)
17:50   퇴근 전       →  git push   (필수)
```

---

## 기능 추가/수정 완료 시 흐름

```
1. git push
2. 디스코드 공지
   예) "app.py - Stage1 EXIF 분석 UI 추가했어요"
3. 팀원들 확인 후 git pull
4. 공지에 ✅ 이모지로 수신 확인
```

✅ 수를 보면 누가 최신 코드를 받았는지 한눈에 파악됩니다.

---

## 필수 규칙

### 1. 작업 시작 전 항상 pull 먼저
```bash
git pull origin main
```
> pull 없이 push 하면 reject 됩니다. 반드시 pull → 작업 → push 순서로.

### 2. 같은 파일 동시에 건드리지 않기

작업 시작 전 디스코드에 **"나 오늘 app.py 작업할게"** 한 마디면 충돌의 90%를 막을 수 있습니다.

---

## 커밋 메시지 작성법

```bash
# 좋은 예
git commit -m "feat: Stage1 EXIF 분석 결과 UI 추가"
git commit -m "fix: Instagram URL 파싱 오류 수정"
git commit -m "test: 이미지 10장 배치 테스트"

# 나쁜 예
git commit -m "수정"
git commit -m "ㅇㅇ"
git commit -m "asdf"
```

| 접두어 | 언제 쓰나 |
|--------|---------|
| `feat:` | 새 기능 추가 |
| `fix:` | 오류 수정 |
| `test:` | 테스트 코드/실험 |
| `refactor:` | 코드 구조 정리 |
| `docs:` | 문서 수정 |

---

## 충돌(Conflict) 발생 시

```bash
git pull origin main
# CONFLICT 메시지 나오면

# 1. 충돌 파일 확인
git status

# 2. 해당 파일 열어서 직접 수정
# <<<<<<< HEAD     ← 내 코드
# =======
# >>>>>>> origin   ← 상대방 코드
# 둘 중 맞는 것만 남기고 표시 줄 삭제

# 3. 저장 후 다시 커밋
git add .
git commit -m "fix: 충돌 해결"
git push origin main
```

> 혼자 해결하기 어려우면 건드리지 말고 팀원에게 먼저 연락하세요.

---

## 절대 하지 말 것

| 행동 | 이유 |
|------|------|
| `git push --force` | 남의 커밋을 덮어써서 작업 내용 사라짐 |
| `.env` 파일 커밋 | API 키가 깃허브에 공개됨 |
| pull 없이 바로 push | 항상 reject, 충돌 원인 1위 |
| 퇴근 전 push 안 함 | 다음날 팀원이 구버전으로 작업하게 됨 |

---

## 한 줄 요약

> **시작할 때 pull, 기능 완료하면 push + 디스코드 공지, 퇴근 전 push**
