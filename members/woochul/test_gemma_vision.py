"""
Gemma 3 / Gemma 4 이미지 분석 비교 테스트
가장 가벼운 모델(1b) 제외하고 각 버전별 테스트
사용법: python test_gemma_vision.py <이미지_경로>
"""

import sys
import time
from pathlib import Path

try:
    import ollama
except ImportError:
    print("ollama 패키지가 없습니다. 설치 중...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ollama"])
    import ollama


# 테스트 모델 목록 (1b 제외)
MODELS = {
    "Gemma 3": ["gemma3:4b", "gemma3:12b", "gemma3:27b"],
    "Gemma 4": ["gemma4:4b", "gemma4:12b", "gemma4:27b"],
}

TEST_PROMPT = "이 이미지를 자세히 분석하고 설명해줘. 어떤 내용이 담겨 있는지 한국어로 답해줘."


def get_installed_models() -> list[str]:
    try:
        return [m.model for m in ollama.list().models]
    except Exception:
        return []


def pull_model(model: str):
    print(f"  다운로드 중: {model} ...")
    try:
        for progress in ollama.pull(model, stream=True):
            status = getattr(progress, "status", "")
            if "pulling" in status or "verifying" in status:
                print(f"  {status}", end="\r")
        print(f"  완료: {model}          ")
    except Exception as e:
        print(f"  다운로드 실패: {e}")
        return False
    return True


def test_model(model: str, image_path: str) -> dict:
    result = {"model": model, "success": False, "response": "", "elapsed": 0.0, "error": ""}
    try:
        start = time.time()
        response = ollama.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": TEST_PROMPT,
                "images": [image_path],
            }],
        )
        result["elapsed"] = round(time.time() - start, 2)
        result["success"] = True
        result["response"] = response.message.content
    except Exception as e:
        result["error"] = str(e)
    return result


def run_tests(image_path: str):
    print(f"\n{'=' * 60}")
    print("Gemma 이미지 분석 비교 테스트")
    print(f"이미지: {image_path}")
    print(f"{'=' * 60}")

    installed = get_installed_models()
    all_results = []

    for family, models in MODELS.items():
        print(f"\n▶ {family}")
        print("-" * 50)

        for model in models:
            print(f"\n[ {model} ]")

            if model not in installed:
                print(f"  미설치 모델 — 다운로드를 시작합니다.")
                ok = pull_model(model)
                if not ok:
                    all_results.append({
                        "family": family, "model": model,
                        "success": False, "elapsed": 0.0,
                        "response": "", "error": "다운로드 실패"
                    })
                    continue

            print("  분석 중...")
            result = test_model(model, image_path)
            all_results.append({"family": family, **result})

            if result["success"]:
                print(f"  소요 시간: {result['elapsed']}초")
                print(f"  응답 미리보기:\n")
                # 응답 500자까지 출력
                preview = result["response"][:500]
                for line in preview.splitlines():
                    print(f"    {line}")
                if len(result["response"]) > 500:
                    print("    ...")
            else:
                print(f"  오류: {result['error']}")

    # 결과 요약 테이블
    print(f"\n{'=' * 60}")
    print("최종 요약")
    print(f"{'=' * 60}")
    print(f"{'모델':<22} {'상태':<10} {'소요시간'}")
    print("-" * 45)
    for r in all_results:
        if r["success"]:
            status = "✓ 성공"
            elapsed = f"{r['elapsed']}초"
        else:
            status = "✗ 실패"
            elapsed = "-"
        print(f"{r['model']:<22} {status:<10} {elapsed}")

    # 성공한 모델만 응답 전체 출력
    successful = [r for r in all_results if r["success"]]
    if successful:
        print(f"\n{'=' * 60}")
        print("전체 응답 내용")
        print(f"{'=' * 60}")
        for r in successful:
            print(f"\n[ {r['model']} ] ({r['elapsed']}초)")
            print("-" * 50)
            print(r["response"])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python test_gemma_vision.py <이미지_경로>")
        print("예시:  python test_gemma_vision.py ./test.jpg")
        sys.exit(1)

    img = sys.argv[1]
    if not Path(img).exists():
        print(f"오류: 이미지 파일을 찾을 수 없습니다 → {img}")
        sys.exit(1)

    run_tests(img)
