import google.generativeai as genai
import os
from PIL import Image
from dotenv import load_dotenv

# Load API Key from .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    genai.configure(api_key=api_key)

def analyze_with_gemini(file_path, media_type):
    if not api_key:
        return "Gemini API Key not found. Please set GEMINI_API_KEY in .env file."
        
    try:
        # Use the model that was confirmed to exist in your ListModels output
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        if media_type == "image":
            img = Image.open(file_path)
            prompt = """
            이 이미지를 정밀 분석하여 생성형 AI로 만들어졌을 가능성이 있는지 판단해줘.
            특히 다음 항목에 주목해줘:
            1. 신체 구조 오류 (손가락, 치아, 눈 등)
            2. 배경의 왜곡이나 부자연스러운 연결
            3. AI 모델 특유의 질감이나 광택
            4. 조명과 그림자의 불일치
            
            상세 보고서를 한국어로 작성하고 마지막에 '결론: AI 생성 가능성 [높음/낮음/판단불가]'를 포함해줘.
            """
            response = model.generate_content([prompt, img])
            return response.text
            
        elif media_type == "video":
            prompt = """
            이 비디오를 분석하여 AI가 생성한 영상(Sora, Kling 등)인지 실제 촬영 영상인지 판단해줘.
            1. 일관성 (물체가 변하거나 사라지는지)
            2. 물리 법칙 위반
            3. 고속 움직임에서의 노이즈나 어색함
            
            상세 보고서를 한국어로 작성하고 마지막에 '결론: AI 생성 가능성 [높음/낮음/판단불가]'를 포함해줘.
            """
            video_file = genai.upload_file(path=file_path)
            response = model.generate_content([prompt, video_file])
            return response.text
        else:
            return "Gemini 분석을 지원하지 않는 미디어 유형입니다."
            
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg:
            return "현재 Gemini API 사용 한도가 초과되었습니다. (Google AI Studio에서 할당량을 확인하거나 잠시 후 다시 시도해 주세요.)"
        return f"Gemini 분석 오류: {error_msg}"
