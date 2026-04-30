# -*- coding: utf-8 -*-
import gradio as gr
import os
import shutil
from utils.downloader import download_instagram_media
from utils.c2pa_checker import check_c2pa
from utils.synthid_checker import check_synthid
from utils.gemini_analyzer import analyze_with_gemini

def process_url(url):
    if not url:
        return None, "인스타그램 URL을 입력해주세요.", "", "", "", None
    
    target_dir = "temp"
    
    # Step 0: Download Media
    file_path, media_type_or_error = download_instagram_media(url, target_dir)
    if not file_path:
        return None, f"에러: {media_type_or_error}", "", "", "", None
    
    media_type = media_type_or_error
    
    # Step 1: C2PA Detection
    c2pa_found, c2pa_result = check_c2pa(file_path)
    
    # Step 2: Google SynthID Detection (Now returns viz_path)
    synth_found, synth_result, synth_viz = check_synthid(file_path, media_type)
    
    # Step 3: Gemini Analysis
    gemini_result = analyze_with_gemini(file_path, media_type)
    
    # Final Summary
    summary = "### [결과 요약]\n"
    if c2pa_found:
        summary += "✅ **C2PA 탐지 성공:** 생성형 AI 메타데이터가 발견되었습니다.\n"
    elif synth_found:
        summary += "✅ **SynthID 탐지 성공:** Google의 AI 워터마크가 발견되었습니다.\n"
    else:
        summary += "🔍 **메타데이터/워터마크 미감지:** Gemini AI의 시각적 분석 결과를 확인하세요.\n"
        
    return file_path, summary, c2pa_result, synth_result, gemini_result, synth_viz

# Gradio Interface
with gr.Blocks(title="AI Media Detector for Instagram") as demo:
    gr.Markdown("# 📸 AI 생성 미디어 판별 서비스 (Instagram)")
    gr.Markdown("인스타그램 URL을 입력하면 3단계(C2PA → Google SynthID → Gemini) 분석을 통해 AI 생성 여부를 판별합니다.")
    
    with gr.Row():
        with gr.Column():
            url_input = gr.Textbox(label="Instagram URL (Post/Reel)", placeholder="https://www.instagram.com/p/...")
            submit_btn = gr.Button("분석 시작", variant="primary")
            
        with gr.Column():
            output_media = gr.File(label="다운로드된 미디어")
            
    gr.Markdown("---")
    
    with gr.Tabs():
        with gr.TabItem("종합 요약"):
            summary_output = gr.Markdown()
        with gr.TabItem("1단계: C2PA 결과"):
            c2pa_output = gr.Textbox(label="C2PA Metadata")
        with gr.TabItem("2단계: SynthID 결과"):
            with gr.Row():
                with gr.Column():
                    synth_output = gr.Markdown()
                with gr.Column():
                    synth_viz_output = gr.Image(label="스펙트럼 분석 시각화")
        with gr.TabItem("3단계: Gemini AI 상세 분석"):
            gemini_output = gr.Markdown()

    submit_btn.click(
        fn=process_url,
        inputs=url_input,
        outputs=[output_media, summary_output, c2pa_output, synth_output, gemini_output, synth_viz_output]
    )

if __name__ == "__main__":
    if not os.path.exists("temp"):
        os.makedirs("temp")
    demo.launch()
