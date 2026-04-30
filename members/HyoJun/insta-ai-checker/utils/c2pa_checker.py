import c2pa
import json
import os

def check_c2pa(file_path):
    if not os.path.exists(file_path):
        return False, "File not found"
        
    try:
        # Check if the file is a video, c2pa-python might have limited support for some video formats
        # but works well with images and many common video containers.
        reader = c2pa.Reader(file_path)
        manifest = reader.json()
        
        if manifest:
            manifest_data = json.loads(manifest)
            active_manifest_label = manifest_data.get("active_manifest")
            
            if not active_manifest_label:
                return False, "C2PA manifest found, but no active manifest data detected."
            
            manifests = manifest_data.get("manifests", {})
            active_manifest = manifests.get(active_manifest_label, {})
            assertions = active_manifest.get("assertions", [])
            
            ai_info = []
            for assertion in assertions:
                label = assertion.get("label")
                # Look for creative work metadata which often contains generator info
                if "stds.schema-org.CreativeWork" in label:
                    data = assertion.get("data", {})
                    if "generator" in data:
                        ai_info.append(f"Generator: {data['generator']}")
                
                # Look for AI-specific assertions if available
                if "com.adobe.generative-ai" in label:
                    ai_info.append("Adobe Generative AI assertion found.")
            
            return True, f"C2PA manifest detected! {', '.join(ai_info) if ai_info else 'Provenance info found, but no explicit AI generator listed.'}"
        else:
            return False, "No C2PA manifest detected in this file."
            
    except Exception as e:
        error_msg = str(e)
        if "No manifest found" in error_msg or "ManifestNotFound" in error_msg:
            return False, "C2PA 메타데이터가 발견되지 않았습니다. (일반 미디어 또는 메타데이터가 제거된 AI 이미지일 수 있습니다.)"
        return False, f"분석 중 참고사항: {error_msg}"
