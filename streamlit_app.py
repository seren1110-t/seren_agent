import os
import streamlit as st
from transformers import pipeline
import glob
import asyncio
import torch

# 환경 변수 설정
os.environ["STREAMLIT_SERVER_ENABLE_STATIC_SERVE"] = "false"
os.environ["HF_HOME"] = "/tmp/hf_home"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_home/models"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/tmp/hf_home/hub"
os.environ["STREAMLIT_CONFIG_DIR"] = "/tmp/.streamlit"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
os.environ["HOME"] = "/tmp"  # 중요: HOME 환경변수도 설정

# 디렉토리 생성 및 권한 설정
cache_dirs = [
    "/tmp/hf_home",
    "/tmp/hf_home/models",
    "/tmp/hf_home/hub",
    "/tmp/.streamlit"
]

def fix_permissions(path):
    for root, dirs, files in os.walk(path):
        for d in dirs:
            os.chmod(os.path.join(root, d), 0o777)
        for f in files:
            os.chmod(os.path.join(root, f), 0o777)

for dir_path in cache_dirs:
    os.makedirs(dir_path, exist_ok=True)
    os.chmod(dir_path, 0o777)
    fix_permissions(dir_path)

# 락 파일 삭제
lock_files = glob.glob("/tmp/hf_home/**/*.lock", recursive=True)
for lock_file in lock_files:
    try:
        os.remove(lock_file)
    except Exception:
        pass

# PyTorch JIT 설정 (문제 있다면 주석 처리)
try:
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
except Exception:
    pass

def ensure_event_loop():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

@st.cache_resource
def load_model():
    ensure_event_loop()
    return pipeline(
        'text-generation',
        model='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        device_map="auto",
        torch_dtype="auto",
        cache_dir="/tmp/hf_home/models"
    )

def main():
    st.title("🚀 TinyLlama 테스트")
    prompt = st.text_input("프롬프트 입력:")
    
    if st.button("실행"):
        try:
            pipe = load_model()
            output = pipe(prompt, max_new_tokens=200)[0]['generated_text']
            st.write(output)
        except Exception as e:
            st.error(f"오류 발생: {str(e)}")
            st.info("""
            **추가 조치 방법**
            1. 페이지 새로고침
            2. 터미널에서 다음 명령 실행:
               ```
               rm -rf /tmp/hf_home/**/*.lock
               ```
            3. 모델 캐시 수동 삭제:
               ```
               rm -rf /tmp/hf_home/models
               ```
            """)

if __name__ == "__main__":
    main()
