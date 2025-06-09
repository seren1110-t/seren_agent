import streamlit as st
from transformers import pipeline
import torch

st.title("🤖 작은 LLM 모델 로드 확인")

@st.cache_resource
def load_model():
    """작은 LLM 모델 로드"""
    try:
        # DistilGPT-2 (작은 모델, 약 82MB)
        model = pipeline(
            "text-generation", 
            model="distilgpt2",
            device=0 if torch.cuda.is_available() else -1
        )
        return model, "✅ 모델 로드 성공!"
    except Exception as e:
        return None, f"❌ 모델 로드 실패: {str(e)}"

# 모델 로드
with st.spinner("모델 로딩 중..."):
    model, status = load_model()

st.write(status)

if model:
    # 간단한 텍스트 생성 테스트
    st.subheader("텍스트 생성 테스트")
    
    prompt = st.text_input("프롬프트 입력:", "Hello, I am")
    
    if st.button("생성"):
        if prompt:
            with st.spinner("생성 중..."):
                result = model(prompt, max_length=50, num_return_sequences=1)
                st.write("**생성 결과:**")
                st.write(result[0]['generated_text'])
        else:
            st.warning("프롬프트를 입력해주세요.")

# 시스템 정보
st.sidebar.write("**시스템 정보:**")
st.sidebar.write(f"CUDA 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    st.sidebar.write(f"GPU: {torch.cuda.get_device_name(0)}")
st.sidebar.write(f"PyTorch 버전: {torch.__version__}")
