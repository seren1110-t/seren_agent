import os
import sys

# 환경 변수 설정 (토크나이저 경고 방지)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# PyTorch 클래스 경로 충돌 해결 (최상단에 위치)
try:
    import torch
    import importlib.util
    torch_classes_path = os.path.join(os.path.dirname(importlib.util.find_spec("torch").origin), "classes")
    if hasattr(torch, "classes"):
        torch.classes.__path__ = [torch_classes_path]
except Exception as e:
    pass  # torch 미설치 시 무시

import streamlit as st

# transformers 라이브러리 import 및 상태 체크
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    st.error(f"Transformers 라이브러리를 불러올 수 없습니다: {e}")

st.title("🦙 TinyLlama 1.1B (CPU 전용) 데모")

if not TRANSFORMERS_AVAILABLE:
    st.stop()

@st.cache_resource
def load_llama_model():
    """TinyLlama 1.1B 모델 로드 (CPU Only)"""
    try:
        model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        )
        model.to("cpu")
        
        # 토크나이저 설정
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
            
        return model, tokenizer, "✅ TinyLlama 로드 성공!"
    except Exception as e:
        return None, None, f"❌ 모델 로드 실패: {str(e)}"

# 모델 로드
with st.spinner("TinyLlama 모델 로딩 중..."):
    model, tokenizer, status = load_llama_model()

st.write(status)

if model is not None and tokenizer is not None:
    st.success("TinyLlama 1.1B 준비 완료!")
    
    # 텍스트 생성 파라미터 설정
    st.sidebar.subheader("생성 설정")
    max_new_tokens = st.sidebar.slider("최대 토큰 수", 50, 500, 200)
    temperature = st.sidebar.slider("Temperature", 0.1, 1.0, 0.7)
    
    # 텍스트 생성 인터페이스
    st.subheader("챗 인터페이스")
    
    prompt = st.text_area("프롬프트 입력:", "The future of AI is", height=100)
    
    if st.button("생성 시작"):
        if prompt:
            with st.spinner("텍스트 생성 중..."):
                try:
                    # 토큰화
                    inputs = tokenizer(
                        prompt, 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True
                    )
                    
                    # 생성
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            do_sample=True,
                            pad_token_id=tokenizer.eos_token_id,
                            repetition_penalty=1.1
                        )
                    
                    # 디코딩
                    result = tokenizer.decode(
                        outputs[0], 
                        skip_special_tokens=True
                    )
                    
                    st.write("**생성 결과:**")
                    st.markdown(result)
                    
                except Exception as e:
                    st.error(f"생성 오류: {str(e)}")
        else:
            st.warning("프롬프트를 입력해주세요.")
else:
    st.error("모델 초기화 실패")

# 시스템 정보
st.sidebar.subheader("시스템 상태")
st.sidebar.write(f"Python: {sys.version.split()[0]}")

if TRANSFORMERS_AVAILABLE:
    st.sidebar.write(f"PyTorch: {torch.__version__}")
    st.sidebar.write(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        st.sidebar.write(f"GPU: {torch.cuda.get_device_name(0)}")
