import streamlit as st
import sys
import os

# torch import 문제 해결을 위한 환경 설정
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    st.error(f"Transformers 라이브러리를 불러올 수 없습니다: {e}")

st.title("🤖 작은 LLM 모델 로드 확인")

if not TRANSFORMERS_AVAILABLE:
    st.stop()

@st.cache_resource
def load_simple_model():
    """가장 작은 GPT 모델 로드"""
    try:
        model_name = "sshleifer/tiny-gpt2"  # 매우 작은 테스트용 모델
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # 패딩 토큰 설정
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return model, tokenizer, "✅ 모델 로드 성공!"
    except Exception as e:
        return None, None, f"❌ 모델 로드 실패: {str(e)}"

# 모델 로드
with st.spinner("모델 로딩 중..."):
    model, tokenizer, status = load_simple_model()

st.write(status)

if model is not None and tokenizer is not None:
    st.success("모델이 정상적으로 로드되었습니다!")
    
    # 간단한 텍스트 생성 테스트
    st.subheader("텍스트 생성 테스트")
    
    prompt = st.text_input("프롬프트 입력:", "Hello")
    
    if st.button("생성"):
        if prompt:
            with st.spinner("생성 중..."):
                try:
                    # 토큰화
                    inputs = tokenizer.encode(prompt, return_tensors="pt")
                    
                    # 생성
                    with torch.no_grad():
                        outputs = model.generate(
                            inputs, 
                            max_length=inputs.shape[1] + 20,
                            num_return_sequences=1,
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=tokenizer.eos_token_id
                        )
                    
                    # 디코딩
                    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    st.write("**생성 결과:**")
                    st.write(result)
                    
                except Exception as e:
                    st.error(f"생성 중 오류: {e}")
        else:
            st.warning("프롬프트를 입력해주세요.")
else:
    st.error("모델을 로드할 수 없습니다.")

# 시스템 정보
st.sidebar.write("**시스템 정보:**")
st.sidebar.write(f"Python: {sys.version.split()[0]}")

if TRANSFORMERS_AVAILABLE:
    st.sidebar.write(f"PyTorch: {torch.__version__}")
    st.sidebar.write(f"CUDA 사용 가능: {torch.cuda.is_available()}")
else:
    st.sidebar.write("PyTorch: 불러오기 실패")
