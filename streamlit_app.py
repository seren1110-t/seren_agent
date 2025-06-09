import os
import sys
import gc

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
        
        st.info("모델 다운로드 중... (첫 실행 시 시간이 걸립니다)")
        
        # 토크나이저 먼저 로드
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        st.info("모델 로딩 중...")
        
        # 모델 로드 (accelerate 없이 CPU 전용 설정)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        
        # CPU로 명시적 이동
        model = model.to("cpu")
        model.eval()  # 평가 모드로 설정
        
        # 토크나이저 설정
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # 생성에는 left padding이 더 적합
        
        # 메모리 정리
        gc.collect()
            
        return model, tokenizer, "✅ TinyLlama 로드 성공!"
        
    except Exception as e:
        error_msg = str(e)
        if "accelerate" in error_msg.lower():
            return None, None, f"❌ accelerate 라이브러리가 필요합니다: pip install accelerate"
        elif "connection" in error_msg.lower() or "timeout" in error_msg.lower():
            return None, None, f"❌ 네트워크 연결 오류: {error_msg}"
        else:
            return None, None, f"❌ 모델 로드 실패: {error_msg}"

def generate_text(model, tokenizer, prompt, max_new_tokens=200, temperature=0.7):
    """텍스트 생성 함수"""
    try:
        # 프롬프트 길이 제한 (안전한 길이로)
        max_input_length = 512  # 입력 길이 제한
        
        # 토큰화 (패딩 없이, CPU 텐서로)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
            padding=False  # 패딩 제거
        )
        
        # CPU로 명시적 이동
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
        
        input_length = inputs['input_ids'].shape[1]
        
        # 생성 길이 조정
        if input_length + max_new_tokens > 1024:  # 안전한 총 길이
            max_new_tokens = max(50, 1024 - input_length)
        
        # 생성 파라미터 최적화
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": True,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": True
        }
        
        # 메모리 정리
        gc.collect()
        
        # 생성
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **generation_config
            )
        
        # 새로 생성된 부분만 디코딩
        new_tokens = outputs[0][input_length:]
        result = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return prompt + result
        
    except Exception as e:
        raise Exception(f"생성 중 오류: {str(e)}")

# 모델 로드
with st.spinner("TinyLlama 모델 로딩 중... (처음 실행 시 시간이 걸릴 수 있습니다)"):
    model, tokenizer, status = load_llama_model()

st.write(status)

if model is not None and tokenizer is not None:
    st.success("TinyLlama 1.1B 준비 완료!")
    
    # 텍스트 생성 파라미터 설정
    st.sidebar.subheader("생성 설정")
    max_new_tokens = st.sidebar.slider("최대 새 토큰 수", 50, 300, 150)
    temperature = st.sidebar.slider("Temperature", 0.1, 1.0, 0.7)
    
    # 사용 가이드
    st.sidebar.subheader("사용 가이드")
    st.sidebar.info("""
    - 프롬프트는 간결하게 작성하세요
    - 긴 텍스트는 자동으로 잘립니다
    - CPU 전용이므로 생성에 시간이 걸립니다
    - accelerate 라이브러리 없이 작동합니다
    """)
    
    # accelerate 설치 안내
    st.sidebar.subheader("라이브러리 설치")
    with st.sidebar.expander("필요한 라이브러리"):
        st.code("""
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers
pip install streamlit
        """, language="bash")
    
    # 텍스트 생성 인터페이스
    st.subheader("텍스트 생성")
    
    # 예제 프롬프트 제공
    example_prompts = [
        "The future of AI is",
        "Once upon a time,",
        "Python is a programming language that",
        "Climate change is"
    ]
    
    selected_example = st.selectbox("예제 프롬프트 선택:", ["직접 입력"] + example_prompts)
    
    if selected_example != "직접 입력":
        prompt = st.text_area("프롬프트 입력:", selected_example, height=100)
    else:
        prompt = st.text_area("프롬프트 입력:", "", height=100)
    
    # 프롬프트 길이 표시
    if prompt:
        token_count = len(tokenizer.encode(prompt))
        st.caption(f"현재 프롬프트 토큰 수: {token_count}")
        if token_count > 512:
            st.warning("프롬프트가 너무 깁니다. 512 토큰으로 잘립니다.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        generate_button = st.button("🚀 생성 시작", type="primary")
    
    with col2:
        if st.button("🗑️ 결과 지우기"):
            if 'generated_text' in st.session_state:
                del st.session_state['generated_text']
            st.rerun()
    
    if generate_button:
        if prompt.strip():
            with st.spinner("텍스트 생성 중... (CPU에서 실행되므로 시간이 걸립니다)"):
                try:
                    # 진행률 표시
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("토큰화 중...")
                    progress_bar.progress(25)
                    
                    status_text.text("텍스트 생성 중...")
                    progress_bar.progress(50)
                    
                    result = generate_text(
                        model, 
                        tokenizer, 
                        prompt.strip(), 
                        max_new_tokens, 
                        temperature
                    )
                    
                    progress_bar.progress(100)
                    status_text.text("완료!")
                    
                    # 결과 저장 및 표시
                    st.session_state['generated_text'] = result
                    
                    # 진행률 표시 제거
                    progress_bar.empty()
                    status_text.empty()
                    
                except Exception as e:
                    st.error(f"생성 오류: {str(e)}")
                    st.info("다시 시도해보거나 더 짧은 프롬프트를 사용해보세요.")
        else:
            st.warning("프롬프트를 입력해주세요.")
    
    # 생성 결과 표시
    if 'generated_text' in st.session_state:
        st.subheader("생성 결과")
        st.markdown(f"```\n{st.session_state['generated_text']}\n```")
        
        # 결과 복사 버튼
        st.download_button(
            label="📋 텍스트 다운로드",
            data=st.session_state['generated_text'],
            file_name="generated_text.txt",
            mime="text/plain"
        )

else:
    st.error("모델 초기화 실패")
    st.info("인터넷 연결을 확인하고 필요한 라이브러리가 설치되어 있는지 확인하세요.")

# 시스템 정보
st.sidebar.subheader("시스템 상태")
st.sidebar.write(f"Python: {sys.version.split()[0]}")
if TRANSFORMERS_AVAILABLE:
    st.sidebar.write(f"PyTorch: {torch.__version__}")
    st.sidebar.write(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        st.sidebar.write(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        st.sidebar.write("현재 CPU 모드로 실행 중")

# 성능 팁
with st.sidebar.expander("성능 최적화 팁"):
    st.write("""
    **CPU 최적화:**
    - 프롬프트를 짧게 유지
    - 토큰 수를 200개 이하로 제한
    - 한 번에 하나씩 생성
    
    **메모리 절약:**
    - 브라우저 탭을 여러 개 열지 마세요
    - 다른 무거운 프로그램 종료
    """)
