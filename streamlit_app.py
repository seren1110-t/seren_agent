import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import zipfile
import os
import tempfile
import gdown
import time
from pathlib import Path
import json

# 페이지 설정
st.set_page_config(
    page_title="KoAlpaca CPU 모델 테스트",
    page_icon="🤖",
    layout="wide"
)

def initialize_session_state():
    """세션 상태 초기화"""
    if 'gdrive_file_id' not in st.session_state:
        st.session_state.gdrive_file_id = ""
    
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    
    if 'model' not in st.session_state:
        st.session_state.model = None
    
    if 'tokenizer' not in st.session_state:
        st.session_state.tokenizer = None
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'model_path' not in st.session_state:
        st.session_state.model_path = None

def download_and_extract_model(file_id):
    """구글 드라이브에서 모델 다운로드 및 압축 해제"""
    try:
        # 임시 디렉토리 생성
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, "koalpaca_cpu_deployment.zip")
        extract_path = os.path.join(temp_dir, "koalpaca_cpu_deployment")
        
        # 구글 드라이브에서 다운로드
        st.info("🔄 구글 드라이브에서 모델 다운로드 중...")
        download_url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(download_url, zip_path, quiet=False)
        
        # 압축 해제
        st.info("📦 모델 압축 해제 중...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        # 압축 해제된 폴더 구조 확인
        model_path = None
        for root, dirs, files in os.walk(extract_path):
            if "tokenizer.json" in files or "config.json" in files or "cpu_quantized_model.pt" in files:
                model_path = root
                break
        
        if model_path is None:
            # 첫 번째 하위 디렉토리를 모델 경로로 사용
            subdirs = [d for d in os.listdir(extract_path) if os.path.isdir(os.path.join(extract_path, d))]
            if subdirs:
                model_path = os.path.join(extract_path, subdirs[0])
            else:
                model_path = extract_path
        
        st.success(f"✅ 모델 다운로드 완료: {model_path}")
        return model_path
        
    except Exception as e:
        st.error(f"❌ 모델 다운로드 실패: {str(e)}")
        return None

def check_model_files(model_path):
    """모델 파일 구조 확인"""
    if not os.path.exists(model_path):
        return False, "모델 경로가 존재하지 않습니다."
    
    files = os.listdir(model_path)
    
    expected_files = {
        'cpu_quantized_model.pt': os.path.exists(os.path.join(model_path, 'cpu_quantized_model.pt')),
        'cpu_model_config.json': os.path.exists(os.path.join(model_path, 'cpu_model_config.json')),
        'cpu_loading_example.py': os.path.exists(os.path.join(model_path, 'cpu_loading_example.py')),
        'tokenizer_files': any(f.startswith('tokenizer') for f in files) or 'tokenizer.json' in files
    }
    
    return expected_files, f"발견된 파일: {files}"

def load_koalpaca_model(model_path):
    """KoAlpaca CPU 모델 로드 (내장 파일 구조에 맞춤)"""
    try:
        st.info("🧠 KoAlpaca 모델 로딩 중...")
        
        # 파일 구조 확인
        file_check, file_info = check_model_files(model_path)
        st.info(f"📁 {file_info}")
        
        # 1. 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False
        )
        
        # 2. cpu_quantized_model.pt 파일이 있는지 확인
        quantized_model_path = os.path.join(model_path, "cpu_quantized_model.pt")
        config_path = os.path.join(model_path, "cpu_model_config.json")
        
        if os.path.exists(quantized_model_path):
            # 양자화된 모델 로드
            st.info("📦 양자화된 CPU 모델 로드 중...")
            
            # 설정 파일 로드
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_info = json.load(f)
                st.info(f"📋 모델 설정: {config_info.get('model_type', 'Unknown')}")
            
            # 베이스 모델 먼저 로드
            model = AutoModelForCausalLM.from_pretrained(
                "beomi/KoAlpaca-Polyglot-5.8B",
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # 양자화된 state_dict 로드
            checkpoint = torch.load(quantized_model_path, map_location="cpu")
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                st.success("✅ 양자화된 모델 로드 완료!")
            else:
                st.warning("⚠️ 표준 모델로 로드합니다.")
        else:
            # 표준 모델 로드
            st.info("📦 표준 모델 로드 중...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        
        model.eval()
        
        st.success("✅ KoAlpaca 모델 로드 완료!")
        return model, tokenizer
        
    except Exception as e:
        st.error(f"❌ 모델 로드 실패: {str(e)}")
        return None, None

def generate_response(model, tokenizer, question, max_new_tokens=200):
    """응답 생성"""
    try:
        # 프롬프트 포맷팅
        prompt = f"### 질문: {question}\n\n### 답변:"
        
        # 토크나이징
        inputs = tokenizer(
            prompt, 
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        
        # 생성
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3
            )
        
        # 응답 디코딩
        generated_ids = outputs[0][len(inputs.input_ids[0]):]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return response.strip()
        
    except Exception as e:
        return f"응답 생성 중 오류 발생: {str(e)}"

def render_sidebar():
    """사이드바 렌더링"""
    with st.sidebar:
        st.header("📋 모델 정보")
        st.info("""
        **모델**: KoAlpaca-Polyglot-5.8B CPU 버전
        **용도**: 한국어 질의응답
        **최적화**: CPU 추론용
        """)
        
        st.header("🔧 설정")
        max_tokens = st.slider("최대 토큰 수", 50, 500, 200)
        
        st.header("📝 사용법")
        st.markdown("""
        1. 구글 드라이브 파일 ID를 입력하세요
        2. 모델 다운로드 버튼을 클릭하세요
        3. 질문을 입력하고 응답을 생성하세요
        """)
        
        # 모델 상태 표시
        st.header("🔍 모델 상태")
        if st.session_state.model_loaded:
            st.success("✅ 모델 로드됨")
        else:
            st.warning("⚠️ 모델 로드 필요")
        
        return max_tokens

def render_file_id_input():
    """파일 ID 입력 섹션 렌더링"""
    st.header("🔗 구글 드라이브 설정")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        file_id = st.text_input(
            "구글 드라이브 파일 ID 입력",
            value=st.session_state.gdrive_file_id,
            help="구글 드라이브 공유 링크에서 파일 ID를 추출하여 입력하세요",
            key="file_id_input"
        )
    
    with col2:
        st.markdown("**파일 ID 추출 방법:**")
        st.code("https://drive.google.com/file/d/FILE_ID/view")
        st.caption("FILE_ID 부분을 복사하세요")
    
    # session_state 업데이트
    if file_id != st.session_state.gdrive_file_id:
        st.session_state.gdrive_file_id = file_id
    
    return file_id

def render_model_download():
    """모델 다운로드 섹션 렌더링"""
    st.header("📥 모델 다운로드 및 로드")
    
    if not st.session_state.gdrive_file_id:
        st.warning("⚠️ 구글 드라이브 파일 ID를 입력해주세요.")
        return False
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        download_btn = st.button("🔄 모델 다운로드 시작", type="primary")
    
    with col2:
        if st.button("🗑️ 모델 초기화"):
            st.session_state.model_loaded = False
            st.session_state.model = None
            st.session_state.tokenizer = None
            st.session_state.model_path = None
            st.rerun()
    
    if download_btn:
        with st.spinner("모델 다운로드 중... 잠시만 기다려주세요."):
            model_path = download_and_extract_model(st.session_state.gdrive_file_id)
            
            if model_path:
                st.session_state.model_path = model_path
                model, tokenizer = load_koalpaca_model(model_path)
                
                if model and tokenizer:
                    st.session_state.model = model
                    st.session_state.tokenizer = tokenizer
                    st.session_state.model_loaded = True
                    st.success("🎉 모델 로드 완료! 이제 질문을 입력할 수 있습니다.")
                    st.rerun()
                else:
                    st.error("❌ 모델 로드에 실패했습니다.")
            else:
                st.error("❌ 모델 다운로드에 실패했습니다.")
    
    return st.session_state.model_loaded

def render_qa_interface(max_tokens):
    """질의응답 인터페이스 렌더링"""
    st.header("💬 질의응답 테스트")
    
    # 예시 질문들
    example_questions = [
        "삼성전자의 재무상태는 어떻습니까?",
        "네이버의 사업 전망을 분석해주세요.",
        "카카오의 주요 사업 영역은 무엇인가요?",
        "한국 IT 산업의 현황을 설명해주세요.",
        "투자할 때 고려해야 할 요소들은 무엇인가요?"
    ]
    
    # 질문 입력
    col1, col2 = st.columns([3, 1])
    
    with col1:
        question = st.text_area(
            "질문을 입력하세요:",
            height=100,
            placeholder="예: 삼성전자의 재무상태는 어떻습니까?",
            key="question_input"
        )
    
    with col2:
        st.markdown("**예시 질문:**")
        for i, ex_q in enumerate(example_questions):
            if st.button(f"예시 {i+1}", key=f"example_{i}"):
                st.session_state.question_input = ex_q
                st.rerun()
    
    # 응답 생성 버튼
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        generate_btn = st.button("🚀 응답 생성", type="primary")
    
    with col2:
        if st.button("🗑️ 대화 초기화"):
            st.session_state.chat_history = []
            st.rerun()
    
    # 응답 생성 처리
    if generate_btn and question.strip():
        if st.session_state.model and st.session_state.tokenizer:
            with st.spinner("🤔 응답 생성 중..."):
                start_time = time.time()
                
                response = generate_response(
                    st.session_state.model, 
                    st.session_state.tokenizer, 
                    question,
                    max_tokens
                )
                
                end_time = time.time()
                generation_time = end_time - start_time
                
                # 대화 기록에 추가
                st.session_state.chat_history.append({
                    'question': question,
                    'response': response,
                    'time': generation_time
                })
                
                st.rerun()
        else:
            st.error("❌ 모델이 로드되지 않았습니다.")

def render_chat_history():
    """대화 기록 렌더링"""
    if st.session_state.chat_history:
        st.header("📜 대화 기록")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"💬 대화 {len(st.session_state.chat_history) - i}", expanded=(i == 0)):
                st.markdown(f"**❓ 질문:** {chat['question']}")
                st.markdown(f"**🤖 응답:** {chat['response']}")
                st.caption(f"⏱️ 생성 시간: {chat['time']:.2f}초")

def render_system_info():
    """시스템 정보 렌더링"""
    with st.expander("🔧 시스템 정보"):
        system_info = {
            "Python 버전": "3.8+",
            "PyTorch 버전": torch.__version__,
            "디바이스": "CPU",
            "모델 상태": "로드됨" if st.session_state.model_loaded else "로드 안됨",
            "모델 경로": st.session_state.model_path if st.session_state.model_path else "없음",
            "대화 기록 수": len(st.session_state.chat_history)
        }
        st.json(system_info)

def main():
    """메인 함수"""
    # 세션 상태 초기화
    initialize_session_state()
    
    # 페이지 제목
    st.title("🤖 KoAlpaca CPU 모델 테스트")
    st.markdown("구글 드라이브에서 KoAlpaca CPU 모델을 다운로드하고 검색 기능을 테스트합니다.")
    
    # 사이드바 렌더링
    max_tokens = render_sidebar()
    
    # 파일 ID 입력
    file_id = render_file_id_input()
    
    # 모델 다운로드
    model_loaded = render_model_download()
    
    # 모델이 로드되지 않았으면 안내 메시지 표시
    if not model_loaded:
        st.info("👆 먼저 '모델 다운로드 시작' 버튼을 클릭하여 모델을 로드해주세요.")
        return
    
    # 질의응답 인터페이스
    render_qa_interface(max_tokens)
    
    # 대화 기록
    render_chat_history()
    
    # 시스템 정보
    render_system_info()

if __name__ == "__main__":
    main()
