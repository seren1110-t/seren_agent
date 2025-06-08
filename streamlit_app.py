# kospi_research_app.py
import os
import json

# PyTorch와 Streamlit 호환성 문제 해결 (반드시 가장 먼저 실행)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# PyTorch 임포트 및 호환성 수정
import torch
try:
    torch.classes.__path__ = [os.path.join(torch.__path__[0], 'classes')]
except Exception:
    try:
        if hasattr(torch, 'classes'):
            torch.classes.__path__._path = [os.path.join(torch.__path__[0], 'classes')]
    except Exception:
        pass

import streamlit as st
import pandas as pd
import sqlite3
import gdown
import zipfile
import tarfile
import torch.quantization
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import numpy as np
import gc

st.set_page_config(page_title="📈 KOSPI Analyst AI", layout="wide")

def cleanup_memory():
    """메모리 정리 함수"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def find_best_checkpoint(base_path, preferred_checkpoint="checkpoint-200"):
    """최적의 체크포인트 경로 찾기"""
    # 선호하는 체크포인트 먼저 확인
    preferred_path = os.path.join(base_path, preferred_checkpoint)
    if os.path.exists(preferred_path):
        adapter_config = os.path.join(preferred_path, "adapter_config.json")
        adapter_model = os.path.join(preferred_path, "adapter_model.safetensors")
        if not os.path.exists(adapter_model):
            adapter_model = os.path.join(preferred_path, "adapter_model.bin")
        
        if os.path.exists(adapter_config) and os.path.exists(adapter_model):
            return preferred_path, preferred_checkpoint
    
    # 선호하는 체크포인트가 없으면 다른 체크포인트 탐색
    checkpoint_dirs = []
    if os.path.exists(base_path):
        for item in os.listdir(base_path):
            if item.startswith("checkpoint-") and os.path.isdir(os.path.join(base_path, item)):
                checkpoint_dirs.append(item)
    
    # 체크포인트 번호 순으로 정렬 (높은 번호부터)
    checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]), reverse=True)
    
    for checkpoint_dir in checkpoint_dirs:
        checkpoint_path = os.path.join(base_path, checkpoint_dir)
        adapter_config = os.path.join(checkpoint_path, "adapter_config.json")
        adapter_model = os.path.join(checkpoint_path, "adapter_model.safetensors")
        if not os.path.exists(adapter_model):
            adapter_model = os.path.join(checkpoint_path, "adapter_model.bin")
        
        if os.path.exists(adapter_config) and os.path.exists(adapter_model):
            return checkpoint_path, checkpoint_dir
    
    return None, None

@st.cache_resource
def download_and_load_models():
    """Google Drive에서 모델 다운로드 및 CPU 최적화 QLoRA 로드"""
    
    # Google Drive 파일 ID
    base_model_id = "1CGpO7EO64hkUTU_eQQuZXbh-R84inkIc"
    qlora_adapter_id = "1l2F6a5HpmEmdOwTKOpu5UNRQG_jrXeW0"
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        cleanup_memory()
        
        # 베이스 모델 다운로드
        status_text.text("🔄 베이스 모델 다운로드 중... (1/8)")
        progress_bar.progress(10)
        
        if not os.path.exists("./base_model"):
            base_model_url = f"https://drive.google.com/uc?id={base_model_id}"
            gdown.download(base_model_url, "./my_base_model.tar.gz", quiet=False)
            
            with tarfile.open("./my_base_model.tar.gz", "r:gz") as tar:
                tar.extractall("./base_model/")
        
        # QLoRA 어댑터 다운로드
        status_text.text("🔄 QLoRA 어댑터 다운로드 중... (2/8)")
        progress_bar.progress(20)
        
        if not os.path.exists("./qlora_adapter"):
            qlora_url = f"https://drive.google.com/uc?id={qlora_adapter_id}"
            gdown.download(qlora_url, "./qlora_results.zip", quiet=False)
            
            with zipfile.ZipFile("./qlora_results.zip", 'r') as zip_ref:
                zip_ref.extractall("./qlora_adapter/")
        
        # 체크포인트 경로 확인
        status_text.text("🔧 QLoRA 체크포인트 확인 중... (3/8)")
        progress_bar.progress(30)
        
        # checkpoint-200을 우선적으로 찾기
        adapter_path, checkpoint_name = find_best_checkpoint("./qlora_adapter", "checkpoint-200")
        
        if adapter_path is None:
            st.error("❌ QLoRA 체크포인트를 찾을 수 없습니다.")
            return None, None
        
        st.info(f"✅ 사용할 체크포인트: {checkpoint_name}")
        
        # 체크포인트 정보 표시
        try:
            with open(os.path.join(adapter_path, "adapter_config.json"), 'r') as f:
                adapter_config = json.load(f)
            st.info(f"📋 LoRA 설정: r={adapter_config.get('r', 'N/A')}, alpha={adapter_config.get('lora_alpha', 'N/A')}")
        except:
            pass
        
        # 토크나이저 로드 (베이스 모델에서)
        status_text.text("📝 토크나이저 로드 중... (4/8)")
        progress_bar.progress(40)
        
        tokenizer = AutoTokenizer.from_pretrained(
            "./base_model",
            trust_remote_code=True,
            use_fast=False,  # CPU 환경에서 안정성 우선
            padding_side="right"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # 베이스 모델 로드 (CPU 최적화)
        status_text.text("🧠 베이스 모델 로드 중 (CPU 최적화)... (5/8)")
        progress_bar.progress(50)
        
        base_model = AutoModelForCausalLM.from_pretrained(
            "./base_model",
            torch_dtype=torch.float32,  # CPU에서는 float32 사용
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            use_cache=False,  # 메모리 절약
            torch_compile=False  # CPU에서는 컴파일 비활성화
        )
        
        cleanup_memory()
        
        # QLoRA 어댑터 설정 확인
        status_text.text("🔧 QLoRA 어댑터 설정 확인 중... (6/8)")
        progress_bar.progress(60)
        
        try:
            peft_config = PeftConfig.from_pretrained(adapter_path)
            st.info(f"✅ PEFT 설정: {peft_config.task_type}, target_modules={len(peft_config.target_modules)}개")
        except Exception as e:
            st.warning(f"PeftConfig 로드 실패: {e}")
        
        # QLoRA 어댑터 적용
        status_text.text(f"🔧 {checkpoint_name} 어댑터 적용 중... (7/8)")
        progress_bar.progress(70)
        
        model = PeftModel.from_pretrained(
            base_model, 
            adapter_path,
            torch_dtype=torch.float32,
            is_trainable=False  # 추론 전용
        )
        
        # 모델을 평가 모드로 설정
        model.eval()
        
        cleanup_memory()
        
        # CPU 동적 양자화 적용
        status_text.text("⚡ CPU 동적 양자화 적용 중... (8/8)")
        progress_bar.progress(80)
        
        try:
            with torch.no_grad():
                # Linear 레이어만 INT8로 양자화
                quantized_model = torch.quantization.quantize_dynamic(
                    model,
                    {torch.nn.Linear},
                    dtype=torch.qint8,
                    inplace=False
                )
            model = quantized_model
            st.success("✅ CPU 동적 양자화 적용 완료!")
        except Exception as e:
            st.warning(f"동적 양자화 실패, 원본 모델 사용: {e}")
        
        progress_bar.progress(90)
        
        # 모델 정보 표시
        def get_model_size_safe(model):
            try:
                param_size = sum(p.numel() * p.element_size() for p in model.parameters())
                buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
                return (param_size + buffer_size) / 1024 / 1024
            except:
                return 0
        
        model_size = get_model_size_safe(model)
        
        # 어댑터 정보 표시
        if hasattr(model, 'peft_config') and model.peft_config:
            config = list(model.peft_config.values())[0]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("모델 크기", f"{model_size:.1f} MB")
            with col2:
                st.metric("체크포인트", checkpoint_name.split("-")[1])
            with col3:
                st.metric("LoRA Rank", f"{config.r}")
            with col4:
                st.metric("LoRA Alpha", f"{config.lora_alpha}")
        
        progress_bar.progress(100)
        status_text.text("✅ QLoRA 모델 로드 완료!")
        st.success(f"⚡ CPU 최적화된 {checkpoint_name} QLoRA 모델이 성공적으로 로드되었습니다!")
        
        # 임시 파일 정리
        try:
            for temp_file in ["./my_base_model.tar.gz", "./qlora_results.zip"]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        except:
            pass
        
        cleanup_memory()
        
        # UI 정리
        progress_bar.empty()
        status_text.empty()
        
        return model, tokenizer
        
    except Exception as e:
        st.error(f"❌ 모델 로드 실패: {e}")
        progress_bar.empty()
        status_text.empty()
        cleanup_memory()
        return None, None

@st.cache_data
def load_data(db_name="financial_data.db", table_name="financial_data"):
    """데이터베이스에서 데이터 로드"""
    try:
        conn = sqlite3.connect(db_name)
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        conn.close()
        
        # 숫자 컬럼들을 numeric으로 변환하고 NaN 처리
        numeric_columns = ['PER_최근', 'PBR_최근', 'ROE_최근', '부채비율_최근', '현재가', 
                          '유보율_최근', '매출액_최근', '영업이익_최근', '순이익_최근']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(0)
        
        return df
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return pd.DataFrame()

def generate_ai_response(model, tokenizer, question, company_data):
    """CPU 최적화된 QLoRA 모델을 사용한 AI 응답 생성"""
    if model is None or tokenizer is None:
        return "AI 모델이 로드되지 않았습니다."
    
    # QLoRA 파인튜닝 형식에 맞는 프롬프트 생성 (첫 번째, 두 번째 코드 참고)
    company_info = f"""종목명: {company_data['종목명']}
티커: {company_data['티커']}
현재가: {company_data['현재가']:,.0f}원
PER: {company_data['PER_최근']:.2f}
PBR: {company_data['PBR_최근']:.2f}
ROE: {company_data['ROE_최근']:.2f}%
부채비율: {company_data['부채비율_최근']:.2f}%"""
    
    # QLoRA 파인튜닝된 모델에 맞는 프롬프트 형식 사용
    prompt = f"""질문: {question}
정보: {company_info}
답변:"""
    
    try:
        # CPU 최적화된 토큰화
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,  # CPU에서는 짧은 길이 사용
            padding=True,
            add_special_tokens=True
        )
        
        # CPU에서 안전한 추론
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,  # CPU에서는 토큰 수 제한
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_beams=1,  # CPU에서는 beam search 비활성화
                early_stopping=True,
                use_cache=False,  # 메모리 절약
                repetition_penalty=1.1,
                no_repeat_ngram_size=2  # 반복 방지
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = response[len(prompt):].strip()
        
        # 메모리 정리
        del inputs, outputs
        cleanup_memory()
        
        return generated_text if generated_text else "응답을 생성할 수 없습니다."
        
    except Exception as e:
        cleanup_memory()
        return f"응답 생성 중 오류가 발생했습니다: {e}"

def get_initial(korean_char):
    """한글 초성 추출"""
    try:
        ch_code = ord(korean_char) - ord('가')
        if 0 <= ch_code < 11172:
            cho = ch_code // 588
            return ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ'][cho]
    except:
        pass
    return ""

# AI 모델 로드 (앱 시작 시 한 번만)
if 'model_loaded' not in st.session_state:
    st.info("🤖 CPU 최적화된 QLoRA checkpoint-200 모델을 로드합니다...")
    model, tokenizer = download_and_load_models()
    st.session_state.model = model
    st.session_state.tokenizer = tokenizer
    st.session_state.model_loaded = True
    
    if model is not None:
        st.success("✅ CPU 최적화된 QLoRA checkpoint-200 모델 로드 완료! 이제 전문적인 금융 분석이 가능합니다.")
        st.rerun()

# 데이터 로드
df = load_data()

if df.empty:
    st.error("❌ 데이터를 로드할 수 없습니다.")
    st.stop()

# 사이드바 필터
st.sidebar.header("📂 필터 옵션")

# 초성 필터
initials = ['전체', 'ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
selected_initial = st.sidebar.selectbox("🔡 종목명 초성:", initials)

if selected_initial != "전체":
    df = df[df["종목명"].apply(lambda x: get_initial(x[0]) == selected_initial if x else "")]

# 텍스트 검색
search_term = st.sidebar.text_input("🔍 종목명 또는 티커 검색")
if search_term:
    mask1 = df["종목명"].str.contains(search_term, case=False, na=False)
    mask2 = df["티커"].str.contains(search_term, case=False, na=False)
    df = df[mask1 | mask2]

종목_list = df["종목명"].tolist()

if not 종목_list:
    st.warning("❌ 조건에 맞는 종목이 없습니다.")
    st.stop()

선택한_종목 = st.sidebar.selectbox("📌 종목 선택:", 종목_list)
종목_df = df[df["종목명"] == 선택한_종목].iloc[0]

# 메인 컨텐츠
st.title(f"📊 {선택한_종목} ({종목_df['티커']}) QLoRA-200 AI 분석")

col1, col2 = st.columns(2)

with col1:
    st.metric("현재가", f"{종목_df['현재가']:,.0f}원")
    st.metric("ROE (최근)", f"{종목_df['ROE_최근']:.2f}%")
    st.metric("PER (최근)", f"{종목_df['PER_최근']:.2f}")
    st.metric("PBR (최근)", f"{종목_df['PBR_최근']:.2f}")
    st.metric("부채비율", f"{종목_df['부채비율_최근']:.2f}%")

with col2:
    # 안전한 메트릭 표시
    metrics = [
        ("유보율", "유보율_최근", "%"),
        ("매출액", "매출액_최근", "원"),
        ("영업이익", "영업이익_최근", "원"),
        ("순이익", "순이익_최근", "원")
    ]
    
    for label, col_name, unit in metrics:
        try:
            if col_name in 종목_df and pd.notna(종목_df[col_name]):
                if unit == "원":
                    st.metric(label, f"{종목_df[col_name]:,.0f}{unit}")
                else:
                    st.metric(label, f"{종목_df[col_name]:.2f}{unit}")
            else:
                st.metric(label, "데이터 없음")
        except:
            st.metric(label, "데이터 없음")

# 그래프
st.markdown("### 📈 주가 추이")
price_cols = [col for col in df.columns if col.isdigit() and len(col) == 8]
if price_cols:
    try:
        price_series = 종목_df[price_cols].astype(float)
        price_series.index = pd.to_datetime(price_cols, format='%Y%m%d')
        chart_df = price_series.reset_index().rename(columns={'index': '날짜'})
        st.line_chart(chart_df.set_index("날짜"))
    except:
        st.info("주가 차트 데이터를 표시할 수 없습니다.")

# 뉴스
st.markdown("### 📰 최근 뉴스")
if "최신뉴스" in 종목_df and isinstance(종목_df["최신뉴스"], str) and 종목_df["최신뉴스"].strip():
    for i, link in enumerate(종목_df["최신뉴스"].splitlines(), 1):
        if link.strip():
            st.markdown(f"{i}. [뉴스 링크]({link.strip()})")
else:
    st.info("최근 뉴스가 없습니다.")

# AI 리서치 질의
st.markdown("### 🤖 QLoRA Checkpoint-200 AI 리서치 질의")

# 미리 정의된 질문들
preset_questions = [
    "이 종목의 투자 매력도는 어떤가요?",
    "PER과 PBR 지표를 어떻게 해석해야 하나요?",
    "현재 재무 상태의 강점과 약점은 무엇인가요?",
    "이 종목의 리스크 요인은 무엇인가요?",
    "동종 업계 대비 경쟁력은 어떤가요?"
]

col1, col2 = st.columns([3, 1])

with col1:
    user_question = st.text_input("🧠 궁금한 점을 입력하세요:", 
                                placeholder="예: 이 종목의 PER이 높으면 어떤 해석이 가능해?")

with col2:
    selected_preset = st.selectbox("📋 미리 정의된 질문:", ["직접 입력"] + preset_questions)

if selected_preset != "직접 입력":
    user_question = selected_preset

if user_question:
    if st.session_state.get('model') is not None:
        if st.button("🔍 QLoRA-200 AI 분석 요청", type="primary"):
            with st.spinner("🤖 QLoRA checkpoint-200 모델이 CPU에서 분석 중입니다..."):
                ai_response = generate_ai_response(
                    st.session_state.model, 
                    st.session_state.tokenizer, 
                    user_question, 
                    종목_df
                )
            
            st.markdown("#### 🎯 QLoRA-200 AI 분석 결과")
            st.markdown(ai_response)
            
            # 분석 결과 저장 옵션
            analysis_text = f"""
# {선택한_종목} QLoRA Checkpoint-200 AI 분석 결과

**질문:** {user_question}

**QLoRA-200 AI 분석:**
{ai_response}

**기본 정보:**
- 종목명: {종목_df['종목명']}
- 티커: {종목_df['티커']}
- 현재가: {종목_df['현재가']:,.0f}원
- PER: {종목_df['PER_최근']:.2f}
- PBR: {종목_df['PBR_최근']:.2f}
- ROE: {종목_df['ROE_최근']:.2f}%

*본 분석은 QLoRA checkpoint-200으로 파인튜닝된 모델을 사용하여 생성되었습니다.*
"""
            
            st.download_button(
                label="📥 QLoRA-200 분석 보고서 다운로드",
                data=analysis_text,
                file_name=f"{선택한_종목}_QLoRA200_AI분석_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown"
            )
    else:
        st.warning("🔎 QLoRA-200 모델이 로드되지 않았습니다.")

# QLoRA 체크포인트 정보 표시
if st.session_state.get('model_loaded'):
    with st.expander("⚡ QLoRA Checkpoint-200 모델 정보"):
        st.markdown("""
        **QLoRA Checkpoint-200 모델 특징:**
        - 🔹 **체크포인트**: checkpoint-200 (최적화된 훈련 단계)
        - 🔹 **QLoRA 파인튜닝**: 금융 도메인 특화 학습 완료
        - 🔹 **CPU 동적 양자화**: INT8 양자화로 메모리 효율성 극대화
        - 🔹 **LoRA 어댑터**: 효율적인 파라미터 업데이트 방식
        - 🔹 **메모리 최적화**: CPU 환경에 특화된 추론 최적화
        
        **기술적 세부사항:**
        - 베이스 모델: Llama 기반 모델
        - 양자화: 4bit → INT8 동적 양자화
        - 어댑터: LoRA (Low-Rank Adaptation)
        - 훈련 데이터: 금융 Q&A 데이터셋
        - 최적화: CPU 추론에 특화된 설정
        """)
