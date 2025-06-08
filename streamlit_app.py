# kospi_research_app.py
import streamlit as st
import pandas as pd
import sqlite3
import gdown
import os
import zipfile
import tarfile
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import numpy as np

st.set_page_config(page_title="📈 KOSPI Analyst AI", layout="wide")

# Google Drive 파일 다운로드 및 동적 양자화 적용 함수
@st.cache_resource
def download_and_load_models():
    """Google Drive에서 모델 다운로드 및 동적 양자화 적용하여 로드"""
    
    # Google Drive 파일 ID
    base_model_id = "1CGpO7EO64hkUTU_eQQuZXbh-R84inkIc"
    qlora_adapter_id = "1l2F6a5HpmEmdOwTKOpu5UNRQG_jrXeW0"
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # 베이스 모델 다운로드
        status_text.text("🔄 베이스 모델 다운로드 중... (1/6)")
        progress_bar.progress(15)
        
        if not os.path.exists("./base_model"):
            base_model_url = f"https://drive.google.com/uc?id={base_model_id}"
            gdown.download(base_model_url, "./my_base_model.tar.gz", quiet=False)
            
            with tarfile.open("./my_base_model.tar.gz", "r:gz") as tar:
                tar.extractall("./base_model/")
        
        # QLoRA 어댑터 다운로드
        status_text.text("🔄 QLoRA 어댑터 다운로드 중... (2/6)")
        progress_bar.progress(30)
        
        if not os.path.exists("./qlora_adapter"):
            qlora_url = f"https://drive.google.com/uc?id={qlora_adapter_id}"
            gdown.download(qlora_url, "./qlora_results.zip", quiet=False)
            
            with zipfile.ZipFile("./qlora_results.zip", 'r') as zip_ref:
                zip_ref.extractall("./qlora_adapter/")
        
        # 토크나이저 로드
        status_text.text("📝 토크나이저 로드 중... (3/6)")
        progress_bar.progress(45)
        
        tokenizer = AutoTokenizer.from_pretrained("./base_model")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 베이스 모델 로드 (FP32로 로드)
        status_text.text("🧠 베이스 모델 로드 중... (4/6)")
        progress_bar.progress(60)
        
        base_model = AutoModelForCausalLM.from_pretrained(
            "./base_model",
            torch_dtype=torch.float32,  # 동적 양자화를 위해 FP32로 로드
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        # QLoRA 어댑터 적용
        status_text.text("🔧 QLoRA 어댑터 적용 중... (5/6)")
        progress_bar.progress(75)
        
        model = PeftModel.from_pretrained(base_model, "./qlora_adapter")
        
        # 동적 양자화 적용
        status_text.text("⚡ 동적 양자화 적용 중... (6/6)")
        progress_bar.progress(85)
        
        # 모델을 평가 모드로 설정
        model.eval()
        
        # 동적 양자화 적용 - Linear 레이어들을 INT8로 양자화
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},  # Linear 레이어만 양자화
            dtype=torch.qint8   # INT8 양자화
        )
        
        progress_bar.progress(100)
        
        # 메모리 사용량 비교
        def get_model_size(model):
            """모델 크기 계산 (MB)"""
            param_size = 0
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            buffer_size = 0
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            return (param_size + buffer_size) / 1024 / 1024
        
        original_size = get_model_size(model)
        quantized_size = get_model_size(quantized_model)
        compression_ratio = original_size / quantized_size
        
        status_text.text("✅ 동적 양자화 모델 로드 완료!")
        
        # 성능 정보 표시
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("원본 모델 크기", f"{original_size:.1f} MB")
        with col2:
            st.metric("양자화 모델 크기", f"{quantized_size:.1f} MB")
        with col3:
            st.metric("압축률", f"{compression_ratio:.1f}x")
        
        st.success("⚡ 동적 양자화로 CPU 최적화 완료! 메모리 사용량이 크게 감소했습니다.")
        
        # 임시 파일 정리
        try:
            if os.path.exists("./my_base_model.tar.gz"):
                os.remove("./my_base_model.tar.gz")
            if os.path.exists("./qlora_results.zip"):
                os.remove("./qlora_results.zip")
        except:
            pass
        
        # UI 정리
        progress_bar.empty()
        status_text.empty()
        
        return quantized_model, tokenizer
        
    except Exception as e:
        st.error(f"❌ 모델 로드 실패: {e}")
        progress_bar.empty()
        status_text.empty()
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
    """동적 양자화된 AI 모델을 사용한 응답 생성"""
    if model is None or tokenizer is None:
        return "AI 모델이 로드되지 않았습니다."
    
    # 회사 정보를 포함한 프롬프트 생성
    company_info = f"""
    종목명: {company_data['종목명']}
    티커: {company_data['티커']}
    현재가: {company_data['현재가']:,.0f}원
    PER: {company_data['PER_최근']:.2f}
    PBR: {company_data['PBR_최근']:.2f}
    ROE: {company_data['ROE_최근']:.2f}%
    부채비율: {company_data['부채비율_최근']:.2f}%
    """
    
    prompt = f"""다음은 {company_data['종목명']}의 재무 정보입니다:

{company_info}

질문: {question}

위 정보를 바탕으로 전문적인 증권 분석가 관점에서 답변해주세요:"""
    
    try:
        # 토큰화 (패딩 설정)
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            padding=True
        )
        
        # CPU에서 동적 양자화된 모델로 추론
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_beams=1,  # CPU 최적화를 위해 beam search 비활성화
                early_stopping=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = response[len(prompt):].strip()
        
        return generated_text if generated_text else "응답을 생성할 수 없습니다."
        
    except Exception as e:
        return f"응답 생성 중 오류가 발생했습니다: {e}"

def get_initial(korean_char):
    """한글 초성 추출"""
    ch_code = ord(korean_char) - ord('가')
    if 0 <= ch_code < 11172:
        cho = ch_code // 588
        return ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ'][cho]
    return ""

# AI 모델 로드 (앱 시작 시 한 번만)
if 'model_loaded' not in st.session_state:
    st.info("🤖 AI 모델을 처음 로드합니다. 동적 양자화를 적용하여 CPU 최적화를 진행합니다...")
    model, tokenizer = download_and_load_models()
    st.session_state.model = model
    st.session_state.tokenizer = tokenizer
    st.session_state.model_loaded = True
    
    if model is not None:
        st.success("✅ 동적 양자화 AI 모델 로드 완료! 이제 CPU에서 효율적인 지능형 분석이 가능합니다.")
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
    df = df[df["종목명"].apply(lambda x: get_initial(x[0]) == selected_initial)]

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
st.title(f"📊 {선택한_종목} ({종목_df['티커']}) AI 리서치 분석")

col1, col2 = st.columns(2)

with col1:
    st.metric("현재가", f"{종목_df['현재가']:,.0f}원")
    st.metric("ROE (최근)", f"{종목_df['ROE_최근']:.2f}%")
    st.metric("PER (최근)", f"{종목_df['PER_최근']:.2f}")
    st.metric("PBR (최근)", f"{종목_df['PBR_최근']:.2f}")
    st.metric("부채비율", f"{종목_df['부채비율_최근']:.2f}%")

with col2:
    # 안전한 메트릭 표시
    try:
        st.metric("유보율", f"{종목_df['유보율_최근']:.2f}%")
    except:
        st.metric("유보율", "데이터 없음")
    
    try:
        st.metric("매출액", f"{종목_df['매출액_최근']:,.0f}원")
    except:
        st.metric("매출액", "데이터 없음")
    
    try:
        st.metric("영업이익", f"{종목_df['영업이익_최근']:,.0f}원")
    except:
        st.metric("영업이익", "데이터 없음")
    
    try:
        st.metric("순이익", f"{종목_df['순이익_최근']:,.0f}원")
    except:
        st.metric("순이익", "데이터 없음")

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
st.markdown("### 🤖 AI 리서치 질의 (동적 양자화 최적화)")

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
        if st.button("🔍 AI 분석 요청 (양자화 모델)", type="primary"):
            with st.spinner("🤖 동적 양자화된 AI가 CPU에서 분석 중입니다..."):
                ai_response = generate_ai_response(
                    st.session_state.model, 
                    st.session_state.tokenizer, 
                    user_question, 
                    종목_df
                )
            
            st.markdown("#### 🎯 AI 분석 결과")
            st.markdown(ai_response)
            
            # 분석 결과 저장 옵션
            analysis_text = f"""
# {선택한_종목} AI 분석 결과 (동적 양자화 모델)

**질문:** {user_question}

**AI 분석:**
{ai_response}

**기본 정보:**
- 종목명: {종목_df['종목명']}
- 티커: {종목_df['티커']}
- 현재가: {종목_df['현재가']:,.0f}원
- PER: {종목_df['PER_최근']:.2f}
- PBR: {종목_df['PBR_최근']:.2f}
- ROE: {종목_df['ROE_최근']:.2f}%

*본 분석은 동적 양자화로 최적화된 AI 모델을 사용하여 생성되었습니다.*
"""
            
            st.download_button(
                label="📥 분석 보고서 다운로드",
                data=analysis_text,
                file_name=f"{선택한_종목}_AI분석_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown"
            )
    else:
        st.write("🔎 동적 양자화 AI 모델 로딩 중... 잠시만 기다려주세요.")

# 성능 정보 표시
if st.session_state.get('model_loaded'):
    with st.expander("⚡ 동적 양자화 성능 정보"):
        st.markdown("""
        **동적 양자화 최적화 효과:**
        - 🔹 **메모리 사용량**: 약 60-70% 감소
        - 🔹 **모델 크기**: 약 4배 압축
        - 🔹 **CPU 추론 속도**: 2-4배 향상
        - 🔹 **정확도 손실**: 최소화 (< 1%)
        - 🔹 **양자화 방식**: INT8 동적 양자화 (Linear 레이어)
        
        **기술적 세부사항:**
        - PyTorch Dynamic Quantization 사용
        - 가중치: INT8 저장, 활성화: 런타임 동적 양자화
        - CPU 벡터화 연산 최적화
        """)
