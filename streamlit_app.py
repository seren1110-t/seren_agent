# kospi_research_app.py
import streamlit as st
import pandas as pd
import sqlite3
import requests
from io import StringIO
import gdown
import os
import zipfile
import tarfile
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

st.set_page_config(page_title="📈 KOSPI Analyst AI", layout="wide")

# Google Drive 파일 다운로드 함수
@st.cache_resource
def download_and_load_models():
    """Google Drive에서 모델 다운로드 및 로드"""
    
    # Google Drive 파일 ID (공유 링크에서 추출)
    base_model_id = "YOUR_BASE_MODEL_FILE_ID"  # my_base_model.tar.gz의 파일 ID
    qlora_adapter_id = "YOUR_QLORA_ADAPTER_FILE_ID"  # qlora_results.zip의 파일 ID
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # 베이스 모델 다운로드
        status_text.text("🔄 베이스 모델 다운로드 중... (1/4)")
        progress_bar.progress(25)
        
        if not os.path.exists("./base_model"):
            base_model_url = f"https://drive.google.com/uc?id={base_model_id}"
            gdown.download(base_model_url, "./my_base_model.tar.gz", quiet=False)
            
            # 압축 해제
            with tarfile.open("./my_base_model.tar.gz", "r:gz") as tar:
                tar.extractall("./base_model/")
        
        # QLoRA 어댑터 다운로드
        status_text.text("🔄 QLoRA 어댑터 다운로드 중... (2/4)")
        progress_bar.progress(50)
        
        if not os.path.exists("./qlora_adapter"):
            qlora_url = f"https://drive.google.com/uc?id={qlora_adapter_id}"
            gdown.download(qlora_url, "./qlora_results.zip", quiet=False)
            
            # 압축 해제
            with zipfile.ZipFile("./qlora_results.zip", 'r') as zip_ref:
                zip_ref.extractall("./qlora_adapter/")
        
        # 토크나이저 로드
        status_text.text("📝 토크나이저 로드 중... (3/4)")
        progress_bar.progress(75)
        
        tokenizer = AutoTokenizer.from_pretrained("./base_model")
        
        # 모델 로드 및 어댑터 적용
        status_text.text("🧠 AI 모델 로드 중... (4/4)")
        progress_bar.progress(90)
        
        base_model = AutoModelForCausalLM.from_pretrained(
            "./base_model",
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        
        # QLoRA 어댑터 적용
        model = PeftModel.from_pretrained(base_model, "./qlora_adapter")
        
        progress_bar.progress(100)
        status_text.text("✅ 모델 로드 완료!")
        
        # UI 정리
        progress_bar.empty()
        status_text.empty()
        
        return model, tokenizer
        
    except Exception as e:
        st.error(f"❌ 모델 로드 실패: {e}")
        return None, None

@st.cache_data
def load_data(db_name="financial_data.db", table_name="financial_data"):
    """데이터베이스에서 데이터 로드"""
    try:
        conn = sqlite3.connect(db_name)
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return pd.DataFrame()

def generate_ai_response(model, tokenizer, question, company_data):
    """AI 모델을 사용한 응답 생성"""
    if model is None or tokenizer is None:
        return "AI 모델이 로드되지 않았습니다."
    
    # 회사 정보를 포함한 프롬프트 생성
    company_info = f"""
    종목명: {company_data['종목명']}
    티커: {company_data['티커']}
    현재가: {company_data['현재가']}
    PER: {company_data['PER_최근']}
    PBR: {company_data['PBR_최근']}
    ROE: {company_data['ROE_최근']}
    부채비율: {company_data['부채비율_최근']}
    """
    
    prompt = f"""다음은 {company_data['종목명']}의 재무 정보입니다:

{company_info}

질문: {question}

위 정보를 바탕으로 전문적인 증권 분석가 관점에서 답변해주세요:"""
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 입력 프롬프트 제거
        generated_text = response[len(prompt):].strip()
        
        return generated_text
        
    except Exception as e:
        return f"응답 생성 중 오류가 발생했습니다: {e}"

def get_initial(korean_char):
    """한글 초성 추출"""
    ch_code = ord(korean_char) - ord('가')
    if 0 <= ch_code < 11172:
        cho = ch_code // 588
        return ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ'][cho]
    return ""

# 메인 앱
def main():
    # URL 파라미터 읽기
    url_params = st.experimental_get_query_params()
    
    # AI 모델 로드
    if 'model_loaded' not in st.session_state:
        st.info("🤖 AI 모델을 처음 로드합니다. 잠시만 기다려주세요...")
        model, tokenizer = download_and_load_models()
        st.session_state.model = model
        st.session_state.tokenizer = tokenizer
        st.session_state.model_loaded = True
        
        if model is not None:
            st.success("✅ AI 모델 로드 완료! 이제 지능형 분석이 가능합니다.")
        st.rerun()
    
    # 데이터 로드
    df = load_data()
    
    if df.empty:
        st.error("❌ 데이터를 로드할 수 없습니다.")
        return
    
    # 사이드바 필터
    st.sidebar.header("📂 필터 옵션")
    
    # URL 파라미터에서 기본값 가져오기
    default_initial = url_params.get("initial", ["전체"])[0]
    default_search = url_params.get("search", [""])[0]
    default_company = url_params.get("company", [""])[0]
    
    # 초성 필터
    initials = ['전체', 'ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    selected_initial = st.sidebar.selectbox("🔡 종목명 초성:", initials, 
                                          index=initials.index(default_initial) if default_initial in initials else 0)
    
    # 텍스트 검색
    search_term = st.sidebar.text_input("🔍 종목명 또는 티커 검색", value=default_search)
    
    # 고급 검색 옵션
    with st.sidebar.expander("🔧 고급 검색 옵션"):
        # PER 범위 필터
        per_range = st.slider("PER 범위", 0.0, 50.0, (0.0, 50.0))
        
        # PBR 범위 필터
        pbr_range = st.slider("PBR 범위", 0.0, 10.0, (0.0, 10.0))
        
        # 시가총액 범위 (현재가 기준)
        price_range = st.slider("주가 범위", 
                               int(df["현재가"].min()), 
                               int(df["현재가"].max()), 
                               (int(df["현재가"].min()), int(df["현재가"].max())))
    
    # 데이터 필터링
    filtered_df = df.copy()
    
    # 초성 필터 적용
    if selected_initial != "전체":
        filtered_df = filtered_df[filtered_df["종목명"].apply(lambda x: get_initial(x[0]) == selected_initial)]
    
    # 텍스트 검색 적용
    if search_term:
        mask1 = filtered_df["종목명"].str.contains(search_term, case=False, na=False)
        mask2 = filtered_df["티커"].str.contains(search_term, case=False, na=False)
        filtered_df = filtered_df[mask1 | mask2]
    
    # 고급 필터 적용
    filtered_df = filtered_df[
        (filtered_df["PER_최근"].between(per_range[0], per_range[1])) &
        (filtered_df["PBR_최근"].between(pbr_range[0], pbr_range[1])) &
        (filtered_df["현재가"].between(price_range[0], price_range[1]))
    ]
    
    종목_list = filtered_df["종목명"].tolist()
    
    if not 종목_list:
        st.warning("❌ 조건에 맞는 종목이 없습니다.")
        return
    
    # 종목 선택
    if default_company and default_company in 종목_list:
        default_index = 종목_list.index(default_company)
    else:
        default_index = 0
    
    선택한_종목 = st.sidebar.selectbox("📌 종목 선택:", 종목_list, index=default_index)
    종목_df = filtered_df[filtered_df["종목명"] == 선택한_종목].iloc[0]
    
    # URL 공유 링크 생성
    share_url = f"?initial={selected_initial}&search={search_term}&company={선택한_종목}"
    st.sidebar.markdown(f"🔗 [이 분석 공유하기]({share_url})")
    
    # ----------------------- 메인 컨텐츠 -----------------------
    st.title(f"📊 {선택한_종목} ({종목_df['티커']}) AI 리서치 분석")
    
    # 검색 결과 요약
    if len(종목_list) < len(df):
        st.info(f"🔍 검색 결과: {len(종목_list)}개 종목 (전체 {len(df)}개 중)")
    
    # 재무 지표
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("현재가", f"{종목_df['현재가']:,}원")
        st.metric("ROE (최근)", f"{종목_df['ROE_최근']:.2f}%")
        st.metric("PER (최근)", f"{종목_df['PER_최근']:.2f}")
        st.metric("PBR (최근)", f"{종목_df['PBR_최근']:.2f}")
        st.metric("부채비율", f"{종목_df['부채비율_최근']:.2f}%")
    
    with col2:
        st.metric("유보율", f"{종목_df['유보율_최근']:.2f}%")
        st.metric("매출액", f"{종목_df['매출액_최근']:,}원")
        st.metric("영업이익", f"{종목_df['영업이익_최근']:,}원")
        st.metric("순이익", f"{종목_df['순이익_최근']:,}원")
    
    # 주가 차트
    st.markdown("### 📈 주가 추이")
    price_cols = [col for col in df.columns if col.isdigit() and len(col) == 8]
    if price_cols:
        price_series = 종목_df[price_cols].astype(float)
        price_series.index = pd.to_datetime(price_cols, format='%Y%m%d')
        chart_df = price_series.reset_index().rename(columns={'index': '날짜'})
        st.line_chart(chart_df.set_index("날짜"))
    
    # 뉴스 섹션
    st.markdown("### 📰 최근 뉴스")
    if isinstance(종목_df["최신뉴스"], str) and 종목_df["최신뉴스"].strip():
        for i, link in enumerate(종목_df["최신뉴스"].splitlines(), 1):
            if link.strip():
                st.markdown(f"{i}. [뉴스 링크]({link.strip()})")
    else:
        st.info("최근 뉴스가 없습니다.")
    
    # AI 분석 섹션
    st.markdown("### 🤖 AI 증권 분석")
    
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
        user_question = st.text_input("💬 궁금한 점을 입력하세요:", 
                                    placeholder="예: 이 종목의 PER이 높으면 어떤 해석이 가능해?")
    
    with col2:
        selected_preset = st.selectbox("📋 미리 정의된 질문:", ["직접 입력"] + preset_questions)
    
    if selected_preset != "직접 입력":
        user_question = selected_preset
    
    if user_question and st.button("🔍 AI 분석 요청", type="primary"):
        if st.session_state.get('model') is not None:
            with st.spinner("🤖 AI가 분석 중입니다..."):
                ai_response = generate_ai_response(
                    st.session_state.model, 
                    st.session_state.tokenizer, 
                    user_question, 
                    종목_df
                )
            
            st.markdown("#### 🎯 AI 분석 결과")
            st.markdown(ai_response)
            
            # 분석 결과 저장 옵션
            if st.button("💾 분석 결과 저장"):
                analysis_text = f"""
# {선택한_종목} AI 분석 결과

**질문:** {user_question}

**AI 분석:**
{ai_response}

**기본 정보:**
- 종목명: {종목_df['종목명']}
- 티커: {종목_df['티커']}
- 현재가: {종목_df['현재가']:,}원
- PER: {종목_df['PER_최근']:.2f}
- PBR: {종목_df['PBR_최근']:.2f}
- ROE: {종목_df['ROE_최근']:.2f}%
"""
                st.download_button(
                    label="📥 분석 보고서 다운로드",
                    data=analysis_text,
                    file_name=f"{선택한_종목}_AI분석_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.md",
                    mime="text/markdown"
                )
        else:
            st.error("❌ AI 모델이 로드되지 않았습니다. 페이지를 새로고침해주세요.")
    
    # 검색 통계
    with st.expander("📊 검색 통계"):
        st.write(f"**전체 종목 수:** {len(df)}")
        st.write(f"**필터링된 종목 수:** {len(filtered_df)}")
        st.write(f"**평균 PER:** {filtered_df['PER_최근'].mean():.2f}")
        st.write(f"**평균 PBR:** {filtered_df['PBR_최근'].mean():.2f}")
        st.write(f"**평균 ROE:** {filtered_df['ROE_최근'].mean():.2f}%")

if __name__ == "__main__":
    main()
