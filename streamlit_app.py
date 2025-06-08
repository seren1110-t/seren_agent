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
import transformers
from transformers.models.auto import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import gc

st.set_page_config(page_title="📈 KOSPI Analyst AI", layout="wide")

def cleanup_memory():
    """메모리 정리 함수"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@st.cache_resource
def download_and_load_models():
    """Google Drive에서 모델 다운로드 및 CPU 최적화 모델 로드 (첫 번째 코드 저장 방식 기반)"""
    
    # Google Drive 파일 ID (첫 번째 코드에서 저장된 모델)
    saved_model_id = "1kQs4co-fO5JOTaAQ6Hn8S0s4fwUh6qyo"  # 실제 Google Drive ID로 변경 필요
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        cleanup_memory()
        
        # 저장된 모델 다운로드
        status_text.text("🔄 저장된 CPU 호환 모델 다운로드 중... (1/6)")
        progress_bar.progress(15)
        
        model_dir = "./koalpaca_streamlit_model"
        
        if not os.path.exists(model_dir):
            saved_model_url = f"https://drive.google.com/open?id={saved_model_id}"
            
            # .zip 형식으로 다운로드
            gdown.download(saved_model_url, "./koalpaca_streamlit_model.zip", quiet=False)
            
            # ZIP 파일 압축 해제
            with zipfile.ZipFile("./koalpaca_streamlit_model.zip", 'r') as zip_ref:
                zip_ref.extractall("./")
        
        # 설정 정보 로드
        status_text.text("🔧 모델 설정 정보 확인 중... (2/6)")
        progress_bar.progress(30)
        
        config_path = os.path.join(model_dir, "cpu_deployment_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config_info = json.load(f)
            st.info(f"✅ 모델 정보: {config_info.get('model_type', 'Unknown')}")
            st.info(f"📋 용도: {config_info.get('purpose', 'Unknown')}")
            st.info(f"🔧 최적화: CPU + {config_info.get('quantization_method', 'Unknown')}")
        else:
            st.warning("⚠️ 설정 파일을 찾을 수 없습니다.")
        
        # 토크나이저 로드
        status_text.text("📝 토크나이저 로드 중... (3/6)")
        progress_bar.progress(45)
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=True,
            use_fast=False,  # CPU 환경에서 안정성 우선
            padding_side="right"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # 베이스 모델 로드 (첫 번째 코드에서 저장된 방식)
        status_text.text("🧠 베이스 모델 로드 중 (CPU 최적화)... (4/6)")
        progress_bar.progress(60)
        
        base_model_path = os.path.join(model_dir, "base_model")
        
        if os.path.exists(base_model_path):
            # 베이스 모델 로드
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.float32,  # CPU에서는 float32 사용
                device_map="cpu",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                use_cache=False,  # 메모리 절약
                torch_compile=False  # CPU에서는 컴파일 비활성화
            )
        else:
            st.error(f"❌ 베이스 모델 디렉토리를 찾을 수 없습니다: {base_model_path}")
            return None, None
        
        cleanup_memory()
        
        # 모델을 평가 모드로 설정
        model.eval()
        
        # 저장된 양자화 모델이 있는지 확인
        status_text.text("⚡ 양자화 모델 확인 중... (5/6)")
        progress_bar.progress(75)
        
        quantized_model_path = os.path.join(model_dir, "cpu_quantized_model.pt")
        
        if os.path.exists(quantized_model_path):
            try:
                # 저장된 양자화 모델 로드
                checkpoint = torch.load(quantized_model_path, map_location='cpu')
                
                if 'quantization_info' in checkpoint:
                    quant_info = checkpoint['quantization_info']
                    st.success(f"✅ 저장된 양자화 모델 발견: {quant_info['method']}")
                    
                    # 양자화된 state_dict가 있으면 사용
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                        st.info("📦 저장된 양자화 가중치 적용 완료")
                
            except Exception as e:
                st.warning(f"저장된 양자화 모델 로드 실패, 동적 양자화 적용: {e}")
                
                # 동적 양자화 적용
                try:
                    with torch.no_grad():
                        quantized_model = torch.quantization.quantize_dynamic(
                            model,
                            {torch.nn.Linear},
                            dtype=torch.qint8,
                            inplace=False
                        )
                    model = quantized_model
                    st.success("✅ CPU 동적 양자화 적용 완료!")
                except Exception as qe:
                    st.warning(f"동적 양자화 실패, 원본 모델 사용: {qe}")
        else:
            # 양자화 모델이 없으면 동적 양자화 적용
            try:
                with torch.no_grad():
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
        
        # 모델 정보 표시
        status_text.text("📊 모델 정보 수집 중... (6/6)")
        progress_bar.progress(90)
        
        def get_model_size_safe(model):
            try:
                param_size = sum(p.numel() * p.element_size() for p in model.parameters())
                buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
                return (param_size + buffer_size) / 1024 / 1024
            except:
                return 0
        
        model_size = get_model_size_safe(model)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("모델 크기", f"{model_size:.1f} MB")
        with col2:
            st.metric("모델 타입", "KoAlpaca-Polyglot-5.8B")
        with col3:
            st.metric("최적화", "CPU + 양자화")
        
        progress_bar.progress(100)
        status_text.text("✅ CPU 최적화 모델 로드 완료!")
        st.success("⚡ CPU 최적화된 KoAlpaca 모델이 성공적으로 로드되었습니다!")
        
        # 임시 파일 정리
        try:
            temp_file = "./koalpaca_streamlit_model.zip"
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

def generate_response(model, tokenizer, prompt, max_new_tokens=512):
    """KoAlpaca 모델을 사용한 응답 생성"""
    try:
        # KoAlpaca 프롬프트 형식 적용
        formatted_prompt = f"### 질문: {prompt}\n\n### 답변:"
        
        # 토크나이징
        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            return_token_type_ids=False
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
                eos_token_id=2,
                use_cache=True,
                repetition_penalty=1.1,
            )
        
        # 디코딩 (입력 부분 제거)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
        ]
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()
        
    except Exception as e:
        st.error(f"❌ 응답 생성 실패: {e}")
        return "죄송합니다. 응답 생성 중 오류가 발생했습니다."

def create_sample_data():
    """샘플 KOSPI 데이터 생성"""
    companies = [
        "삼성전자", "SK하이닉스", "NAVER", "카카오", "LG화학",
        "삼성SDI", "현대차", "기아", "POSCO홀딩스", "KB금융"
    ]
    
    data = []
    for company in companies:
        data.append({
            "회사명": company,
            "현재가": np.random.randint(50000, 500000),
            "전일대비": np.random.randint(-10000, 10000),
            "등락률": round(np.random.uniform(-5.0, 5.0), 2),
            "거래량": np.random.randint(100000, 10000000),
            "시가총액": np.random.randint(1000000, 100000000)
        })
    
    return pd.DataFrame(data)

def load_financial_data():
    """금융 데이터 로드 (SQLite 데이터베이스 시뮬레이션)"""
    try:
        # 임시 SQLite 데이터베이스 생성
        conn = sqlite3.connect(':memory:')
        
        # 샘플 재무 데이터 생성
        financial_data = {
            '회사명': ['삼성전자', 'SK하이닉스', 'NAVER', '카카오', 'LG화학'],
            '매출액': [279000, 44000, 8800, 6800, 44000],
            '영업이익': [43000, 8900, 1400, 500, 3200],
            '당기순이익': [26900, 7300, 1200, 400, 2800],
            '부채비율': [15.2, 23.1, 8.9, 12.4, 67.8],
            'ROE': [9.8, 12.4, 8.7, 2.1, 7.9]
        }
        
        df = pd.DataFrame(financial_data)
        df.to_sql('financial_data', conn, index=False)
        
        # 데이터 조회
        result = pd.read_sql_query("SELECT * FROM financial_data", conn)
        conn.close()
        
        return result
    except Exception as e:
        st.error(f"재무 데이터 로드 실패: {e}")
        return pd.DataFrame()

def load_market_news():
    """시장 뉴스 데이터 로드"""
    news_data = [
        {
            "제목": "삼성전자, 3분기 실적 예상치 상회",
            "내용": "삼성전자가 3분기 영업이익이 시장 예상치를 상회했다고 발표했습니다.",
            "날짜": "2025-06-08",
            "카테고리": "실적"
        },
        {
            "제목": "SK하이닉스, AI 메모리 수요 급증",
            "내용": "AI 반도체 수요 증가로 SK하이닉스의 HBM 메모리 주문이 급증하고 있습니다.",
            "날짜": "2025-06-07",
            "카테고리": "산업동향"
        },
        {
            "제목": "KOSPI, 외국인 매수세로 상승",
            "내용": "외국인 투자자들의 매수세가 이어지며 KOSPI가 상승세를 보이고 있습니다.",
            "날짜": "2025-06-06",
            "카테고리": "시장동향"
        }
    ]
    
    return pd.DataFrame(news_data)

def main():
    st.title("📈 KOSPI Analyst AI")
    st.markdown("**KoAlpaca-Polyglot-5.8B 기반 한국 주식 분석 AI**")
    
    # 모델 로드
    with st.spinner("🔄 AI 모델 로딩 중..."):
        model, tokenizer = download_and_load_models()
    
    if model is None or tokenizer is None:
        st.error("❌ 모델 로드에 실패했습니다. 페이지를 새로고침해주세요.")
        return
    
    # 사이드바
    st.sidebar.header("📊 KOSPI 데이터")
    
    # 실시간 데이터 표시
    df = create_sample_data()
    st.sidebar.dataframe(df, height=300)
    
    # 재무 데이터 표시
    st.sidebar.header("💰 재무 데이터")
    financial_df = load_financial_data()
    if not financial_df.empty:
        st.sidebar.dataframe(financial_df, height=200)
    
    # 시장 뉴스 표시
    st.sidebar.header("📰 시장 뉴스")
    news_df = load_market_news()
    for _, news in news_df.iterrows():
        with st.sidebar.expander(f"📰 {news['제목'][:20]}..."):
            st.write(f"**날짜:** {news['날짜']}")
            st.write(f"**카테고리:** {news['카테고리']}")
            st.write(f"**내용:** {news['내용']}")
    
    # 메인 컨텐츠
    tab1, tab2, tab3, tab4 = st.tabs(["💬 AI 분석", "📈 차트 분석", "📋 보고서 생성", "📊 데이터 분석"])
    
    with tab1:
        st.header("💬 AI 주식 분석")
        
        # 질문 입력
        user_question = st.text_area(
            "📝 질문을 입력하세요:",
            placeholder="예: 삼성전자의 투자 전망을 분석해주세요.",
            height=100
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🤖 분석 시작", type="primary"):
                if user_question.strip():
                    with st.spinner("🧠 AI가 분석 중입니다..."):
                        response = generate_response(model, tokenizer, user_question)
                    
                    st.markdown("### 🤖 AI 분석 결과")
                    st.markdown(f"**질문:** {user_question}")
                    st.markdown(f"**답변:**\n{response}")
                else:
                    st.warning("⚠️ 질문을 입력해주세요.")
        
        with col2:
            max_tokens = st.slider("최대 토큰 수", 100, 1000, 512)
    
    with tab2:
        st.header("📈 차트 분석")
        
        # 샘플 차트 데이터
        chart_data = pd.DataFrame(
            np.random.randn(20, 3).cumsum(axis=0),
            columns=['KOSPI', 'KOSDAQ', 'KRX100']
        )
        
        st.line_chart(chart_data)
        
        # 개별 종목 차트
        st.subheader("📊 개별 종목 분석")
        selected_stock = st.selectbox("종목 선택", df["회사명"].tolist())
        
        # 선택된 종목의 가격 데이터 시뮬레이션
        dates = pd.date_range(start='2025-05-01', end='2025-06-08', freq='D')
        prices = np.random.randint(50000, 100000, len(dates))
        stock_data = pd.DataFrame({
            '날짜': dates,
            '주가': prices
        })
        stock_data.set_index('날짜', inplace=True)
        
        st.line_chart(stock_data)
    
    with tab3:
        st.header("📋 보고서 생성")
        
        selected_company = st.selectbox(
            "분석할 회사를 선택하세요:",
            df["회사명"].tolist()
        )
        
        report_type = st.radio(
            "보고서 유형:",
            ["투자 분석", "재무 분석", "기술적 분석", "종합 분석"]
        )
        
        if st.button("📄 보고서 생성", type="primary"):
            if report_type == "투자 분석":
                report_prompt = f"{selected_company}의 상세한 투자 분석 보고서를 작성해주세요. 재무상태, 성장성, 리스크 요인을 포함해서 분석해주세요."
            elif report_type == "재무 분석":
                report_prompt = f"{selected_company}의 재무 분석 보고서를 작성해주세요. 매출, 수익성, 안정성 지표를 중심으로 분석해주세요."
            elif report_type == "기술적 분석":
                report_prompt = f"{selected_company}의 기술적 분석 보고서를 작성해주세요. 차트 패턴, 거래량, 기술적 지표를 중심으로 분석해주세요."
            else:
                report_prompt = f"{selected_company}의 종합 분석 보고서를 작성해주세요. 투자, 재무, 기술적 분석을 모두 포함해서 작성해주세요."
            
            with st.spinner("📝 보고서 생성 중..."):
                report = generate_response(model, tokenizer, report_prompt, max_new_tokens=800)
            
            st.markdown(f"### 📋 {selected_company} {report_type} 보고서")
            st.markdown(report)
            
            # 다운로드 버튼
            st.download_button(
                label="📥 보고서 다운로드",
                data=f"{selected_company} {report_type} 보고서\n\n{report}",
                file_name=f"{selected_company}_{report_type}_report.txt",
                mime="text/plain"
            )
    
    with tab4:
        st.header("📊 데이터 분석")
        
        # 주식 데이터 분석
        st.subheader("📈 주식 데이터 현황")
        st.dataframe(df, use_container_width=True)
        
        # 통계 정보
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 시장 통계")
            st.metric("상승 종목", len(df[df['등락률'] > 0]))
            st.metric("하락 종목", len(df[df['등락률'] < 0]))
            st.metric("평균 등락률", f"{df['등락률'].mean():.2f}%")
        
        with col2:
            st.subheader("💰 재무 현황")
            if not financial_df.empty:
                st.dataframe(financial_df, use_container_width=True)
        
        # 시장 뉴스 분석
        st.subheader("📰 뉴스 분석")
        st.dataframe(news_df, use_container_width=True)
        
        # 데이터 시각화
        st.subheader("📊 데이터 시각화")
        
        # 등락률 분포
        fig_data = df['등락률'].values
        st.bar_chart(pd.DataFrame({'등락률': fig_data}, index=df['회사명']))
        
        # 거래량 분석
        st.subheader("📊 거래량 분석")
        volume_data = df[['회사명', '거래량']].set_index('회사명')
        st.bar_chart(volume_data)
    
    # 푸터
    st.markdown("---")
    st.markdown("**🤖 Powered by KoAlpaca-Polyglot-5.8B | CPU 최적화 버전**")
    st.markdown("**📊 실시간 KOSPI 데이터 분석 | 📋 AI 보고서 생성**")

if __name__ == "__main__":
    main()
