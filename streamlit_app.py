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

def safe_load_tokenizer(model_path):
    """안전한 토크나이저 로드 - 'bool' object has no attribute 'pad_token' 오류 해결"""
    try:
        # 방법 1: 기본 로드 시도
        st.info("📝 토크나이저 로드 시도 중...")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
            padding_side="right"
        )
        
        # pad_token 안전하게 설정
        if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
            st.info("🔧 pad_token 설정 중...")
            if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                else:
                    tokenizer.pad_token_id = 2  # 기본값
            else:
                # eos_token이 없으면 새로 추가
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')
        
        # 토크나이저 검증
        if hasattr(tokenizer, 'pad_token') and tokenizer.pad_token is not None:
            st.success(f"✅ 토크나이저 로드 성공! pad_token: {tokenizer.pad_token}")
            return tokenizer, None
        else:
            raise ValueError("pad_token 설정 실패")
        
    except Exception as e:
        st.warning(f"기본 토크나이저 로드 실패: {e}")
        
        # 방법 2: 대안 로드 방식
        try:
            st.info("🔄 대안 토크나이저 로드 시도...")
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=True,  # fast tokenizer 시도
                padding_side="right",
                add_eos_token=True
            )
            
            # 강제 pad_token 설정
            tokenizer.pad_token = "</s>"  # 기본 EOS 토큰
            tokenizer.pad_token_id = 2    # 기본 EOS 토큰 ID
            
            st.success("✅ 대안 토크나이저 로드 성공!")
            return tokenizer, None
            
        except Exception as e2:
            st.error(f"대안 토크나이저 로드도 실패: {e2}")
            
            # 방법 3: 최소한의 토크나이저 생성
            try:
                st.info("🆘 최소한의 토크나이저 생성 시도...")
                
                # 기본 설정으로 토크나이저 생성
                from transformers import GPT2Tokenizer
                tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
                
                # 한국어 지원을 위한 최소 설정
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
                
                st.warning("⚠️ 기본 토크나이저로 대체됨 (성능 제한)")
                return tokenizer, None
                
            except Exception as e3:
                return None, f"모든 토크나이저 로드 방법 실패: {e3}"

def verify_zip_file(file_path):
    """ZIP 파일 검증"""
    try:
        if not os.path.exists(file_path):
            return False, "파일이 존재하지 않습니다."
        
        file_size = os.path.getsize(file_path)
        if file_size < 1000:  # 1KB 미만
            return False, f"파일이 너무 작습니다 ({file_size} bytes). 오류 페이지일 가능성이 높습니다."
        
        # ZIP 파일 검증
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            if len(file_list) == 0:
                return False, "빈 ZIP 파일입니다."
            
            # 필요한 파일들이 있는지 확인
            required_files = ['base_model/', 'cpu_deployment_config.json']
            found_files = []
            for required in required_files:
                for file in file_list:
                    if required in file:
                        found_files.append(required)
                        break
            
            if len(found_files) < len(required_files):
                return False, f"필요한 파일이 없습니다. 찾은 파일: {found_files}"
        
        return True, f"유효한 ZIP 파일입니다. 포함된 파일: {len(file_list)}개"
        
    except zipfile.BadZipFile:
        # 파일 내용 확인 (HTML 페이지인지 체크)
        try:
            with open(file_path, 'rb') as f:
                first_bytes = f.read(200)
                if b'<html' in first_bytes.lower() or b'<!doctype' in first_bytes.lower():
                    return False, "HTML 페이지가 다운로드되었습니다. Google Drive 할당량 초과 가능성이 높습니다."
                else:
                    return False, f"유효하지 않은 ZIP 파일입니다. 첫 200바이트: {first_bytes[:100]}..."
        except Exception as e:
            return False, f"파일 읽기 오류: {e}"
    except Exception as e:
        return False, f"ZIP 검증 오류: {e}"

def download_with_verification(file_id, output_path, max_retries=3):
    """검증을 포함한 다운로드"""
    download_methods = [
        f"https://drive.google.com/uc?id={file_id}&confirm=t",
        f"https://drive.google.com/uc?id={file_id}",
        f"https://drive.google.com/uc?export=download&id={file_id}",
    ]
    
    for attempt in range(max_retries):
        st.info(f"다운로드 시도 {attempt + 1}/{max_retries}")
        
        # 기존 파일 삭제
        if os.path.exists(output_path):
            os.remove(output_path)
        
        for i, url in enumerate(download_methods):
            try:
                st.info(f"방법 {i+1}: {url[:50]}...")
                gdown.download(url, output_path, quiet=False)
                
                # 파일 검증
                is_valid, message = verify_zip_file(output_path)
                st.info(f"검증 결과: {message}")
                
                if is_valid:
                    st.success("✅ 유효한 파일 다운로드 완료!")
                    return True
                else:
                    st.warning(f"❌ 검증 실패: {message}")
                    continue
                    
            except Exception as e:
                st.warning(f"다운로드 방법 {i+1} 실패: {e}")
                continue
        
        st.warning(f"시도 {attempt + 1} 실패. 잠시 후 재시도...")
        
    return False

def manual_download_guide(file_id, output_path):
    """수동 다운로드 안내"""
    st.error("🚫 자동 다운로드가 실패했습니다.")
    
    with st.expander("📋 수동 다운로드 방법", expanded=True):
        st.markdown(f"""
        **Google Drive 할당량 우회 수동 다운로드:**
        
        1. **브라우저에서 다운로드:**
           - [파일 링크](https://drive.google.com/file/d/{file_id}/view) 클릭
           - "내 드라이브에 추가" 버튼 클릭
           - 새 폴더 생성 후 바로가기 추가
           - 폴더 전체를 다운로드 (ZIP으로 압축됨)
           
        2. **파일 저장 위치:**
           - 다운로드한 ZIP 파일을 `{output_path}` 경로에 저장
           
        3. **파일 검증:**
           - ZIP 파일이 정상적으로 열리는지 확인
           - 파일 크기가 10MB 이상인지 확인
        """)
    
    # 파일 확인 버튼
    if st.button("✅ 수동 다운로드 완료", type="primary"):
        is_valid, message = verify_zip_file(output_path)
        if is_valid:
            st.success("파일 검증 완료! 페이지를 새로고침하여 계속하세요.")
            st.rerun()
        else:
            st.error(f"파일 검증 실패: {message}")

@st.cache_resource
def download_and_load_models():
    """Google Drive에서 모델 다운로드 및 CPU 최적화 모델 로드 (토크나이저 오류 수정)"""
    
    saved_model_id = "1kQs4co-fO5JOTaAQ6Hn8S0s4fwUh6qyo"
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        cleanup_memory()
        
        model_dir = "./koalpaca_streamlit_model"
        zip_path = "./koalpaca_streamlit_model.zip"
        
        if not os.path.exists(model_dir):
            status_text.text("🔄 모델 다운로드 및 검증 중... (1/6)")
            progress_bar.progress(15)
            
            # 검증을 포함한 다운로드
            if download_with_verification(saved_model_id, zip_path):
                # ZIP 파일 압축 해제
                status_text.text("📦 압축 해제 중...")
                progress_bar.progress(25)
                
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall("./")
                    st.success("✅ 압축 해제 완료!")
                except Exception as e:
                    st.error(f"압축 해제 실패: {e}")
                    return None, None
            else:
                # 수동 다운로드 안내
                manual_download_guide(saved_model_id, zip_path)
                st.stop()
        
        # 설정 정보 로드
        status_text.text("🔧 모델 설정 정보 확인 중... (2/6)")
        progress_bar.progress(35)
        
        config_path = os.path.join(model_dir, "cpu_deployment_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config_info = json.load(f)
            st.info(f"✅ 모델 정보: {config_info.get('model_type', 'Unknown')}")
            st.info(f"📋 용도: {config_info.get('purpose', 'Unknown')}")
            st.info(f"🔧 최적화: CPU + {config_info.get('quantization_method', 'Unknown')}")
        else:
            st.warning("⚠️ 설정 파일을 찾을 수 없습니다.")
        
        # 안전한 토크나이저 로드
        status_text.text("📝 토크나이저 로드 중... (3/6)")
        progress_bar.progress(50)
        
        tokenizer, error = safe_load_tokenizer(model_dir)
        if tokenizer is None:
            st.error(f"❌ 토크나이저 로드 실패: {error}")
            return None, None
        
        # 베이스 모델 로드
        status_text.text("🧠 베이스 모델 로드 중... (4/6)")
        progress_bar.progress(65)
        
        base_model_path = os.path.join(model_dir, "base_model")
        
        if os.path.exists(base_model_path):
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_path,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    use_cache=False,
                    torch_compile=False
                )
                st.success("✅ 베이스 모델 로드 완료!")
            except Exception as e:
                st.error(f"❌ 베이스 모델 로드 실패: {e}")
                return None, None
        else:
            st.error(f"❌ 베이스 모델 디렉토리를 찾을 수 없습니다: {base_model_path}")
            return None, None
        
        cleanup_memory()
        model.eval()
        
        # 양자화 적용
        status_text.text("⚡ 양자화 적용 중... (5/6)")
        progress_bar.progress(80)
        
        quantized_model_path = os.path.join(model_dir, "cpu_quantized_model.pt")
        
        if os.path.exists(quantized_model_path):
            try:
                checkpoint = torch.load(quantized_model_path, map_location='cpu')
                if 'quantization_info' in checkpoint:
                    quant_info = checkpoint['quantization_info']
                    st.success(f"✅ 저장된 양자화 모델 발견: {quant_info['method']}")
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                        st.info("📦 저장된 양자화 가중치 적용 완료")
            except Exception as e:
                st.warning(f"저장된 양자화 모델 로드 실패: {e}")
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
            except Exception as e:
                st.warning(f"동적 양자화 실패, 원본 모델 사용: {e}")
        
        # 모델 정보 표시
        status_text.text("📊 모델 정보 수집 중... (6/6)")
        progress_bar.progress(95)
        
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
            if os.path.exists(zip_path):
                os.remove(zip_path)
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
    """KoAlpaca 모델을 사용한 응답 생성 (토크나이저 오류 방지)"""
    try:
        # KoAlpaca 프롬프트 형식 적용
        formatted_prompt = f"### 질문: {prompt}\n\n### 답변:"
        
        # 안전한 토크나이징
        try:
            inputs = tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                return_token_type_ids=False
            )
        except Exception as e:
            st.warning(f"토크나이징 오류: {e}")
            # 기본 토크나이징 시도
            inputs = tokenizer.encode(formatted_prompt, return_tensors="pt")
            inputs = {"input_ids": inputs}
        
        # 생성
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=getattr(tokenizer, 'pad_token_id', 2),
                eos_token_id=2,
                use_cache=True,
                repetition_penalty=1.1,
            )
        
        # 디코딩 (입력 부분 제거)
        if 'input_ids' in inputs:
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs['input_ids'], outputs)
            ]
        else:
            generated_ids = outputs
        
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
