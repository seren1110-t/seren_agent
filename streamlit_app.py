# kospi_research_app.py
import os

# PyTorch와 Streamlit 호환성 문제 해결 (반드시 가장 먼저 실행)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# PyTorch 임포트 및 호환성 수정
import torch
try:
    # 방법 1: torch.classes 경로 수정
    torch.classes.__path__ = [os.path.join(torch.__path__[0], 'classes')]
except Exception:
    try:
        # 방법 2: 대안 경로 설정
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
from peft import PeftModel
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
        
        tokenizer = AutoTokenizer.from_pretrained(
            "./base_model",
            trust_remote_code=True,
            use_fast=False  # 호환성을 위해 slow tokenizer 사용
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 베이스 모델 로드 (안전한 설정)
        status_text.text("🧠 베이스 모델 로드 중... (4/6)")
        progress_bar.progress(60)
        
        base_model = AutoModelForCausalLM.from_pretrained(
            "./base_model",
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            use_cache=False,  # 캐시 비활성화로 메모리 절약
            torch_compile=False  # 컴파일 비활성화
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
        
        # 안전한 동적 양자화 적용
        try:
            with torch.no_grad():
                quantized_model = torch.quantization.quantize_dynamic(
                    model,
                    {torch.nn.Linear},
                    dtype=torch.qint8,
                    inplace=False  # 원본 모델 보존
                )
        except Exception as e:
            st.warning(f"동적 양자화 실패, 원본 모델 사용: {e}")
            quantized_model = model
        
        progress_bar.progress(100)
        
        # 메모리 사용량 계산
        def get_model_size(model):
            """모델 크기 계산 (MB)"""
            try:
                param_size = sum(p.numel() * p.element_size() for p in model.parameters())
                buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
                return (param_size + buffer_size) / 1024 / 1024
            except:
                return 0
        
        original_size = get_model_size(model)
        quantized_size = get_model_size(quantized_model)
        
        if original_size > 0 and quantized_size > 0:
            compression_ratio = original_size / quantized_size
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("원본 모델 크기", f"{original_size:.1f} MB")
            with col2:
                st.metric("양자화 모델 크기", f"{quantized_size:.1f} MB")
            with col3:
                st.metric("압축률", f"{compression_ratio:.1f}x")
        
        status_text.text("✅ 동적 양자화 모델 로드 완료!")
        st.success("⚡ 동적 양자화로 CPU 최적화 완료! 메모리 사용량이 크게 감소했습니다.")
        
        # 임시 파일 정리
        try:
            for temp_file in ["./my_base_model.tar.gz", "./qlora_results.zip"]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
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
    """안전한 AI 응답 생성"""
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
        # 안전한 토큰화
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            padding=True,
            add_special_tokens=True
        )
        
        # 안전한 추론
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,  # 토큰 수 줄여서 안정성 향상
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_beams=1,
                early_stopping=True,
                use_cache=False
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = response[len(prompt):].strip()
        
        return generated_text if generated_text else "응답을 생성할 수 없습니다."
        
    except Exception as e:
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
    st.info("🤖 AI 모델을 처음 로드합니다. 동적 양자화를 적용하여 CPU 최적화를 진행합니다...")
    model, tokenizer = download_and_load_models()
    st.session_state.model = model
    st.session_state.tokenizer = tokenizer
    st.session_state.model_loaded = True
    
    if model is not None:
        st.success("✅ 동적 양자화 AI 모델 로드 완료! 이제 CPU에서 효율적인 지능형 분석이 가능합니다.")
        st.rerun()

# 나머지 코드는 동일하게 유지...
