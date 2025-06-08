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

def fix_config_json(model_path):
    """config.json 파일의 model_type 키 수정"""
    config_path = os.path.join(model_path, "config.json")
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # model_type이 없으면 추가
            if 'model_type' not in config:
                # 첫 번째, 두 번째 코드에서 Llama 모델을 사용하므로
                config['model_type'] = 'llama'
                
                # 수정된 config.json 저장
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                
                st.info(f"✅ config.json에 model_type 추가: {config['model_type']}")
            else:
                st.info(f"✅ config.json에 model_type 확인: {config['model_type']}")
                
        except Exception as e:
            st.warning(f"config.json 수정 실패: {e}")
    else:
        st.error(f"config.json 파일이 없습니다: {config_path}")

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
        status_text.text("🔄 베이스 모델 다운로드 중... (1/9)")
        progress_bar.progress(10)
        
        if not os.path.exists("./base_model"):
            base_model_url = f"https://drive.google.com/uc?id={base_model_id}"
            gdown.download(base_model_url, "./my_base_model.tar.gz", quiet=False)
            
            with tarfile.open("./my_base_model.tar.gz", "r:gz") as tar:
                tar.extractall("./base_model/")
        
        # config.json 파일 수정 (model_type 추가)
        status_text.text("🔧 베이스 모델 설정 수정 중... (2/9)")
        progress_bar.progress(20)
        fix_config_json("./base_model")
        
        # QLoRA 어댑터 다운로드
        status_text.text("🔄 QLoRA 어댑터 다운로드 중... (3/9)")
        progress_bar.progress(25)
        
        if not os.path.exists("./qlora_adapter"):
            qlora_url = f"https://drive.google.com/uc?id={qlora_adapter_id}"
            gdown.download(qlora_url, "./qlora_results.zip", quiet=False)
            
            with zipfile.ZipFile("./qlora_results.zip", 'r') as zip_ref:
                zip_ref.extractall("./qlora_adapter/")
        
        # 체크포인트 경로 확인
        status_text.text("🔧 QLoRA 체크포인트 확인 중... (4/9)")
        progress_bar.progress(35)
        
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
        status_text.text("📝 토크나이저 로드 중... (5/9)")
        progress_bar.progress(45)
        
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
        status_text.text("🧠 베이스 모델 로드 중 (CPU 최적화)... (6/9)")
        progress_bar.progress(55)
        
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
        status_text.text("🔧 QLoRA 어댑터 설정 확인 중... (7/9)")
        progress_bar.progress(65)
        
        try:
            peft_config = PeftConfig.from_pretrained(adapter_path)
            st.info(f"✅ PEFT 설정: {peft_config.task_type}, target_modules={len(peft_config.target_modules)}개")
        except Exception as e:
            st.warning(f"PeftConfig 로드 실패: {e}")
        
        # QLoRA 어댑터 적용
        status_text.text(f"🔧 {checkpoint_name} 어댑터 적용 중... (8/9)")
        progress_bar.progress(75)
        
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
        status_text.text("⚡ CPU 동적 양자화 적용 중... (9/9)")
        progress_bar.progress(85)
        
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

# 나머지 코드는 이전과 동일...
