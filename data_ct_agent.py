# -*- coding: utf-8 -*-
"""
데이터 수집 에이전트 - 기존 뉴스/재무 데이터 수집 스크립트를 지능적으로 관리
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import torch
from langgraph.graph import StateGraph, END
from langchain_core.language_models import BaseLLM
from langchain_openai import ChatOpenAI

# 기존 모듈들 임포트 (원본 코드를 모듈화)
from news_collector import (
    scrape_news_until_cutoff_today, 
    clean_korean_text,
    HuggingFaceEmbeddings,
    FAISS,
    Document,
    RecursiveCharacterTextSplitter
)
from financial_collector import (
    get_kospi_tickers,
    collect_kospi_reports_async,
    collect_and_save_all
)

class CollectionStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class CollectionResult:
    status: CollectionStatus
    message: str
    data_count: int = 0
    file_path: Optional[str] = None
    error: Optional[str] = None

class DataCollectionState:
    """데이터 수집 상태 관리"""
    def __init__(self):
        self.news_result: Optional[CollectionResult] = None
        self.financial_result: Optional[CollectionResult] = None
        self.vector_result: Optional[CollectionResult] = None
        self.start_time: datetime = datetime.now()
        self.context: Dict[str, Any] = {}
        self.error_log: List[str] = []

class SmartDataCollectorAgent:
    """지능형 데이터 수집 에이전트"""
    
    def __init__(self, llm: BaseLLM = None):
        self.llm = llm or ChatOpenAI(model="gpt-4", temperature=0)
        self.state = DataCollectionState()
        
    def analyze_collection_context(self) -> Dict[str, Any]:
        """현재 수집 상황을 분석하여 최적 전략 결정"""
        
        context_prompt = f"""
        현재 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        다음 조건들을 분석하여 데이터 수집 전략을 결정해주세요:
        
        1. 파일 존재 여부:
           - financial_data.db: {os.path.exists('financial_data.db')}
           - bk_faiss_index/: {os.path.exists('bk_faiss_index')}
           - bk_docs.pkl: {os.path.exists('bk_docs.pkl')}
           
        2. 시장 상황 (현재 시간 기준):
           - 주식 시장 개장 시간: 09:00-15:30
           - 뉴스 수집 기준 시간: 09:00 이후
           
        3. 마지막 수집 시간 (파일 수정 시간 기준):
           - DB 파일: {self._get_file_age('financial_data.db')}
           - 벡터 DB: {self._get_file_age('bk_faiss_index')}
        
        응답 형식:
        {{
            "news_collection": "필요함/불필요함/조건부",
            "financial_collection": "필요함/불필요함/조건부", 
            "priority": "news/financial/both",
            "reason": "판단 근거",
            "recommended_schedule": "즉시/1시간후/내일"
        }}
        """
        
        try:
            response = self.llm.invoke(context_prompt)
            # 실제로는 응답을 파싱해야 하지만, 여기서는 기본 로직 사용
            return self._default_collection_strategy()
        except Exception as e:
            print(f"LLM 분석 실패, 기본 전략 사용: {e}")
            return self._default_collection_strategy()
    
    def _default_collection_strategy(self) -> Dict[str, Any]:
        """기본 수집 전략"""
        current_hour = datetime.now().hour
        
        # 기본 전략: 오전 9시 이후에는 모든 데이터 수집
        if current_hour >= 9:
            return {
                "news_collection": "필요함",
                "financial_collection": "필요함",
                "priority": "both",
                "reason": "정규 수집 시간",
                "recommended_schedule": "즉시"
            }
        else:
            return {
                "news_collection": "불필요함",
                "financial_collection": "불필요함", 
                "priority": "none",
                "reason": "시장 개장 전",
                "recommended_schedule": "9시 이후"
            }
    
    def _get_file_age(self, filepath: str) -> str:
        """파일의 나이를 반환"""
        if not os.path.exists(filepath):
            return "파일 없음"
        
        mtime = os.path.getmtime(filepath)
        file_time = datetime.fromtimestamp(mtime)
        age = datetime.now() - file_time
        
        if age.days > 0:
            return f"{age.days}일 전"
        elif age.seconds > 3600:
            return f"{age.seconds // 3600}시간 전"
        else:
            return f"{age.seconds // 60}분 전"

class NewsCollectionAgent:
    """뉴스 수집 전용 에이전트"""
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def execute(self, state: DataCollectionState) -> CollectionResult:
        """뉴스 데이터 수집 실행"""
        
        try:
            print("📰 뉴스 데이터 수집 시작...")
            
            # 1. 뉴스 크롤링 (기존 코드 사용)
            news_data = scrape_news_until_cutoff_today(cutoff_hour=9)
            
            if not news_data:
                return CollectionResult(
                    status=CollectionStatus.SKIPPED,
                    message="새로운 뉴스가 없습니다.",
                    data_count=0
                )
            
            # 2. 데이터 정제
            df = pd.DataFrame(news_data)
            df['제목'] = df['제목'].apply(clean_korean_text)
            df['본문'] = df['본문'].apply(clean_korean_text)
            df.dropna(subset=['제목', '본문'], inplace=True)
            df['내용'] = df['제목'] + '\n' + df['본문']
            df['내용'] = df['내용'].str.slice(0, 500)
            
            # 3. 벡터 DB 업데이트
            self._update_vector_db(df)
            
            return CollectionResult(
                status=CollectionStatus.COMPLETED,
                message="뉴스 수집 및 벡터 DB 업데이트 완료",
                data_count=len(df),
                file_path="bk_faiss_index"
            )
            
        except Exception as e:
            return CollectionResult(
                status=CollectionStatus.FAILED,
                message="뉴스 수집 실패",
                error=str(e)
            )
    
    def _update_vector_db(self, df):
        """벡터 DB 업데이트 (기존 로직)"""
        
        embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': self.device}
        )
        
        # 기존 문서 로드
        try:
            with open("bk_docs.pkl", "rb") as f:
                bk_docs = pickle.load(f)
            bk_faiss_db = FAISS.load_local("bk_faiss_index", embedding_model, 
                                         allow_dangerous_deserialization=True)
        except:
            bk_docs = []
            bk_faiss_db = None
        
        # 중복 제거 및 새 문서 추가
        existing_dates = set(doc.metadata.get("일자") for doc in bk_docs if "일자" in doc.metadata)
        
        new_docs = []
        for idx, row in df.iterrows():
            news_date = row["날짜"]
            if news_date in existing_dates:
                continue
            metadata = {"일자": news_date, "제목": row["제목"], "URL": row["URL"]}
            new_docs.append(Document(page_content=row["내용"], metadata=metadata))
        
        if new_docs:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=100)
            new_split_docs = text_splitter.split_documents(new_docs)
            bk_docs.extend(new_split_docs)
            
            if bk_faiss_db is None:
                bk_faiss_db = FAISS.from_documents(new_split_docs, embedding_model)
            else:
                bk_faiss_db.add_documents(new_split_docs)
            
            # 저장
            with open("bk_docs.pkl", "wb") as f:
                pickle.dump(bk_docs, f)
            bk_faiss_db.save_local("bk_faiss_index")

class FinancialCollectionAgent:
    """재무 데이터 수집 전용 에이전트"""
    
    def execute(self, state: DataCollectionState) -> CollectionResult:
        """재무 데이터 수집 실행"""
        
        try:
            print("💰 재무 데이터 수집 시작...")
            
            # 기존 collect_and_save_all() 함수 사용
            collect_and_save_all()
            
            # DB 파일 확인
            if os.path.exists("financial_data.db"):
                import sqlite3
                conn = sqlite3.connect("financial_data.db")
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM financial_data")
                count = cursor.fetchone()[0]
                conn.close()
                
                return CollectionResult(
                    status=CollectionStatus.COMPLETED,
                    message="재무 데이터 수집 완료",
                    data_count=count,
                    file_path="financial_data.db"
                )
            else:
                return CollectionResult(
                    status=CollectionStatus.FAILED,
                    message="DB 파일 생성 실패"
                )
                
        except Exception as e:
            return CollectionResult(
                status=CollectionStatus.FAILED,
                message="재무 데이터 수집 실패",
                error=str(e)
            )

# ============ LangGraph 워크플로우 정의 ============

def create_data_collection_workflow():
    """데이터 수집 워크플로우 생성"""
    
    # 에이전트 인스턴스 생성
    smart_agent = SmartDataCollectorAgent()
    news_agent = NewsCollectionAgent()
    financial_agent = FinancialCollectionAgent()
    
    def analyze_context_node(state: dict):
        """상황 분석 노드"""
        print("🔍 데이터 수집 상황 분석 중...")
        
        context = smart_agent.analyze_collection_context()
        state["collection_strategy"] = context
        state["analysis_complete"] = True
        
        print(f"📊 분석 결과: {context['reason']}")
        return state
    
    def collect_news_node(state: dict):
        """뉴스 수집 노드"""
        strategy = state.get("collection_strategy", {})
        
        if strategy.get("news_collection") == "필요함":
            result = news_agent.execute(smart_agent.state)
            state["news_result"] = result
            print(f"📰 뉴스 수집 결과: {result.message}")
        else:
            state["news_result"] = CollectionResult(
                status=CollectionStatus.SKIPPED,
                message="뉴스 수집 불필요"
            )
            print("📰 뉴스 수집 스킵")
        
        return state
    
    def collect_financial_node(state: dict):
        """재무 데이터 수집 노드"""
        strategy = state.get("collection_strategy", {})
        
        if strategy.get("financial_collection") == "필요함":
            result = financial_agent.execute(smart_agent.state)
            state["financial_result"] = result
            print(f"💰 재무 데이터 수집 결과: {result.message}")
        else:
            state["financial_result"] = CollectionResult(
                status=CollectionStatus.SKIPPED,
                message="재무 데이터 수집 불필요"
            )
            print("💰 재무 데이터 수집 스킵")
        
        return state
    
    def finalize_node(state: dict):
        """최종 정리 노드"""
        news_result = state.get("news_result")
        financial_result = state.get("financial_result")
        
        # 결과 요약
        summary = {
            "execution_time": datetime.now(),
            "news_status": news_result.status.value if news_result else "unknown",
            "financial_status": financial_result.status.value if financial_result else "unknown",
            "total_news_count": news_result.data_count if news_result else 0,
            "total_financial_count": financial_result.data_count if financial_result else 0,
            "generated_files": []
        }
        
        # 생성된 파일 확인
        expected_files = ["financial_data.db", "bk_faiss_index", "bk_docs.pkl"]
        for file in expected_files:
            if os.path.exists(file):
                summary["generated_files"].append(file)
        
        state["final_summary"] = summary
        
        print("✅ 데이터 수집 완료!")
        print(f"📊 최종 결과: {summary}")
        
        return state
    
    # 워크플로우 구성
    workflow = StateGraph(dict)
    
    # 노드 추가
    workflow.add_node("analyze_context", analyze_context_node)
    workflow.add_node("collect_news", collect_news_node)
    workflow.add_node("collect_financial", collect_financial_node)
    workflow.add_node("finalize", finalize_node)
    
    # 엣지 연결
    workflow.set_entry_point("analyze_context")
    workflow.add_edge("analyze_context", "collect_news")
    workflow.add_edge("collect_news", "collect_financial")
    workflow.add_edge("collect_financial", "finalize")
    workflow.add_edge("finalize", END)
    
    return workflow.compile()

# ============ 실행 함수 ============

def run_data_collection_agent():
    """데이터 수집 에이전트 실행"""
    
    print("🚀 데이터 수집 에이전트 시작!")
    print(f"⏰ 실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 워크플로우 실행
    app = create_data_collection_workflow()
    
    initial_state = {
        "start_time": datetime.now(),
        "execution_mode": "auto"
    }
    
    try:
        final_state = app.invoke(initial_state)
        
        # 실행 결과 리포트
        summary = final_state.get("final_summary", {})
        
        print("\n" + "="*50)
        print("📋 실행 결과 리포트")
        print("="*50)
        print(f"뉴스 수집: {summary.get('news_status', 'Unknown')}")
        print(f"재무 수집: {summary.get('financial_status', 'Unknown')}")
        print(f"뉴스 개수: {summary.get('total_news_count', 0)}")
        print(f"재무 데이터 개수: {summary.get('total_financial_count', 0)}")
        print(f"생성된 파일: {', '.join(summary.get('generated_files', []))}")
        print("="*50)
        
        return final_state
        
    except Exception as e:
        print(f"❌ 에이전트 실행 실패: {e}")
        return None

if __name__ == "__main__":
    # GitHub Actions나 cron에서 실행될 때
    run_data_collection_agent()
