# -*- coding: utf-8 -*-
"""
데이터 수집 에이전트 - 수정된 버전
"""

import asyncio
import os
import sys
import pandas as pd  # 추가
import pickle       # 추가
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import torch
from langgraph.graph import StateGraph, END
from langchain_core.language_models import BaseLLM
from langchain_openai import ChatOpenAI

# 기존 모듈들 임포트
from news_collector import NewsCollector, VectorDBManager, clean_korean_text
from financial_collector import FinancialDataManager

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
        self.llm = llm or ChatOpenAI(model="gpt-4", temperature=0) if llm else None
        self.state = DataCollectionState()
        
    def analyze_collection_context(self) -> Dict[str, Any]:
        """현재 수집 상황을 분석하여 최적 전략 결정"""
        
        # LLM이 없는 경우 기본 전략 사용
        if not self.llm:
            return self._default_collection_strategy()
            
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
        """기본 수집 전략 - 수정됨"""
        current_hour = datetime.now().hour
        
        # 더 관대한 수집 정책: 하루 종일 수집 가능
        return {
            "news_collection": "필요함",
            "financial_collection": "필요함",
            "priority": "both",
            "reason": f"현재 시간 {current_hour}시, 데이터 수집 실행",
            "recommended_schedule": "즉시"
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
    """뉴스 수집 전용 에이전트 - 수정됨"""
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.collector = NewsCollector()  # 인스턴스 생성
        self.db_manager = VectorDBManager()  # 인스턴스 생성
        
    def execute(self, state: DataCollectionState) -> CollectionResult:
        """뉴스 데이터 수집 실행 - 수정됨"""
        
        try:
            print("📰 뉴스 데이터 수집 시작...")
            
            # 1. 뉴스 크롤링 (수정된 호출 방식)
            news_data = self.collector.scrape_news_until_cutoff_today(cutoff_hour=9)
            
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
            
            print(f"정제된 뉴스 데이터: {len(df)}개")
            
            # 3. 벡터 DB 업데이트 (수정된 방식)
            success = self._update_vector_db(df.to_dict('records'))
            
            if success:
                return CollectionResult(
                    status=CollectionStatus.COMPLETED,
                    message="뉴스 수집 및 벡터 DB 업데이트 완료",
                    data_count=len(df),
                    file_path="bk_faiss_index"
                )
            else:
                return CollectionResult(
                    status=CollectionStatus.COMPLETED,
                    message="뉴스 수집 완료 (중복 제거로 인해 벡터 DB 업데이트 없음)",
                    data_count=len(df),
                    file_path=None
                )
            
        except Exception as e:
            print(f"뉴스 수집 오류: {e}")
            return CollectionResult(
                status=CollectionStatus.FAILED,
                message="뉴스 수집 실패",
                error=str(e)
            )
    
    def _update_vector_db(self, news_data):
        """벡터 DB 업데이트 - 수정됨"""
        try:
            # 임베딩 모델 초기화
            self.db_manager.initialize_embedding_model()
            
            # 기존 데이터 로드
            self.db_manager.load_existing_data()
            
            # 새 데이터 추가
            success = self.db_manager.add_news_to_vector_db(news_data)
            
            if success:
                # 데이터 저장
                self.db_manager.save_data()
                print("✅ 벡터 DB 업데이트 및 저장 완료")
                return True
            else:
                print("ℹ️  새로 추가할 뉴스가 없음 (중복 제거)")
                return False
                
        except Exception as e:
            print(f"❌ 벡터 DB 업데이트 실패: {e}")
            return False

class FinancialCollectionAgent:
    """재무 데이터 수집 전용 에이전트 - 수정됨"""
    
    def __init__(self):
        self.manager = FinancialDataManager()  # 인스턴스 생성
    
    def execute(self, state: DataCollectionState) -> CollectionResult:
        """재무 데이터 수집 실행 - 수정됨"""
        
        try:
            print("💰 재무 데이터 수집 시작...")
            
            # 수정된 호출 방식
            result_df = self.manager.collect_and_save_all(limit=None)  # 테스트를 위해 50개로 제한
            
            # DB 파일 확인
            if os.path.exists("financial_data.db"):
                import sqlite3
                conn = sqlite3.connect("financial_data.db")
                cursor = conn.cursor()
                
                # 테이블 존재 확인
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='financial_data'")
                if cursor.fetchone():
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
                    conn.close()
                    return CollectionResult(
                        status=CollectionStatus.FAILED,
                        message="DB 테이블 생성 실패"
                    )
            else:
                return CollectionResult(
                    status=CollectionStatus.FAILED,
                    message="DB 파일 생성 실패"
                )
                
        except Exception as e:
            print(f"재무 데이터 수집 오류: {e}")
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
            "execution_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "news_status": news_result.status.value if news_result else "unknown",
            "financial_status": financial_result.status.value if financial_result else "unknown",
            "total_news_count": news_result.data_count if news_result else 0,
            "total_financial_count": financial_result.data_count if financial_result else 0,
            "generated_files": [],
            "errors": []
        }
        
        # 에러 수집
        if news_result and news_result.error:
            summary["errors"].append(f"뉴스: {news_result.error}")
        if financial_result and financial_result.error:
            summary["errors"].append(f"재무: {financial_result.error}")
        
        # 생성된 파일 확인
        expected_files = ["financial_data.db", "bk_faiss_index", "bk_docs.pkl"]
        for file in expected_files:
            if os.path.exists(file):
                file_size = os.path.getsize(file) if os.path.isfile(file) else "디렉토리"
                summary["generated_files"].append(f"{file} ({file_size} bytes)" if file_size != "디렉토리" else f"{file} (디렉토리)")
        
        state["final_summary"] = summary
        
        print("\n" + "="*60)
        print("✅ 데이터 수집 완료!")
        print("="*60)
        print(f"📊 실행 시간: {summary['execution_time']}")
        print(f"📰 뉴스 상태: {summary['news_status']} ({summary['total_news_count']}개)")
        print(f"💰 재무 상태: {summary['financial_status']} ({summary['total_financial_count']}개)")
        print(f"📁 생성 파일: {', '.join(summary['generated_files']) if summary['generated_files'] else '없음'}")
        if summary['errors']:
            print(f"❌ 오류: {', '.join(summary['errors'])}")
        print("="*60)
        
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
    print(f"💻 작업 디렉토리: {os.getcwd()}")
    
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
        
        print("\n" + "🎯 최종 실행 결과")
        print(f"성공 여부: {'성공' if not summary.get('errors') else '부분 성공'}")
        
        return final_state
        
    except Exception as e:
        print(f"❌ 에이전트 실행 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

# 간단한 테스트 함수
def test_individual_components():
    """개별 컴포넌트 테스트"""
    print("🧪 개별 컴포넌트 테스트 시작")
    
    # 1. 뉴스 수집 테스트
    try:
        print("📰 뉴스 수집 테스트...")
        news_agent = NewsCollectionAgent()
        state = DataCollectionState()
        result = news_agent.execute(state)
        print(f"뉴스 결과: {result.status.value} - {result.message}")
    except Exception as e:
        print(f"뉴스 테스트 실패: {e}")
    
    # 2. 재무 데이터 수집 테스트
    try:
        print("💰 재무 데이터 수집 테스트...")
        financial_agent = FinancialCollectionAgent()
        state = DataCollectionState()
        result = financial_agent.execute(state)
        print(f"재무 결과: {result.status.value} - {result.message}")
    except Exception as e:
        print(f"재무 테스트 실패: {e}")

if __name__ == "__main__":
    # 개별 테스트 실행 (디버깅용)
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_individual_components()
    else:
        # 전체 에이전트 실행
        run_data_collection_agent()
