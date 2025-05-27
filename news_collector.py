# -*- coding: utf-8 -*-
"""
뉴스 수집 및 벡터 데이터베이스 관리 모듈
"""
import requests
from lxml import html
from datetime import datetime
import time
import pandas as pd
import re
import os
import pickle
import torch

from dotenv import load_dotenv
load_dotenv()

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 상수 정의
HEADERS = {"User-Agent": "Mozilla/5.0"}
DEFAULT_CUTOFF_HOUR = 9
DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_TEXT_LIMIT = 500

class NewsCollector:
    """뉴스 수집을 위한 클래스"""
    
    def __init__(self, headers=None):
        self.headers = headers or HEADERS
    
    def get_last_page(self, date_str):
        """특정 날짜의 마지막 페이지 번호를 가져옴"""
        url = f"https://finance.naver.com/news/mainnews.naver?date={date_str}"
        try:
            response = requests.get(url, headers=self.headers, timeout=5)
            response.encoding = 'euc-kr'
            tree = html.fromstring(response.text)
            last_page_link = tree.xpath('//td[@class="pgRR"]/a')
            if last_page_link:
                match = re.search(r'page=(\d+)', last_page_link[0].get('href'))
                if match:
                    return int(match.group(1))
        except Exception as e:
            print(f"마지막 페이지 조회 실패: {e}")
        return 1

    def get_news_list_on_page(self, date_str, page):
        """특정 날짜의 특정 페이지에서 뉴스 목록을 가져옴"""
        url = f"https://finance.naver.com/news/mainnews.naver?date={date_str}&page={page}"
        try:
            response = requests.get(url, headers=self.headers, timeout=5)
            response.encoding = 'euc-kr'
            tree = html.fromstring(response.text)
            news_items = tree.xpath('//*[@id="contentarea_left"]/div[2]/ul/li')
        except Exception as e:
            print(f"뉴스 목록 조회 실패 (페이지 {page}): {e}")
            return []

        news_list = []
        for item in news_items:
            try:
                title_a = item.xpath('./dl/dd[1]/a')
                wdate = item.xpath('.//span[@class="wdate"]/text()')
                if not title_a or not wdate:
                    continue
                title = title_a[0].text.strip()
                href = "https://finance.naver.com" + title_a[0].get('href')
                news_datetime = datetime.strptime(wdate[0], "%Y-%m-%d %H:%M:%S")
                news_list.append({
                    "title": title,
                    "url": href,
                    "datetime": news_datetime
                })
            except Exception as e:
                print(f"뉴스 항목 파싱 실패: {e}")
                continue
        return news_list

    def get_news_body(self, news_url):
        """뉴스 본문을 가져옴"""
        try:
            response = requests.get(news_url, headers=self.headers, timeout=5)
            response.encoding = 'euc-kr'
            if "top.location.href" in response.text:
                redirected_url = re.search(r"top\.location\.href='(.*?)'", response.text)
                if redirected_url:
                    news_url = redirected_url.group(1)
                    response = requests.get(news_url, headers=self.headers, timeout=5)
                    response.encoding = 'utf-8'
            tree = html.fromstring(response.text)
            xpath_id = tree.xpath('//*[@id="newsct_article"]')
            if xpath_id:
                return xpath_id[0].text_content().strip()
            xpath_class = tree.xpath('//*[contains(@class, "newsct_article_article_body")]')
            if xpath_class:
                return xpath_class[0].text_content().strip()
        except Exception as e:
            print(f"뉴스 본문 조회 실패 ({news_url}): {e}")
        return None

    def scrape_news_until_cutoff_today(self, cutoff_hour=DEFAULT_CUTOFF_HOUR):
        """오늘 날짜에서 특정 시간 이전까지의 뉴스를 수집"""
        today_str = datetime.today().strftime("%Y-%m-%d")
        cutoff_time = datetime.strptime(f"{today_str} {cutoff_hour:02}:00:00", "%Y-%m-%d %H:%M:%S")
        last_page = self.get_last_page(today_str)
        all_news = []

        print(f"뉴스 수집 시작: {today_str}, 기준시간: {cutoff_time}, 총 페이지: {last_page}")

        for page in range(last_page, 0, -1):
            news_list = self.get_news_list_on_page(today_str, page)
            if not news_list:
                continue

            early_news_found = False

            for news in news_list:
                if news["datetime"] > cutoff_time:
                    continue

                early_news_found = True
                body = self.get_news_body(news["url"])
                if not body or len(body) < 100:
                    continue

                all_news.append({
                    "제목": news["title"],
                    "URL": news["url"],
                    "날짜": news["datetime"].strftime("%Y-%m-%d %H:%M:%S"),
                    "본문": body
                })
                time.sleep(0.5)

            if early_news_found:
                continue

            if all(news["datetime"] > cutoff_time for news in news_list):
                break

        print(f"뉴스 수집 완료: 총 {len(all_news)}개")
        return all_news


class VectorDBManager:
    """벡터 데이터베이스 관리를 위한 클래스"""
    
    def __init__(self, model_name="BAAI/bge-m3", docs_file="bk_docs.pkl", 
                 index_path="bk_faiss_index"):
        self.model_name = model_name
        self.docs_file = docs_file
        self.index_path = index_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedding_model = None
        self.faiss_db = None
        self.docs = []
        
    def initialize_embedding_model(self):
        """임베딩 모델 초기화"""
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': self.device}
        )
        
    def load_existing_data(self):
        """기존 문서와 FAISS 인덱스 로드"""
        try:
            with open(self.docs_file, "rb") as f:
                self.docs = pickle.load(f)
            print(f"기존 문서 {len(self.docs)}개 로드 완료")
        except FileNotFoundError:
            print("기존 문서 파일이 없습니다. 새로 생성합니다.")
            self.docs = []
        
        try:
            self.faiss_db = FAISS.load_local(
                self.index_path, 
                self.embedding_model, 
                allow_dangerous_deserialization=True
            )
            print("기존 FAISS 인덱스 로드 완료")
        except Exception as e:
            print(f"기존 FAISS 인덱스 로드 실패: {e}")
            self.faiss_db = None
    
    def save_data(self):
        """문서와 FAISS 인덱스 저장"""
        with open(self.docs_file, "wb") as f:
            pickle.dump(self.docs, f)
        
        if self.faiss_db:
            self.faiss_db.save_local(self.index_path)
        
        print("데이터 저장 완료")
    
    def add_news_to_vector_db(self, news_data, chunk_size=DEFAULT_CHUNK_SIZE, 
                             chunk_overlap=DEFAULT_CHUNK_OVERLAP, 
                             text_limit=DEFAULT_TEXT_LIMIT):
        """뉴스 데이터를 벡터 DB에 추가"""
        if not self.embedding_model:
            self.initialize_embedding_model()
        
        # 기존 데이터의 날짜 집합
        existing_dates = set(doc.metadata.get("일자") for doc in self.docs 
                           if "일자" in doc.metadata)
        
        # 새로운 문서 생성
        new_docs = []
        for news in news_data:
            news_date = news["날짜"]
            if news_date in existing_dates:
                continue
            
            content = f"{news['제목']}\n{news['본문']}"
            if text_limit:
                content = content[:text_limit]
            
            metadata = {
                "일자": news_date, 
                "제목": news["제목"], 
                "URL": news["URL"]
            }
            new_docs.append(Document(page_content=content, metadata=metadata))
        
        if not new_docs:
            print("새로 추가할 뉴스가 없습니다.")
            return False
        
        # 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        new_split_docs = text_splitter.split_documents(new_docs)
        
        # 문서 추가
        self.docs.extend(new_split_docs)
        
        # FAISS DB 업데이트
        if self.faiss_db is None:
            self.faiss_db = FAISS.from_documents(
                new_split_docs, 
                self.embedding_model
            )
        else:
            self.faiss_db.add_documents(new_split_docs)
        
        print(f"새로운 문서 {len(new_split_docs)}개 추가 완료")
        return True


# 유틸리티 함수들
def clean_korean_text(text):
    """한국어 텍스트 정리"""
    if pd.isnull(text):
        return ''
    text = re.sub(r'[\n\t\r]', ' ', text)
    text = re.sub(r'[^\uAC00-\uD7A3a-zA-Z0-9 .,?!]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def scrape_news_until_cutoff_today(cutoff_hour=DEFAULT_CUTOFF_HOUR):
    """편의를 위한 래퍼 함수"""
    collector = NewsCollector()
    return collector.scrape_news_until_cutoff_today(cutoff_hour)


def update_news_vector_db(cutoff_hour=DEFAULT_CUTOFF_HOUR, 
                         docs_file="bk_docs.pkl", 
                         index_path="bk_faiss_index"):
    """뉴스 수집 및 벡터 DB 업데이트 통합 함수"""
    # 뉴스 수집
    collector = NewsCollector()
    news_data = collector.scrape_news_until_cutoff_today(cutoff_hour)
    
    if not news_data:
        print("수집된 뉴스가 없습니다.")
        return
    
    # 데이터 정리
    df = pd.DataFrame(news_data)
    df['제목'] = df['제목'].apply(clean_korean_text)
    df['본문'] = df['본문'].apply(clean_korean_text)
    df.dropna(subset=['제목', '본문'], inplace=True)
    
    news_data_clean = df.to_dict('records')
    
    # 벡터 DB 업데이트
    db_manager = VectorDBManager(docs_file=docs_file, index_path=index_path)
    db_manager.initialize_embedding_model()
    db_manager.load_existing_data()
    
    if db_manager.add_news_to_vector_db(news_data_clean):
        db_manager.save_data()
        print("뉴스 벡터 DB 업데이트 완료!")
    else:
        print("업데이트할 뉴스가 없습니다.")


# 하위 호환성을 위한 함수들 (기존 코드와의 호환성 유지)
def get_last_page(date_str):
    collector = NewsCollector()
    return collector.get_last_page(date_str)

def get_news_list_on_page(date_str, page):
    collector = NewsCollector()
    return collector.get_news_list_on_page(date_str, page)

def get_news_body(news_url):
    collector = NewsCollector()
    return collector.get_news_body(news_url)


if __name__ == "__main__":
    # 테스트 실행
    update_news_vector_db()
