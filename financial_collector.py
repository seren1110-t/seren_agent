"""
KOSPI 주식 정보 및 재무 데이터 수집 모듈
"""
import asyncio
import aiohttp
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import time
import sqlite3
from pykrx import stock

# 상수 정의
DEFAULT_MAX_PAGES = 20
DEFAULT_CONCURRENT_LIMIT = 10
DEFAULT_DB_PATH = "financial_data.db"
DEFAULT_MARKET = "ALL"
DEFAULT_HISTORY_DAYS = 365

class KOSPIDataCollector:
    """KOSPI 데이터 수집을 위한 클래스"""
    
    def __init__(self, max_pages=DEFAULT_MAX_PAGES, concurrent_limit=DEFAULT_CONCURRENT_LIMIT):
        self.max_pages = max_pages
        self.concurrent_limit = concurrent_limit
        self.headers = {'User-Agent': 'Mozilla/5.0'}
    
    def get_kospi_tickers(self):
        """KOSPI 티커 목록을 수집"""
        stocks = []
        print(f"KOSPI 티커 수집 시작 (최대 {self.max_pages}페이지)")
        
        for page in range(1, self.max_pages + 1):
            try:
                url = f"https://finance.naver.com/sise/sise_market_sum.naver?sosok=0&page={page}"
                res = requests.get(url, headers=self.headers)
                soup = BeautifulSoup(res.text, "html.parser")
                rows = soup.select("table.type_2 tr")

                page_stocks = []
                for row in rows:
                    a_tag = row.select_one("a.tltle")
                    if a_tag:
                        name = a_tag.text.strip()
                        href = a_tag['href']
                        code = href.split('code=')[-1]
                        page_stocks.append({"종목명": name, "종목코드": code})
                
                stocks.extend(page_stocks)
                print(f"페이지 {page}: {len(page_stocks)}개 종목 수집")
                
            except Exception as e:
                print(f"페이지 {page} 수집 실패: {e}")
                continue
        
        print(f"총 {len(stocks)}개 KOSPI 티커 수집 완료")
        return pd.DataFrame(stocks)

    async def get_stock_report_async(self, code, session, semaphore):
        """비동기로 개별 주식의 재무 정보를 수집"""
        url = f"https://finance.naver.com/item/main.naver?code={code}"
        today = datetime.now().strftime("%Y-%m-%d")

        async with semaphore:
            try:
                async with session.get(url) as res:
                    html = await res.text()
            except Exception as e:
                print(f"[{code}] 요청 오류: {e}")
                return None

        soup = BeautifulSoup(html, "html.parser")

        # 기본 정보 추출
        try:
            name = soup.select_one("div.wrap_company h2 a").text.strip()
            current_price = soup.select_one("p.no_today .blind").text.strip().replace(',', '')
        except:
            name = None
            current_price = None

        # 재무지표 초기화
        financial_data = {
            "roe_최근": None, "roe_전기": None,
            "debt_최근": None, "debt_전기": None,
            "reserve_최근": None, "reserve_전기": None,
            "per_최근": None, "per_전기": None,
            "pbr_최근": None, "pbr_전기": None,
            "sales_최근": None, "sales_전기": None,
            "operating_최근": None, "operating_전기": None,
            "net_income_최근": None, "net_income_전기": None
        }

        # 재무지표 추출
        try:
            table = soup.select_one("#content > div.section.cop_analysis > div.sub_section > table")
            if table:
                rows = table.select("tbody tr")

                def get_text(row_idx, td_idx):
                    try:
                        return rows[row_idx].select("td")[td_idx].text.strip()
                    except:
                        return None

                financial_data.update({
                    "roe_최근": get_text(5, 8), "roe_전기": get_text(5, 7),
                    "debt_최근": get_text(6, 8), "debt_전기": get_text(6, 7),
                    "reserve_최근": get_text(8, 8), "reserve_전기": get_text(8, 7),
                    "per_최근": get_text(10, 8), "per_전기": get_text(10, 7),
                    "pbr_최근": get_text(12, 8), "pbr_전기": get_text(12, 7),
                    "sales_최근": get_text(0, 8), "sales_전기": get_text(0, 7),
                    "operating_최근": get_text(1, 8), "operating_전기": get_text(1, 7),
                    "net_income_최근": get_text(2, 8), "net_income_전기": get_text(2, 7)
                })
        except Exception as e:
            print(f"[{code}] 재무지표 추출 오류: {e}")

        # 뉴스 추출
        latest_news = []
        try:
            base_url = "https://finance.naver.com"
            news_items = soup.select("#content > div.section.new_bbs > div.sub_section.news_section > ul:nth-child(2) > li > span > a")
            for item in news_items[:3]:
                href = item.get("href")
                if href and href.startswith("/item/news_read.naver"):
                    full_url = base_url + href
                    latest_news.append(full_url)
        except Exception as e:
            print(f"[{code}] 뉴스 추출 오류: {e}")

        return {
            "종목명": name,
            "티커": code,
            "작성일자": today,
            "현재가": current_price,
            "ROE_최근": financial_data["roe_최근"], "ROE_전기": financial_data["roe_전기"],
            "부채비율_최근": financial_data["debt_최근"], "부채비율_전기": financial_data["debt_전기"],
            "유보율_최근": financial_data["reserve_최근"], "유보율_전기": financial_data["reserve_전기"],
            "PER_최근": financial_data["per_최근"], "PER_전기": financial_data["per_전기"],
            "PBR_최근": financial_data["pbr_최근"], "PBR_전기": financial_data["pbr_전기"],
            "매출액_최근": financial_data["sales_최근"], "매출액_전기": financial_data["sales_전기"],
            "영업이익_최근": financial_data["operating_최근"], "영업이익_전기": financial_data["operating_전기"],
            "순이익_최근": financial_data["net_income_최근"], "순이익_전기": financial_data["net_income_전기"],
            "최신뉴스": latest_news
        }

    async def collect_kospi_reports_async(self, limit=None):
        """비동기로 KOSPI 종목들의 재무 정보를 수집"""
        tickers_df = self.get_kospi_tickers()
        if limit:
            tickers_df = tickers_df.head(limit)
            print(f"수집 제한: {limit}개 종목")

        print(f"재무 정보 수집 시작: {len(tickers_df)}개 종목")
        
        semaphore = asyncio.Semaphore(self.concurrent_limit)
        async with aiohttp.ClientSession(headers=self.headers) as session:
            tasks = [
                self.get_stock_report_async(code, session, semaphore) 
                for code in tickers_df['종목코드']
            ]
            reports = await asyncio.gather(*tasks)

        # None 제거
        reports = [r for r in reports if r]
        print(f"재무 정보 수집 완료: {len(reports)}개 종목")
        return pd.DataFrame(reports)


class PriceDataCollector:
    """주식 가격 데이터 수집을 위한 클래스"""
    
    def __init__(self, history_days=DEFAULT_HISTORY_DAYS, market=DEFAULT_MARKET):
        self.history_days = history_days
        self.market = market
    
    def get_price_data(self, reference_ticker="005930"):
        """과거 주식 가격 데이터를 수집"""
        end_date = datetime.today()
        start_date = end_date - timedelta(days=self.history_days)
        
        print(f"가격 데이터 수집: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
        
        try:
            # 날짜 리스트 생성 (삼성전자 기준)
            date_list = stock.get_market_ohlcv_by_date(
                start_date.strftime("%Y%m%d"),
                end_date.strftime("%Y%m%d"),
                reference_ticker
            ).index.strftime("%Y%m%d").tolist()
            
            print(f"거래일 수: {len(date_list)}일")
        except Exception as e:
            print(f"날짜 리스트 생성 실패: {e}")
            return pd.DataFrame()

        price_df = pd.DataFrame()
        success_count = 0
        
        for i, date in enumerate(date_list, 1):
            try:
                daily_prices = stock.get_market_ohlcv_by_ticker(
                    date, market=self.market
                )[["종가"]]
                daily_prices.columns = [date]
                price_df = pd.concat([price_df, daily_prices], axis=1)
                success_count += 1
                
                if i % 50 == 0:  # 진행상황 출력
                    print(f"가격 데이터 수집 진행: {i}/{len(date_list)} ({success_count}개 성공)")
                    
            except Exception as e:
                print(f"{date} 가격 데이터 수집 실패: {e}")
                continue

        print(f"가격 데이터 수집 완료: {success_count}/{len(date_list)}일")
        return price_df


class FinancialDataManager:
    """재무 데이터 통합 관리 클래스"""
    
    def __init__(self, db_path=DEFAULT_DB_PATH):
        self.db_path = db_path
        self.kospi_collector = KOSPIDataCollector()
        self.price_collector = PriceDataCollector()
    
    def collect_and_save_all(self, limit=None):
        """모든 데이터를 수집하고 데이터베이스에 저장"""
        print("=== 전체 데이터 수집 시작 ===")
        
        # 1. 재무 정보 수집
        financial_df = asyncio.run(self.kospi_collector.collect_kospi_reports_async(limit=limit))
        financial_df = financial_df.dropna()
        print(f"유효한 재무 데이터: {len(financial_df)}개")
        
        # 2. 가격 데이터 수집
        price_df = self.price_collector.get_price_data()
        if price_df.empty:
            print("가격 데이터 수집 실패")
            return
        
        # 3. 데이터 병합
        merged_df = price_df.reset_index().merge(financial_df, on="티커", how="inner")
        print(f"병합된 데이터: {len(merged_df)}개 종목")
        
        # 4. 뉴스 리스트를 문자열로 변환
        if '최신뉴스' in merged_df.columns:
            merged_df['최신뉴스'] = merged_df['최신뉴스'].apply(
                lambda x: '\n'.join(x) if isinstance(x, list) else x
            )
        
        # 5. 데이터베이스 저장
        self.save_to_database(merged_df)
        print("=== 전체 데이터 수집 완료 ===")
        
        return merged_df
    
    def save_to_database(self, df):
        """데이터프레임을 SQLite 데이터베이스에 저장"""
        try:
            conn = sqlite3.connect(self.db_path)
            df.to_sql('financial_data', conn, if_exists='replace', index=False)
            conn.close()
            print(f"✅ 데이터베이스 저장 완료: {self.db_path}")
        except Exception as e:
            print(f"❌ 데이터베이스 저장 실패: {e}")
    
    def load_from_database(self, table_name='financial_data'):
        """데이터베이스에서 데이터를 로드"""
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            conn.close()
            print(f"데이터베이스 로드 완료: {len(df)}개 레코드")
            return df
        except Exception as e:
            print(f"데이터베이스 로드 실패: {e}")
            return pd.DataFrame()


# 편의 함수들 (하위 호환성 유지)
def get_kospi_tickers():
    """KOSPI 티커 목록 반환 (기존 코드 호환성)"""
    collector = KOSPIDataCollector()
    return collector.get_kospi_tickers()

async def collect_kospi_reports_async(limit=None):
    """비동기 재무 정보 수집 (기존 코드 호환성)"""
    collector = KOSPIDataCollector()
    return await collector.collect_kospi_reports_async(limit)

def collect_and_save_all():
    """전체 데이터 수집 및 저장 (기존 코드 호환성)"""
    manager = FinancialDataManager()
    return manager.collect_and_save_all()


# 새로운 편의 함수들
def quick_collect(limit=50, db_path=DEFAULT_DB_PATH):
    """빠른 데이터 수집 (제한된 종목 수)"""
    manager = FinancialDataManager(db_path=db_path)
    return manager.collect_and_save_all(limit=limit)

def get_financial_summary(db_path=DEFAULT_DB_PATH):
    """저장된 재무 데이터의 요약 정보 반환"""
    manager = FinancialDataManager(db_path=db_path)
    df = manager.load_from_database()
    
    if df.empty:
        return "저장된 데이터가 없습니다."
    
    summary = {
        "종목_수": len(df),
        "데이터_수집일": df['작성일자'].iloc[0] if '작성일자' in df.columns else "정보없음",
        "평균_PER": df['PER_최근'].replace(['-', ''], None).astype(float).mean() if 'PER_최근' in df.columns else None,
        "평균_PBR": df['PBR_최근'].replace(['-', ''], None).astype(float).mean() if 'PBR_최근' in df.columns else None,
    }
    
    return summary


if __name__ == "__main__":
    # 테스트 실행 (50개 종목으로 제한)
    print("테스트 실행: 50개 종목 데이터 수집")
    quick_collect(limit=50)
    
    # 요약 정보 출력
    summary = get_financial_summary()
    print(f"수집 요약: {summary}")
