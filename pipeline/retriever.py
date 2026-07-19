from typing import Any
import re
import numpy as np
import pandas as pd
import json
import requests
OLLAMA_MODEL = "qwen3.5:9b"
def _ticker_variants(ticker: str) -> set[str]:
    """여러 형태로 등장하는 ticker 표기 반환"""
    return {
        ticker,
        ticker.replace("-", "."),
        ticker.replace("-","")
    }

class Retriever:


    def __init__(
            self,
            *,
            chunk_df: pd.DataFrame,
            embeddings: np.ndarray,
            index: Any,
            embedding_model: Any,
            reranker: Any,
            ticker_to_company: dict[str, str],
            entity_df: pd.DataFrame,
            entity_embeddings: np.ndarray,
                    ) -> None:
        """Retriever에 필요한 모델과 데이터를 전달받아 불일치 여부를 확인함"""

        # 청크와 임베딩의 데이터 개수가 일치하는지 확인
        if len(chunk_df) != embeddings.shape[0]:
            raise ValueError(
                "chunk_df와 embeddings의 데이터 개수 불일치"
            )
        # 청크와 FAISS 인덱스의 데이터 개수가 일치하는지 확인
        if index.ntotal != len(chunk_df):
            raise ValueError(
                "FAISS index와 chunk_df의 데이터 개수 불일치"
            )
        
        # 임베딩한 차원과 FAISS에 저장된 차원의 개수가 일치하는지 확인
        # 정상이라면 384차원
        if index.d != embeddings.shape[1]:
            raise ValueError(
                "FAISS index와 embeddings의 차원 불일치"
            )
        
        # 회사 정보와 회사 정보를 임베딩 한 데이터의 개수가 일치하는지 확인
        if len(entity_df) != entity_embeddings.shape[0]:
            raise ValueError(
                "entity_df와 entity_embeddings의 데이터 개수 불일치"
            )
        
        self.chunk_df = chunk_df
        self.embeddings = embeddings
        self.index = index
        self.embedding_model = embedding_model
        self.reranker = reranker
        self.ticker_to_company = ticker_to_company
        self.entity_df = entity_df
        self.entity_embeddings = entity_embeddings
    

    def find_explicit_tickers(self, query: str) -> list[str]:
        """쿼리에 명시적으로 등장한 ticker를 찾아 반환"""
        
        found = []

        for ticker in self.ticker_to_company:
            compact_ticker = ticker.replace("-", "").replace(".","")

            # 너무 짧은 ticker는 일반 단어와 잘못 매칭될 가능성이 높으므로 여기서는 제외
            if len(compact_ticker) < 3:
                continue

            for variant in _ticker_variants(ticker):
                pattern = (
                    r"(?<![A-Za-z0-9])"
                    + re.escape(variant)
                    + r"(?![A-Za-z0-9])"
                )

                if re.search(pattern, query):
                    found.append(ticker)
                    break
        
        return found
    
    def build_ticker_candidates(
            self,
            query: str,
            top_k: int = 10,
    ) -> list[dict[str, str]]:
        """명시적 ticker와 임베딩 검색 결과를 합쳐 후보 목록을 생성"""

        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True,
        ).astype("float32")[0]

        scores = self.entity_embeddings @ query_embedding
        top_indices = np.argsort(scores)[::-1][:top_k]

        tickers = []
        tickers.extend(self.find_explicit_tickers(query))

        for idx in top_indices:
            tickers.append(self.entity_df.iloc[idx]["ticker"])

        # 명시적 검색과 임베딩 검색에서 중복된 ticker 제거
        # 명시적 검색을 우선시 할 것이므로 삽입 순서를 유지하는 dict를 사용
        tickers = list(dict.fromkeys(tickers))

        candidates = [
            {
                "ticker": ticker,
                "company": self.ticker_to_company[ticker],
            }
            for ticker in tickers
        ]

        return candidates
    
    def resolve_tickers(
            self,
            query: str,
            candidates: list[dict[str, str]]
    ) -> dict[str, Any]:
        """후보 ticker들 중에서 질문이 실제로 묻는 회사의 ticker를 LLM이 선택"""
        
        candidate_tickers = [
            candidate["ticker"]
            for candidate in candidates
        ]

        ticker_resolution_schema = {
            "type": "object",
            "properties": {
                "tickers": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": candidate_tickers,
                    },
                },
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                },
            },
            "required": ["tickers", "confidence"],
            "additionalProperties": False,
        }

        candidate_text = "\n".join(
            f"- {candidate['ticker']}: {candidate['company']}"
            for candidate in candidates
        )

        prompt = f"""
You are an entity linker for financial QA.

Your task is to choose which candidate company or companies the question is actually asking about.

Candidates:
{candidate_text}

Return raw JSON only in this format:
{{"tickers": ["AAPL"], "confidence": "high"}}

Rules:
- Use only tickers from the candidate list.
- Select the company or companies that the question is actually asking about.
- If multiple candidates are truly required for comparison or relationship analysis, return multiple tickers.
- Ignore stock exchanges, accounting terms, financial metrics, rating agencies, products, and generic market words unless they are clearly the target company.
- If none of the candidates is clearly the target company, return an empty ticker list and confidence "low".
- Do not explain your answer.
- Do not use markdown code fences.

Question:
{query}
""".strip()

        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "stream": False,
                "format": ticker_resolution_schema,
                "think": False,
                "options": {
                    "temperature": 0,
                },
            },
            timeout=60,
        )

        response.raise_for_status()

        parsed = json.loads(
            response.json()["message"]["content"]
        )

        tickers = [
            ticker
            for ticker in parsed["tickers"]
            if ticker in candidate_tickers
        ]

        confidence = parsed["confidence"]

        if not tickers:
            confidence = "low"

        return {
            "tickers": tickers,
            "confidence": confidence,
        }
    
    def retrieve_candidates(
            self,
            query: str,
            tickers: list[str] | None = None,
            top_k: int = 50,
            fetch_k: int = 200,
    ) -> pd.DataFrame:
        """벡터 유사도를 기반으로 쿼리와 유사한 문서 청크 검색"""

        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True,
        ).astype("float32")

        # 티커가 확정되지 않은 경우엔 전체 문서 청크를 대상으로 검색 수행
        if tickers is None:
            scores, indices = self.index.search(
                query_embedding,
                fetch_k,
            )

            candidates = self.chunk_df.iloc[indices[0]].copy()
            candidates["retriever_score"] = scores[0]

            return candidates[
                ["retriever_score",
                "chunk_id",
                "ticker",
                "chunk_type",
                "section_title",
                "text",
                ]
            ].head(top_k)
        
        # 티커가 확정된 경우, 해당 티커들만을 후보 문서로 둠
        ticker_mask = self.chunk_df["ticker"].isin(tickers)

        ticker_chunk_df = self.chunk_df[ticker_mask].copy()
        ticker_embeddings = self.embeddings[
            ticker_mask.to_numpy()
        ]

        scores = ticker_embeddings @ query_embedding[0]
        candidates = ticker_chunk_df.copy()
        candidates["retriever_score"] = scores

        candidates = candidates.sort_values(
            "retriever_score",
            ascending=False,
        ).head(top_k)

        return candidates[
            [
                "retriever_score",
                "chunk_id",
                "ticker",
                "chunk_type",
                "section_title",
                "text",
            ]
        ]
    
    def rerank_candidates(
            self,
            query: str,
            candidates: pd.DataFrame,
            top_k: int = 5,
    ) -> pd.DataFrame:
        """검색된 후보 문서를 reranker로 재정렬"""

        company_names = (
            candidates["ticker"]
            .map(self.ticker_to_company)
            .fillna(candidates["ticker"])
        )

        reranker_texts = (
            "Company: " + company_names.astype(str)
            +"\nTicker: " + candidates["ticker"].astype(str)
            +"\n\n" + candidates["text"].astype(str)
        ).tolist()

        pairs = [
            [query, text]
            for text in reranker_texts
        ]

        reranker_scores = self.reranker.predict(pairs)

        reranked = candidates.copy()
        reranked["reranker_score"] = reranker_scores

        reranked = reranked.sort_values(
            "reranker_score",
            ascending=False
        )
        return reranked[
            [
                "reranker_score",
                "retriever_score",
                "chunk_id",
                "ticker",
                "chunk_type",
                "section_title",
                "text",
            ]
        ].head(top_k)
    