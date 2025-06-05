"""
title: Llama Index + Qdrant Hybrid Retrieval with Qwen3-32B LLM
author: danny, Libero, Justine
date: 2024-06-05
version: 2.0
license: MIT
description: Hybrid dense + sparse retrieval with LLM answer generation plus qdrant  metadata filtering.
requirements: llama-index, qdrant-client, scikit-learn, llama-index-embeddings-openai-like, llama-index-llms-openai-like
"""

from typing import List, Union, Generator, Iterator, Dict
import os
import re
import joblib
from schemas import OpenAIChatMessage
from pydantic import BaseModel

# Qdrant and TF-IDF
from qdrant_client import QdrantClient, models
from sklearn.feature_extraction.text import TfidfVectorizer

# Embedding and LLM
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import Settings

os.environ["HF_HOME"] = "/app/pipelines/model"

from qdrant_client.http.models import Filter, FieldCondition, MatchValue

def normalize_quotes(text: str) -> str:
    # ‘ ’ → ' / “ ” → "
    return text.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"') 

def split_metadata_and_query(user_message: str) -> tuple[str, str]:
    user_message = normalize_quotes(user_message)

    # Allow: #key=value, #key="value with spaces", #key='value with spaces'
    tag_pattern = r'([#!?]\w+\s*[:=]\s*(?:"[^"]+"|\'[^\']+\'|[^\s#?!]+))'
    raw_tags = re.findall(tag_pattern, user_message)

    # Remove extra whitespace and clean each tag
    cleaned_tags = []
    for tag in raw_tags:
        # Separate key=..., value=... parts
        m = re.match(r'([#!?]\w+\s*[:=]\s*)(["\']?)(.*?)(\2)$', tag.strip())
        if m:
            key_part, quote, value, _ = m.groups()
            cleaned = f"{key_part}{quote}{value.strip()}{quote}"
            cleaned_tags.append(cleaned)
        else:
            cleaned_tags.append(tag.strip())

    meta_part = ' '.join(cleaned_tags)

    # Remove metadata tags from the original message to isolate the query
    query = re.sub(tag_pattern, '', user_message).strip()

    return meta_part, query

def parse_advanced_prompt_to_filter(prompt: str) -> Filter:
    must_conditions = []
    must_not_conditions = []
    should_conditions = []

    # Match patterns like #key=value, !key=value, ?key=value, #key:value, !key:value, ?key:value
    # whitespace tolerant pattern
    pattern = r'([#!?])(\w+)\s*([:=])\s*(?:"([^"]+)"|\'([^\']+)\'|([^\s#]+))'
    matches = re.findall(pattern, prompt)

    for prefix, key, sep, val1, val2, val3 in matches:
        value = val1 or val2 or val3
        condition = FieldCondition(key=key, match=MatchValue(value=value))
        if prefix == "#":
            must_conditions.append(condition)
        elif prefix == "!":
            must_not_conditions.append(condition)
        elif prefix == "?":
            should_conditions.append(condition)

    # If no conditions are found, explicitly return None or an empty Filter (if supported by Qdrant)
    if not must_conditions and not must_not_conditions and not should_conditions:
        return None  # Or: return Filter() → may not be accepted by Qdrant

    filter_kwargs = {}
    if must_conditions:
        filter_kwargs["must"] = must_conditions
    if must_not_conditions:
        filter_kwargs["must_not"] = must_not_conditions
    if should_conditions:
        filter_kwargs["should"] = should_conditions

    return Filter(**filter_kwargs)

class Pipeline:
    class Valves(BaseModel):
        QDRANT_HOST: str
        QDRANT_PORT: str
        QDRANT_COLLECTION_NAME: str
        EMBED_MODEL_NAME: str
        EMBED_API_BASE: str
        EMBED_API_KEY: str
        LLM_MODEL_NAME: str
        LLM_API_BASE: str
        LLM_CONTEXT_WINDOW: str
        LLM_API_KEY: str
        QDRANT_URL: str
        QDRANT_API_KEY: str
    
    VECTORIZER_PATH = "/app/pipelines/resource/fitted_tfidf_vectorizer.joblib"

    def __init__(self):
        # Initialize                                                                                                     
        self.valves = self.Valves(                          
            **{                                             
                "QDRANT_HOST": os.getenv("QDRANT_HOST", "192.168.45.131"),
                "QDRANT_PORT": os.getenv("QDRANT_PORT", "6333"),
                "QDRANT_COLLECTION_NAME": os.getenv("QDRANT_COLLECTION_NAME", "gdrive_docs3"),
                "EMBED_MODEL_NAME": os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3"),
                "EMBED_API_BASE": os.getenv("EMBED_API_BASE", "http://192.168.45.131:11001/v1"),
                "EMBED_API_KEY": os.getenv("EMBED_API_KEY", "fake"),
                "LLM_MODEL_NAME": os.getenv("LLM_MODEL_NAME", "Qwen/Qwen3-32B"),
                "LLM_API_BASE": os.getenv("LLM_API_BASE", "http://192.168.45.131:11000/v1"),
                "LLM_CONTEXT_WINDOW": os.getenv("LLM_CONTEXT_WINDOW", "32768"),
                "LLM_API_KEY": os.getenv("LLM_API_KEY", "fake"),
                "QDRANT_URL": os.getenv("QDRANT_URL", ""),
                "QDRANT_API_KEY": os.getenv("QDRANT_API_KEY", "")
            }
        )

        self.qdrant_client: QdrantClient = None
        self.embedding_model: OpenAILikeEmbedding = None
        self.vectorizer: TfidfVectorizer = None
        self.llm: OpenAILike = None

    async def on_startup(self):
        print("\U0001f527 Initializing models and Qdrant client...")

        # Initialize embedding model
        self.embedding_model = OpenAILikeEmbedding(
            model_name=self.valves.EMBED_MODEL_NAME,
            api_base=self.valves.EMBED_API_BASE,
            api_key=self.valves.EMBED_API_KEY
        )
        Settings.embed_model = self.embedding_model

        # Initialize LLM
        self.llm = OpenAILike(
            model=self.valves.LLM_MODEL_NAME,
            api_base=self.valves.LLM_API_BASE,
            api_key=self.valves.LLM_API_KEY,
            context_window=self.valves.LLM_CONTEXT_WINDOW
        )
        Settings.llm = self.llm

        # Load TF-IDF vectorizer
        try:
            self.vectorizer = joblib.load(self.VECTORIZER_PATH)
        except Exception as e:
            print(f"[ERROR] Failed to load vectorizer: {e}")
            self.vectorizer = None

        # Initialize Qdrant
        self.qdrant_client = (
            QdrantClient(url=self.valves.QDRANT_URL, api_key=self.valves.QDRANT_API_KEY)
            if self.valves.QDRANT_URL
            else QdrantClient(host=self.valves.QDRANT_HOST, port=self.valves.QDRANT_PORT)
        )
        print("\u2705 Initialization complete.")

    async def on_shutdown(self):
        if self.qdrant_client:
            self.qdrant_client.close()
            print("Qdrant client closed.")

    def pipe(
        self, 
        user_message: str, 
        model_id: str, 
        messages: List[dict], 
        body: dict
        ) -> Union[str, Generator, Iterator]:

        if not all([self.qdrant_client, self.embedding_model, self.vectorizer, self.llm]):
            yield "Error: One or more pipeline components failed to initialize."
            return

        try:
            user_message = normalize_quotes(user_message)
            meta_part, query = split_metadata_and_query(user_message)
            metadata_filter = parse_advanced_prompt_to_filter(meta_part)

            dense_vec = self.embedding_model.get_text_embedding(query)
            tfidf = self.vectorizer.transform([query])
            sparse_vec = models.SparseVector(
                indices=tfidf.indices.tolist(),
                values=tfidf.data.tolist()
            )

            # Dense-only search
            dense_results = self.qdrant_client.query_points(
                collection_name=self.valves.QDRANT_COLLECTION_NAME,
                query=dense_vec,
                limit=20,
                query_filter=metadata_filter,
                with_payload=True
            ).points

            # Sparse-only search (must specify 'using="sparse"')
            sparse_results = self.qdrant_client.query_points(
                collection_name=self.valves.QDRANT_COLLECTION_NAME,
                query=sparse_vec,
                using="bm25",
                limit=20,
                query_filter=metadata_filter,
                with_payload=True
            ).points

            def normalize(results):
                if not results:
                    return {}
                scores = [r.score for r in results]
                min_s, max_s = min(scores), max(scores)
                denom = (max_s - min_s) if max_s != min_s else 1e-5
                return {r.id: (r.score - min_s) / denom for r in results}

            dense_scores = normalize(dense_results)
            sparse_scores = normalize(sparse_results)
            alpha = 0.5
            all_ids = set(dense_scores) | set(sparse_scores)
            hybrid_scores = {
                pid: alpha * dense_scores.get(pid, 0.0) + (1 - alpha) * sparse_scores.get(pid, 0.0)
                for pid in all_ids
            }
            sorted_ids = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            top_ids = [pid for pid, _ in sorted_ids]

            all_hits = {r.id: r for r in dense_results + sparse_results if r.id in top_ids}

        except Exception as e:
            yield f"❌ Hybrid search failed: {e}"
            return


        print("here top_ids", top_ids)
        print("here all_hits", all_hits)
        context_docs = "\n\n".join(
            all_hits[pid].payload.get("text", "") for pid in top_ids if "text" in all_hits[pid].payload
        )

        if not context_docs.strip():
            yield "No valid text found in retrieved payloads."
            return

        prompt = f"""너는 주어진 컨텍스트를 바탕으로 질문에 답변하는 AI야. 반드시 한국어로 명확히 대답해줘.

--- Context ---
{context_docs}

--- Task ---
{query}

--- 한국말로 대답 ---"""

        try:
            response = self.llm.complete(prompt=prompt, max_tokens=1024)
            yield f"\ud83d\udcac LLM Response:\n{response.text.strip()}\n"
        except Exception as e:
            yield f"❌ Failed to generate LLM response: {e}"
            return

        seen = set()
        
        count = 0 
        for i, pid in enumerate(top_ids):
            hit = all_hits[pid]
            print("here hit.payload", hit.payload)
            file_path = hit.payload.get("file path", "N/A")
            file_id = hit.payload.get("file id", "N/A")
            print("here file_id", file_id)
            if file_id in seen:
                continue
            seen.add(file_id)
            hyperlink = f'https://drive.google.com/file/d/{file_id}/view' if file_id != "N/A" else "N/A"
            yield f"\n \ud83d\udcc1Ref {count+1}: [{file_path}]({hyperlink})\n"
            count += 1
