import os
import time
import uuid
import joblib

from typing import List
from llama_index.readers.google import GoogleDriveReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    PointStruct,
    SparseVector,
    Filter,
    FieldCondition,
    MatchValue,
)
from sklearn.feature_extraction.text import TfidfVectorizer
import json
from qdrant_client.models import PointIdsList, FilterSelector


from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from concurrent.futures import ThreadPoolExecutor, as_completed
from llama_index.core.node_parser import SentenceSplitter
import nltk
from config import CLIENT_CONFIG, SCOPES, TOKEN_PATH

from functools import lru_cache

INTERVAL = 10  # API 호출 간격 (초 단위)


class GDriveQdrantIndexer:
    def __init__(
        self,
        service_account_key_path: str,
        folder_id: str,
        collection_name: str,
        qdrant_host: str,
        qdrant_port: int,
        embedding_api_base: str,
        model_name: str = "BAAI/bge-m3",
        drive_id: str = None,
        page_token_path: str = "start_page_token.txt",
        vectorizer_path: str = "fitted_tfidf_vectorizer.joblib",
        service_token_path: str = "token.json",
    ):
        self.service_account_key_path = service_account_key_path
        self.folder_id = folder_id
        self.collection_name = collection_name
        self.qdrant_client = QdrantClient(
            host=qdrant_host, port=qdrant_port, timeout=180, prefer_grpc=True
        )
        self.embedding_model = OpenAILikeEmbedding(
            model_name=model_name,
            api_base=embedding_api_base,
            api_key="fake",
            embed_batch_size=10,
        )
        self.docs = []
        # self.nodes = []
        self.drive_id = drive_id
        self.page_token_path = page_token_path
        self.vectorizer_path = vectorizer_path
        if os.path.exists(self.vectorizer_path):
            self.vectorizer = joblib.load(self.vectorizer_path)
        else:
            self.vectorizer = None
        self.service_token_path = service_token_path
        self.drive_service = self.get_drive_service()

        # (a) 없는 리소스만 다운로드
        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)

        # (2) 메인 스레드에서 미리 로드
        from nltk.corpus import stopwords

        _ = stopwords.words("english")  # WordListCorpusReader가 메모리에 올라갑니다

        from nltk.tokenize import sent_tokenize

        _ = sent_tokenize("이 문장을 토크나이즈하면 punkt가 로드됩니다.")

        # 2) SentenceSplitter 인스턴스는 한 번만 생성
        self.parser = SentenceSplitter(chunk_size=512, chunk_overlap=40)

    def _get_start_page_token(self) -> str:
        resp = self.drive_service.changes().getStartPageToken().execute()
        return resp["startPageToken"]

    def _load_saved_token(self) -> str:
        if os.path.exists(self.page_token_path):
            return open(self.page_token_path, "r").read().strip()
        return None

    def _save_token(self, token: str):
        with open(self.page_token_path, "w") as f:
            f.write(token)

    def get_drive_service(self):
        creds = None
        token_path = self.service_token_path
        # (1) 저장된 토큰 파일(token.json)이 있으면 불러오기
        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)

        # (2) 토큰이 없거나 유효하지 않으면 갱신하거나 새로 인증
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                # refresh_token이 있으면 자동으로 갱신
                creds.refresh(Request())
            else:
                # from google.oauth2 import service_account
                # creds = service_account.Credentials.from_service_account_file(self.service_account_key_path, scopes=SCOPES)
                if not creds:
                    # 처음이거나 갱신 불가한 경우, 새로 OAuth 흐름 수행
                    flow = InstalledAppFlow.from_client_config(CLIENT_CONFIG, SCOPES)
                    creds = flow.run_local_server(
                        host="127.0.0.1", port=39511, open_browser=False
                    )

            # (3) 갱신된/새로 받은 토큰을 다시 저장
            with open(token_path, "w") as token_file:
                token_file.write(creds.to_json())

        # (4) Drive API 클라이언트 생성
        service = build("drive", "v3", credentials=creds)
        self.drive_service = service
        return service

    def change_folder(self, folder_id: str):
        self.folder_id = folder_id

    def recreate_collection(self):
        # 1) 컬렉션 존재 여부 확인
        if self.qdrant_client.collection_exists(self.collection_name):
            # 2) 이미 있으면 삭제
            # self.qdrant_client.delete_collection(collection_name=self.collection_name)
            return
        # 3) 새로운 컬렉션 생성
        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            sparse_vectors_config={"bm25": SparseVectorParams()},
        )

    def load_documents(self):
        loader = GoogleDriveReader(
            service_account_key_path=self.service_account_key_path,
            folder_id=self.folder_id,
            drive_id=self.drive_id,
            recursive=True,
            token_path="token.json",
        )
        docs = loader.load_data()
        for doc in docs:
            bname = os.path.basename(doc.metadata["file path"])
            ext = os.path.splitext(bname)[1]
            doc.metadata["file_name"] = bname
            doc.metadata["file_type"] = ext
        print(f"[load] {len(docs)}개의 문서를 로드했습니다.")
        return docs

    def parse_document(self, doc):
        """
        단일 Google Drive 문서를 파싱합니다.
        :doc: 문서 객체 (GoogleDriveReader에서 로드된)
        :return: 파싱된 노드 리스트
        """
        nodes = []
        for node in self.parser.get_nodes_from_documents([doc]):
            node.metadata.update(
                {
                    "original_doc_id": doc.id_,
                    "original_file_name": doc.metadata.get("file_name", "Unknown"),
                }
            )
            nodes.append(node)

        # self.nodes.extend(nodes)
        return nodes

    def parse_documents(self, docs):
        nodes = []
        for doc in docs:
            if doc.text and doc.text.strip():
                node = self.parse_document(doc)
                nodes.extend(node)
        print(f"[parse] {len(nodes)}개의 노드를 생성했습니다.")
        return nodes

    def parse_documents_batch(self, docs, max_workers: int = 16):
        """
        self.docs에 담긴 문서를 병렬로 parse_document에 넘겨 노드 생성 후 self.nodes에 합칩니다.
        """
        nodes = []
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            futures = {exe.submit(self.parse_document, doc): doc for doc in docs}
            for f in as_completed(futures):
                doc = futures[f]
                try:
                    nodes.extend(f.result())
                except Exception as e:
                    fname = doc.metadata.get("file_name", doc.id_)
                    print(f"[parse error] '{fname}' 문서 파싱 중 오류: {e}")

        print(f"[parse] {len(nodes)}개의 노드를 생성했습니다.")
        return nodes

    def generate_embeddings(self, texts: List[str] = None):
        dense = self.embedding_model.get_text_embedding_batch(texts, show_progress=True)
        # 피팅되지 않았다면 전체 문서로 피팅
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(max_df=0.90, min_df=5, max_features=15000)
            sparse_matrix = self.vectorizer.fit_transform(texts)

        else:
            sparse_matrix = self.vectorizer.transform(texts)
        joblib.dump(self.vectorizer, self.vectorizer_path)
        return dense, sparse_matrix

    def _delete_qdrant_points_for(self, file_id: str):
        """
        주어진 Google Drive 파일 ID와 매핑된 모든 Qdrant 포인트를 삭제합니다.
        """
        deletion_filter = FilterSelector(
            filter=Filter(
                must=[
                    FieldCondition(
                        key="original_doc_id",
                        match=MatchValue(value=file_id),
                    )
                ]
            )
        )
        self.qdrant_client.delete(
            collection_name=self.collection_name,
            points_selector=deletion_filter,
            wait=True,
        )

        print(f"[delete] Qdrant에서 original_doc_id={file_id} 포인트 삭제 완료")

    def _index_changed_file(self, file_id: str):
        """
        변경(신규 생성 또는 수정)된 단일 파일만 로드·파싱·임베딩·업서트합니다.
        * file_id: Google Drive 상의 파일 ID
        """
        # 1) 문서 로드
        loader = GoogleDriveReader(
            service_account_key_path=self.service_account_key_path,
            file_ids=[file_id],
            recursive=False,
            token_path="token.json",
        )

        docs = loader.load_data()
        if not docs:
            print(f"[index] 파일을 찾을 수 없습니다: {file_id}")
            return

        # 2) 문서 메타데이터 정리 + 문장 단위 노드 생성
        nodes = []
        for doc in docs:
            # 메타데이터에 파일명/확장자 추가
            path = doc.metadata.get("file path", "")
            name = os.path.basename(path)
            ext = os.path.splitext(name)[1]
            doc.metadata["file_name"] = name
            doc.metadata["file_type"] = ext

            nodes.extend(self.parse_document(doc))

        if not nodes:
            print(f"[index] 문서에 텍스트가 없어 색인할 노드가 없습니다: {file_id}")
            return

        # 3) 임베딩 생성
        texts = [n.get_content() for n in nodes]
        dense_vectors, sparse_matrix = self.generate_embeddings(texts)

        # 4) Qdrant 업서트
        self.upsert_to_qdrant(nodes, dense_vectors, sparse_matrix)
        print(f"[index] Qdrant에 파일 업서트 완료: {file_id}")
        # self.nodes.extend(nodes)

    def upsert_to_qdrant(self, nodes, dense_vectors, sparse_matrix, batch_size=300):
        all_points = []
        for i, node in enumerate(nodes):
            file_id = node.metadata.get("original_doc_id", "")
            stable_uuid = uuid.uuid5(uuid.NAMESPACE_URL, f"{file_id}_{i}")
            point_id = stable_uuid.hex
            payload = {
                **{
                    k: str(node.metadata.get(k, ""))
                    for k in [
                        "file_name",
                        "file_type",
                        "file path",
                        "file id",
                        "author",
                        "created at",
                        "modified at",
                        "mime type",
                    ]
                },
                "original_doc_id": node.metadata.get("original_doc_id", ""),
                "text_chunk_id": node.id_,
                "text": node.text,
            }
            tfidf_row = sparse_matrix[i]
            sparse_vec = SparseVector(
                indices=tfidf_row.indices.tolist(), values=tfidf_row.data.tolist()
            )
            point = PointStruct(
                id=point_id,
                payload=payload,
                vector={"": dense_vectors[i], "bm25": sparse_vec},
            )
            all_points.append(point)

        # batch 단위로 upsert
        from concurrent.futures import ThreadPoolExecutor

        batches = [
            all_points[i : i + batch_size]
            for i in range(0, len(all_points), batch_size)
        ]
        with ThreadPoolExecutor(max_workers=8) as exe:
            exe.map(
                lambda batch: self.qdrant_client.upsert(
                    collection_name=self.collection_name, points=batch, wait=True
                ),
                batches,
            )

    def save_vectorizer(self, path="fitted_tfidf_vectorizer.joblib"):
        if self.vectorizer:
            joblib.dump(self.vectorizer, path)

    def query(self, query_text: str, top_k=10):
        dense_vector = self.embedding_model.get_text_embedding(query_text)
        sparse = self.vectorizer.transform([query_text])
        sparse_vec = SparseVector(
            indices=sparse.indices.tolist(), values=sparse.data.tolist()
        )
        result = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=dense_vector,
            prefetch=[models.Prefetch(query=sparse_vec, using="bm25", limit=top_k)],
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        return result.points if hasattr(result, "points") else []

    def upload_drive_folder(self, folder_id: str):
        """
        Google Drive 폴더에 있는 모든 파일을 Qdrant에 업서트합니다.
        :param folder_id: Google Drive 폴더 ID
        """

        self.change_folder(folder_id)
        self.recreate_collection()
        start_token = self._get_start_page_token()
        docs = self.load_documents()
        nodes = self.parse_documents_batch(docs)
        dense_vectors, sparse_matrix = self.generate_embeddings(
            [node.get_content() for node in nodes]
        )

        self.upsert_to_qdrant(nodes, dense_vectors, sparse_matrix)
        print(f"[upload] {len(nodes)}개의 노드를 Qdrant에 업서트했습니다.")
        self.save_vectorizer(self.vectorizer_path)
        self._save_token(start_token)

    def _is_under_target_folder(
        self, file_id: str, target_folder_id: str, drive_id: str
    ) -> bool:
        """
        file_id 의 조상(parents, parents의 parents, …)을 drive_id까지 타고 올라가며
        target_folder_id가 하나라도 있으면 True, 아니면 False 반환.
        """

        @lru_cache(maxsize=1024)
        def _get_parents(fid: str) -> list:
            # 파일(또는 폴더)의 부모 ID 리스트를 가져와 캐싱
            meta = (
                self.drive_service.files()
                .get(
                    fileId=file_id,
                    fields="id, name, parents",
                    supportsAllDrives=True,
                )
                .execute()
            )
            return meta.get("parents", [])

        stack = [file_id]
        visited = set()

        while stack:
            fid = stack.pop()
            if fid in visited:
                continue
            visited.add(fid)

            parents = _get_parents(fid)
            for pid in parents:
                # 1) 바로 타겟 폴더라면 True
                if pid == target_folder_id:
                    return True
                # 2) 드라이브 루트(또는 공유 드라이브) ID라면 더 이상 올라갈 곳이 없으니 건너뛴다
                if pid == drive_id:
                    continue
                # 3) 그 외 부모 폴더도 이어서 검사
                stack.append(pid)

        return False

        # --- (D) 증분 변경 감지 & 처리 메서드 ---

    def process_changes(self):
        """
        • Google Drive change list에서 생성·수정·삭제된 파일을 순회하며
          - 삭제: Qdrant에서 해당 original_doc_id에 해당하는 모든 포인트 삭제
          - 생성/수정: 문서 로드 → 파싱 → 임베딩 → upsert
        • 페이지 토큰은 한 번만 가져와 저장하고, 매 실행시 갱신된 토큰으로 차이만 처리
        """
        # 1) 초기 토큰 확보
        saved_token = self._load_saved_token()
        if saved_token is None:
            start_token = self._get_start_page_token()
            self._save_token(start_token)
            saved_token = self._load_saved_token()
            print("첫 페이지 토큰을 저장했습니다:", start_token)
            return  # 다음 실행부터 변경사항 감지

        page_token = saved_token
        print("저장된 페이지 토큰을 불러왔습니다:", page_token)
        while True:
            resp = (
                self.drive_service.changes()
                .list(
                    pageToken=page_token,
                    includeRemoved=True,
                    includeItemsFromAllDrives=True,
                    supportsAllDrives=True,
                    fields="nextPageToken,newStartPageToken,changes(fileId,removed,file(name,mimeType,modifiedTime,parents,trashed))",
                    pageSize=1000,
                )
                .execute()
            )
            print(f"처리할 변경사항: {len(resp.get('changes', []))}개")
            # print("here is the response:", json.dumps(resp, indent=2))
            for change in resp.get("changes", []):
                fid = change["fileId"]
                file_meta = change.get("file", {})
                parents = file_meta.get("parents", [])
                removed = change.get("removed") or change["file"].get("trashed", False)
                # — 삭제 혹은 휴지통 이동된 파일
                under_target = self._is_under_target_folder(
                    fid, self.folder_id, self.drive_id
                )

                print("parents:", parents, file_meta)
                # print("under_target:", under_target)
                if self.folder_id not in parents and not under_target:
                    print(
                        f"[skip] {fid}는 타겟 폴더({self.folder_id})에 속하지 않습니다."
                    )
                    continue

                self._delete_qdrant_points_for(fid)

                # — 신규 또는 수정된 파일
                if not removed:
                    # 변경 파일이 우리 폴더에 포함될 때만 색인
                    self._index_changed_file(fid)

            # 페이지 토큰 갱신
            new_start = resp.get("newStartPageToken")
            if new_start:
                self._save_token(new_start)
            next_token = resp.get("nextPageToken")

            if not next_token:
                page_token = new_start
                time.sleep(INTERVAL)
            # print("here page_token:", page_token, new_start)
