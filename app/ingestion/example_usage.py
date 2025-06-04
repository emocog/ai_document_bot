from app.ingestion.gdrive_indexer import GDriveQdrantIndexer

indexer = GDriveQdrantIndexer(
    service_account_key_path="../../config/service_account_key.json",
    folder_id="<<drive_id 변수>>",
    collection_name="gdrive_docs0",
    qdrant_host="localhost",
    qdrant_port=6333,
    embedding_api_base="http://192.168.45.131:11001/v1",
    drive_id="<<drive_id 변수>>",
)  # 선택한 드라이브 ID 사용

indexer.upload_drive_folder(
    folder_id="1IMgRMKhFhNjZTAuhWcrsWFGliAJnnORZ"
)  # 실제 폴더 ID로 변경 필요
indexer.change_folder("1IMgRMKhFhNjZTAuhWcrsWFGliAJnnORZ")  # 폴더 변경
indexer.process_changes()  # 변경 사항 처리

docs = indexer.docs
for doc in docs:
    print(
        f"Document ID: {doc.id_}, "
        f"File Name: {doc.metadata.get('file_name','unknown')}, "
        f"Text Length: {len(doc.text)}, "
        f"Metadata: {doc.metadata}"
    )
    print(doc)
