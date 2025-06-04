import os
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from config import CLIENT_CONFIG, SCOPES, TOKEN_PATH


service_account_key_path = "../../config/service_account_key.json"
service_token_path = "../../config/token.json"


def get_drive_service():
    creds = None
    token_path = service_token_path
    # (1) 저장된 토큰 파일(token.json)이 있으면 불러오기
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    # (2) 토큰이 없거나 유효하지 않으면 갱신하거나 새로 인증
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            # refresh_token이 있으면 자동으로 갱신
            creds.refresh(Request())
        else:
            creds = Credentials.from_json_keyfile_name(service_account_key_path, SCOPES)
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
    return service


def select_shared_drive(service):
    """사용자가 공유 드라이브를 선택하도록 함"""
    drives = service.drives().list(pageSize=100).execute().get("drives", [])
    if not drives:
        print("공유 드라이브가 없습니다.")
        return None

    print("사용 가능한 공유 드라이브:")
    for i, d in enumerate(drives):
        print(f"{i + 1}: {d['name']} (ID: {d['id']})")

    choice = int(input("선택할 드라이브 번호를 입력하세요: ")) - 1
    if 0 <= choice < len(drives):
        return drives[choice]["id"]
    else:
        print("잘못된 선택입니다.")
        return None


# 사용 예시
if __name__ == "__main__":
    drive_service = get_drive_service()
    drive_id = select_shared_drive(drive_service)
