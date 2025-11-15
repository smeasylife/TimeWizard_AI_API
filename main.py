import os
import json
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.concurrency import run_in_threadpool
import base64
import google.generativeai as genai
import requests
import mimetypes
from urllib.parse import urlparse
import pathlib

# ------------------------------------
# 1. Pydantic 요청 모델 정의
# ------------------------------------
class TimetableRequest(BaseModel):
    depart: str = Field(description="학과")
    grade: int = Field(description="학년")
    semester: int = Field(description="학기")
    goal_credit: int = Field(description="목표 학점")
    max_credit: int = Field(description="최대 학점")
    requirement: str = Field(description="요구 사항")
    curriculum_csv_url: str = Field(
        description="수강편람 CSV 파일 URL"
    )
    curriculum_image_url: str = Field(
        default="https://site.hanyang.ac.kr/documents/11050741/13154841/이수체계도(로봇공학과).png?t=1684909574755", 
        description="커리큘럼 이미지 (PNG/JPG) URL"
    )

# ------------------------------------
# 2. Gemini API 호출 함수
# ------------------------------------
def get_timetable_json(request: TimetableRequest) -> dict:
    # API 키 설정
    # api_key = os.getenv("GEMINI_API_KEY")
    api_key = "AIzaSyAiMQjjDCnxkynx49pos9MvPuqBo_Yo8Uk"
    if not api_key:
        raise ValueError("서버에 GEMINI_API_KEY가 설정되지 않았습니다.")

    genai.configure(api_key=api_key)

    # 프롬프트 내용 생성
    contents = []

    # 1. 학생 정보
    user_prompt_text = f"""
제공된 수강편람: (아래 CSV 데이터 참고)
제공된 학과 커리큘럼: (첨부된 이미지 참고)
제공된 학생 정보:
  학년: {request.grade}학년
  학기: {request.semester}학기
  학부/전공: {request.depart}
  요청 사항: {request.requirement}
  목표 학점: {request.goal_credit}
  최대 학점: {request.max_credit}
"""
    contents.append(user_prompt_text)
    print("1. 사용자 정보 생성")

    # 2. 수강편람 CSV 파일
    if request.curriculum_csv_url:
        try:
            print(f"Downloading CSV from: {request.curriculum_csv_url}...")
            response = requests.get(request.curriculum_csv_url)
            response.raise_for_status()

            try:
                csv_text_content = response.content.decode('cp949')
                print("   ... CSV: 'cp949'로 디코딩")
            except UnicodeDecodeError:
                csv_text_content = response.content.decode('utf-8')
                print("   ... CSV: 'utf-8'로 디코딩")

            contents.append("--- 수강편람 CSV 데이터 시작 ---")
            contents.append(csv_text_content)
            contents.append("--- 수강편람 CSV 데이터 끝 ---")
            print("2. 수강편람 DB 추가")

        except Exception as e:
            print(f"CSV 다운로드 또는 처리 실패: {e}")
            contents.append("--- (오류: 수강편람 CSV 파일을 불러오지 못했습니다) ---")
    else:
        print("수강편람 CSV URL이 제공되지 않아 건너뜁니다.")
        contents.append("--- (수강편람 CSV가 제공되지 않았습니다) ---")

    # 3. 커리큘럼 이미지 파일
    image_part = None
    if request.curriculum_image_url:
        try:
            # URL 인코딩 처리 (한글 파일명 등을 위해)
            from urllib.parse import urlparse, quote, urlunparse
            parsed_url = urlparse(request.curriculum_image_url)
            encoded_path = quote(parsed_url.path, safe='/')
            encoded_url = urlunparse((
                parsed_url.scheme,
                parsed_url.netloc,
                encoded_path,
                parsed_url.params,
                parsed_url.query,
                parsed_url.fragment
            ))

            print(f"Downloading Image from: {request.curriculum_image_url}...")
            response = requests.get(encoded_url)
            response.raise_for_status()
            image_data = response.content

            image_path = parsed_url.path
            image_ext = pathlib.Path(image_path).suffix
            image_mime_type, _ = mimetypes.guess_type(f"file{image_ext}")

            if not image_mime_type:
                if b'\x89PNG' in image_data[:8]:
                    image_mime_type = 'image/png'
                elif b'JFIF' in image_data[6:10] or b'Exif' in image_data[6:10]:
                    image_mime_type = 'image/jpeg'
                else:
                    raise ValueError("이미지 MIME 타입을 감지할 수 없습니다.")
                print(f"   ... 내용 기반 MIME 타입: '{image_mime_type}'")
            else:
                print(f"   ... URL 확장자 기반 MIME 타입: '{image_mime_type}'")

            image_part = {
                'mime_type': image_mime_type,
                'data': image_data
            }
            print("3. 커리큘럼 이미지를 추가")

        except Exception as e:
            print(f"이미지 다운로드 또는 처리 실패: {e}")
            contents.append("--- (오류: 커리큘럼 이미지를 불러오지 못했습니다) ---")
    else:
        print("커리큘럼 이미지 URL이 제공되지 않아 건너뜁니다.")
        contents.append("--- (커리큘럼 이미지가 제공되지 않았습니다) ---")

    # 시스템 지시문
    system_instruction = """당신의 임무는 대학생의 시간표를 작성하는 것입니다.
시간표를 생성하면서 아래의 원칙을 지켜야 합니다.

1. 반드시 제공되는 CSV 수강 편람 데이터에 기반하여 실제 수강편람과 완벽히 맞는 데이터로 구성해야 합니다. (없는 강의를 넣거나, 수강 편람과는 시간이나 정보를 다르게 하여서 넣으면 절대 안됩니다. 특히 course_id는 CSV의 course_id 컬럼 값을 정확히 복사해야 합니다.)
2. 수강 과목 사이에 시간이 겹칠 수는 없습니다.
3. 이후 제시되는 커리큘럼 이미지, 사용자의 추가 요청 사항, 목표 학점을 최대한 반영하여야 합니다.
4. ai_comment 부분에서는 왜 이렇게 시간표를 짰는지의 합당한 이유, 추가 설명이 필요한 부분을 친절하게 한국어로 500자 내외로 설명합니다. 포맷은 plaintext입니다. 단, comment 내에서는 당신의 작업 내부 사항을 노출하면 안됩니다. (예를 들어, csv로 제공 받은 수강편람 데이터에서 자료가 부족할 경우 '추가 CSV 데이터가 제공되어야 합니다' 라는 설명은 절대 안됩니다 - 대신 '현재 제공된 편람 자료가 부족합니다' 같은 식으로 사용자단에서 이해가 되도록 설명해야 합니다)
5. 시간표를 짜면서 기본적으로 지켜져야 하는 부분은 다 준수한다고 생각하면 됩니다.
6. 단, 사용자의 요청 사항으로는 수행이 어려울 경우 후순위로 미룰수 있습니다. (최적의 시간표의 작성이 우선입니다)
7. 목표 학점보다 미달되어도 시간표는 완성합니다. 단, 최대 학점은 초과하면 안됩니다.
8. '시간미지정' 강의(강좌)는 사이버/비대면 수업으로, 시간을 고려하지 않고 추가합니다. (각 courseTime의 dayOfWeek를 "NONE", startTime과 endTime을 0으로 설정합니다)
9. 교양필수, 전공필수가 커리큘럼에 있다면 최우선으로 추가해야 합니다. 단, 요청 사항에 배제하라는 요청이 있는 경우 무시합니다.
10. **매우 중요** 응답값의 course_id는 반드시 CSV 파일의 첫 번째 컬럼인 "course_id" 값(1, 2, 3... 같은 순차 정수)을 사용해야 합니다.
11. 출력에는 오직 'JSON' 데이터만 담아야 하며, 제공되는 'Structured Output'을 **무조건** 준수해야 합니다.

**중요: CSV 컬럼 순서 및 의미**
CSV의 각 행은 다음 순서로 구성됩니다:
[0]course_id → 1,2,3,4,5... (시스템 고유 ID - 이것을 응답에 사용!)
[1]course_name → 20001,20004,22498,23715... (수업번호 - 절대 사용 금지!)
[2]course_code → AIX0002,AIX0005... (학수번호)
[3]subject_name → "AI+X:인공지능", "로봇프로그래밍"... (과목명 - 응답의 course_name에 사용)
[4]subject_name_eng → 영문 과목명
...
[12]professor → 교수명
[13]class_time → 수업 시간

**매우 중요: class_time 필드 파싱 및 시간 충돌 확인**
CSV의 class_time 필드는 쉼표(,)로 구분된 여러 시간대를 포함할 수 있습니다.
- 예시 1: "목(09:00-10:30),목(10:30-11:00)" → 목요일에 두 개의 연속된 시간대
- 예시 2: "목(13:00-15:00),금(13:00-14:30)" → 목요일과 금요일에 각각 한 시간대
- 예시 3: "화(14:00-15:00),화(15:00-17:00)" → 화요일에 두 개의 연속된 시간대

**반드시 지켜야 할 규칙:**
1. class_time의 쉼표로 구분된 모든 시간대를 파싱하여 응답의 courseTimes 배열에 포함시켜야 합니다.
2. 시간 충돌 검사 시, 각 강의의 모든 courseTimes를 고려하여 확인해야 합니다.
3. 한 강의의 어떤 courseTime이라도 다른 강의의 courseTime과 겹치면 안 됩니다.
4. 요일은 한글 약자(월, 화, 수, 목, 금, 토, 일)를 영문 대문자(MON, TUE, WED, THU, FRI, SAT, SUN)로 변환합니다.
5. 시간은 "HH:MM" 형식을 분 단위(HH*60 + MM)로 변환합니다. 예: "09:00" → 540, "14:30" → 870

**잘못된 예 (절대 금지!):**
CSV: class_time = "목(13:00-15:00),금(13:00-14:30)"
❌ 잘못된 응답: 첫 번째 시간대만 포함
{
  "courseTimes": [
    {"dayOfWeek": "THU", "startTime": 780, "endTime": 900}
  ]
}

**올바른 예:**
CSV 행: "359,20515,GEE2054,학술영어2:글쓰기,Academic English 2: Writing,...,Douglas James Scott,목(13:00-15:00),금(13:00-14:30),..."
올바른 응답:
{
  "course_id": "359",
  "course_name": "학술영어2:글쓰기",
  "professor": "Douglas James Scott",
  "courseTimes": [
    {
      "dayOfWeek": "THU",
      "startTime": 780,      ← 13*60 = 780
      "endTime": 900          ← 15*60 = 900
    },
    {
      "dayOfWeek": "FRI",
      "startTime": 780,      ← 13*60 = 780
      "endTime": 870          ← 14*60 + 30 = 870
    }
  ]
}

**반드시 기억:**
- course_id 응답값 = CSV의 [0]번째 컬럼 (1, 2, 3, 359, 5...)
- course_name 응답값 = CSV의 [3]번째 컬럼 (subject_name)
- [1]번째 컬럼(20001, 22498, 23715...)은 절대 사용하지 마세요!
- class_time의 모든 시간대를 파싱하여 courseTimes 배열에 포함시키세요!
- 시간 충돌 검사 시 모든 courseTimes를 고려하세요!
- dayOfWeek는 반드시 대문자로 작성하세요 (MON, TUE, WED, THU, FRI, SAT, SUN)!
"""

    # 모델 생성 및 설정
    generation_config = {
        "temperature": 0.5,
        "response_mime_type": "application/json",
        "response_schema": {
            "type": "object",
            "required": ["courses", "ai_comment"],
            "properties": {
                "courses": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["course_id", "course_name", "professor", "courseTimes"],
                        "properties": {
                            "course_id": {"type": "string"},
                            "course_name": {"type": "string"},
                            "professor": {"type": "string"},
                            "courseTimes": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "required": ["dayOfWeek", "startTime", "endTime"],
                                    "properties": {
                                        "dayOfWeek": {
                                            "type": "string",
                                            "enum": ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN", "NONE"]
                                        },
                                        "startTime": {"type": "integer"},
                                        "endTime": {"type": "integer"}
                                    }
                                }
                            }
                        }
                    }
                },
                "ai_comment": {"type": "string"}
            }
        }
    }

    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        },
    ]

    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        generation_config=generation_config,
        safety_settings=safety_settings,
        system_instruction=system_instruction
    )

    # 콘텐츠 구성 (이미지가 있으면 포함)
    final_contents = contents.copy()
    if image_part:
        final_contents.append(image_part)


    # API 호출
    print("Gemini API 호출 시작...")
    start_time = time.perf_counter()

    response = model.generate_content(final_contents)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print("Gemini API 응답 수신 완료.")
    print(f"⏱️  응답 시간: {elapsed_time:.2f}초")

    # 토큰 사용량 추출
    usage_info = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0
    }

    if hasattr(response, 'usage_metadata'):
        usage = response.usage_metadata
        usage_info["input_tokens"] = usage.prompt_token_count
        usage_info["output_tokens"] = usage.candidates_token_count
        usage_info["total_tokens"] = usage.total_token_count

        print(f"📊 토큰 사용량:")
        print(f"   - 입력 토큰: {usage_info['input_tokens']}")
        print(f"   - 출력 토큰: {usage_info['output_tokens']}")
        print(f"   - 총 토큰: {usage_info['total_tokens']}")

    # 메타데이터와 함께 반환
    return {
        "timetable_data": response.text,
        "metadata": {
            "elapsed_time_seconds": round(elapsed_time, 2),
            "usage": usage_info
        }
    }

# ------------------------------------
# 3. FastAPI 앱 및 엔드포인트 생성
# ------------------------------------
app = FastAPI(
    title="TimeWizard AI Timetable Generator",
    description="Gemini AI를 사용하여 대학생 시간표를 생성하는 API입니다."
)

@app.post("/generate-timetable")
async def create_timetable(request: TimetableRequest):
    """
    사용자로부터 시간표 생성 요청을 받아 Gemini AI를 호출하고,
    생성된 시간표 JSON을 반환합니다.
    """
    try:
        print("요청 처리 시작")
        result = await run_in_threadpool(get_timetable_json, request)
        print("API 응답 파싱 중")

        # response.text는 JSON 문자열이므로 파싱
        timetable_data = json.loads(result["timetable_data"])

        # 메타데이터는 print만 (응답에 포함하지 않음)
        metadata = result["metadata"]
        print(f"⏱️  총 처리 시간: {metadata['elapsed_time_seconds']}초")
        print(f"📊 토큰 사용량: 입력 {metadata['usage']['input_tokens']}, "
              f"출력 {metadata['usage']['output_tokens']}, "
              f"총 {metadata['usage']['total_tokens']}")
        print("응답 완료")

        return timetable_data

    except Exception as e:
        print(f"API 처리 중 오류 발생: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"AI 시간표 생성 중 오류가 발생했습니다: {str(e)}"
        )

@app.get("/")
async def root():
    return {"message": "TimeWizard AI Timetable Generator API", "status": "running"}