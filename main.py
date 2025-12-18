import os
import json
import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
import json
from typing import Dict, Any
import base64
import google.generativeai as genai
import requests
import mimetypes
from urllib.parse import urlparse
import pathlib

# ------------------------------------
# 1. Pydantic 요청 모델 정의
# ------------------------------------
class RequirementAnalysisRequest(BaseModel):
    requirement: str = Field(description="사용자의 요구 사항 및 선호사항")

class CourseTimeDTO(BaseModel):
    dayOfWeek: str  # 요일 (MON, TUE, WED 등)
    startTime: int  # 시작 시간 (분 단위)
    endTime: int  # 종료 시간 (분 단위)
    classroom: str | None = None  # 강의실 위치

class CourseTime(BaseModel):
    day: str
    start_time: int
    end_time: int

class CourseData(BaseModel):
    course_id: int
    course_name: str
    professor: str | None = None
    credit: int
    course_type: str  # 이수구분
    course_times: list[CourseTime] = Field(description="한 수업에 있는 여러 수업 요일과 시간")

class CoursePriorityData(BaseModel):
    courseId: int  # courseId
    courseCode: str | None = None  # courseCode
    courseName: str  # courseName
    courseEnglishName: str | None = None  # courseEnglishName
    courseNumber: int | None = None  # courseNumber
    professor: str | None = None  # professor
    major: str | None = None  # major
    section: int | None = None  # section
    grade: int | None = None  # grade (학년)
    credits: int  # credits (학점)
    lectureHours: int | None = None  # lectureHours
    practiceHours: int | None = None  # practiceHours
    courseType: str  # courseType (이수구분)
    capacity: int | None = Field(description="수강 정원")  # capacity
    semester: str | None = None  # semester
    courseTimes: list[CourseTimeDTO] = Field(description="한 수업에 있는 여러 수업 요일과 시간")  # courseTimes
  
class CoursePriorityRequest(BaseModel):
    courses: list[CoursePriorityData] = Field(description="우선순위를 정할 수업 데이터 목록")

class CoursePriorityResponse(BaseModel):
    prioritized_courses: list[CoursePriorityData] = Field(description="우선순위가 정렬된 수업 데이터 목록")
    ai_comment: str = Field(description="전체 우선순위 결정 이유 (3문장 내외)")

class TimetableRequest(BaseModel):
    depart: str = Field(description="학과")
    grade: int = Field(description="학년")
    semester: int = Field(description="학기")
    goal_credit: int = Field(description="목표 학점")
    max_credit: int = Field(description="최대 학점")
    requirement: str = Field(description="요구 사항")
    courses: list[CourseData] = Field(description="필터링된 수업 데이터 목록")

# ------------------------------------
# 2. Gemini API 호출 함수
# ------------------------------------

# 2-1. 1단계: 요구사항 분석 함수
def analyze_requirements(requirement: str) -> dict:
    """
    사용자의 요구사항을 분석하여 필터링 조건을 추출합니다.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("서버에 GEMINI_API_KEY가 설정되지 않았습니다.")

    genai.configure(api_key=api_key)

    system_instruction = """당신의 임무는 대학생의 시간표 요구사항을 분석하여 필터링 조건을 JSON으로 추출하는 것입니다.

**분석해야 할 내용:**
1. 시간대 선호/기피 사항
   - "월요일 오전 없게", "수요일 오후 비워주세요" 등을 파싱
   - 오전: 00:00~12:00 (0~720분)
   - 오후: 12:00~18:00 (720~1080분)
   - 저녁: 18:00~24:00 (1080~1440분)

2. 특정 과목 포함/제외 요청
   - "사회봉사 꼭 넣어주세요" → includeCourses에 추가
   - "체육 제외" → excludeCourses에 추가
   -  수업에는 띄어쓰기가 없으니 띄어쓰기 무시하고 붙여쓰기. ex) "개인정보보호법률이해"

3. 요일 표현
   - all: 모든 요일
   - MON, TUE, WED, THU, FRI, SAT, SUN: 특정 요일

**출력 규칙:**
- excludeTimeBlocks: 기피 시간대 배열 (비워두고 싶은 시간)
- includeCourses: 반드시 포함해야 할 과목명 배열
- excludeCourses: 제외해야 할 과목명 배열
- 명확한 요청이 없으면 빈 배열로 설정
- 시간은 분 단위로 변환 (예: 9시 = 540분, 18시 = 1080분)

**예시:**
입력: "월요일은 오전 수업 없이 짜주세요. 수요일 오후는 비워주세요. 사회봉사는 꼭 넣어주세요."
출력:
{
  "excludeTimeBlocks": [
    { "day": "MON", "startTime": 0, "endTime": 720 },
    { "day": "WED", "startTime": 720, "endTime": 1080 }
  ],
  "includeCourses": ["사회봉사"],
  "excludeCourses": []
}
"""

    generation_config = {
        "temperature": 0.3,
        "response_mime_type": "application/json",
                "response_schema": {
            "type": "object",
            "required": ["excludeTimeBlocks", "includeCourses", "excludeCourses"],
            "properties": {
                "excludeTimeBlocks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["day", "startTime", "endTime"],
                        "properties": {
                            "day": {
                                "type": "string",
                                "enum": ["all", "MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
                            },
                            "startTime": {"type": "integer"},
                            "endTime": {"type": "integer"}
                        }
                    }
                },
                "includeCourses": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "excludeCourses": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        }
    }

    client = genai.GenerativeModel(
        model_name="gemini-3-flash-preview",
        generation_config=generation_config,
        system_instruction=system_instruction
    )

    user_prompt = f"""사용자의 요구사항:
{requirement}

위 요구사항을 분석하여 필터링 조건을 JSON으로 추출하세요."""

    print("1단계: 요구사항 분석 API 호출 중...")
    start_time = time.perf_counter()

    response = client.generate_content(user_prompt)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print("1단계: 응답 수신 완료")
    print(f"⏱️  응답 시간: {elapsed_time:.2f}초")

    # 토큰 사용량 추출
    if hasattr(response, 'usage_metadata'):
        usage = response.usage_metadata
        print(f"📊 토큰 사용량:")
        print(f"   - 입력 토큰: {usage.prompt_token_count}")
        print(f"   - 출력 토큰: {usage.candidates_token_count}")
        print(f"   - 총 토큰: {usage.total_token_count}")

    return json.loads(response.text)


# 2-1-2. 수업 우선순위 결정 함수
def prioritize_courses(courses: list[CoursePriorityData]) -> CoursePriorityResponse:
    """
    수업 데이터를 분석하여 신청 우선순위로 정렬합니다.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("서버에 GEMINI_API_KEY가 설정되지 않았습니다.")

    genai.configure(api_key=api_key)

    # 수업 데이터를 JSON으로 변환
    courses_json = json.dumps(
        [course.model_dump() for course in courses],
        ensure_ascii=False,
        indent=2
    )

    system_instruction = """당신의 임무는 제공된 수업 데이터를 분석하여 수강 신청 우선순위로 정렬하고, 전체 우선순위 결정 이유를 세 줄로 요약하는 것입니다.

**데이터 구조:**
각 수업은 다음 필드를 포함합니다:
- courseId: 수업 고유 ID
- courseCode: 과목 코드
- courseName: 과목명 (한글)
- courseEnglishName: 과목명 (영문)
- courseNumber: 과목번호
- professor: 교수명
- major: 개설학과
- grade: 학년
- credits: 학점
- courseType: 이수구분 (전공필수, 전공선택, 교양필수 등)
- capacity: 수강 정원
- courseTimes: 수업 시간 정보

**우선순위 결정 기준:**
1. **이수구분 중요도** (높은 순서):
   - 전공핵심/전공필수: 가장 높은 우선순위
   - 교양필수/기초교양: 높은 우선순위
   - 전공선택: 중간 우선순위
   - 교양선택: 일반 우선순위

2. **수강 신청 현황**:
   - capacity(정원)이 적은 과목은 조기 마감 위험이 있으므로 높은 우선순위
   - capacity가 없는 경우는 중간 순위로 취급

3. **학점 및 시간**:
   - 학점(credits)이 높은 수업(3학점 이상)은 조금 더 높은 우선순위 고려

**출력 규칙:**
다음 JSON 형식으로 응답해야 합니다:
{
  "prioritized_courses": [우선순위가 높은 순서대로 정렬된 수업 데이터 배열],
  "ai_comment": "우선순위 결정의 논리적 근거 (한국어 3문장 내외)"
}

### 지시 사항
출력한 '수강신청 TODO 리스트'의 우선순위 순서가 결정된 **논리적 근거**를 설명하십시오.

### 분석 기준
1. **예상 경쟁률:** 클릭 경쟁이 치열한 과목인가?
2. **강의 중요도:** 졸업 필수 요건 혹은 선수 과목인가?
3. **수용 인원:** 여석이 적어 조기 마감이 예상되는가?

### 출력 제약 조건
- **범위:** 모든 강의를 나열하지 말고, **최상위 우선순위**와 **최하위 우선순위**를 중심으로 대조하여 설명할 것.
- **분량:** 핵심만 요약하여 **한국어 3문장 내외**로 작성할 것.
"""

    generation_config = {
        "temperature": 0.2,
        "response_mime_type": "application/json",
        "thinking_level": "LOW"
    }

    client = genai.GenerativeModel(
        model_name="gemini-3-flash-preview",
        generation_config=generation_config,
        system_instruction=system_instruction
    )

    user_prompt = f"""수업 데이터:

{courses_json}

위 수업 데이터를 우선순위가 높은 순서대로 정렬하고, 우선순위 결정의 논리적 근거를 한국어 3문장 내외로 작성하여 ai_comment 필드에 포함해주세요.
최상위 우선순위와 최하위 우선순위 수업을 중심으로 경쟁률, 강의 중요도, 수용 인원을 대조하여 설명해주세요.
지정된 JSON 형식에 맞춰 응답해주세요."""

    print("수업 우선순위 분석 API 호출 중...")
    start_time = time.perf_counter()

    response = client.generate_content(user_prompt)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print("수업 우선순위 분석 응답 수신 완료")
    print(f"⏱️  응답 시간: {elapsed_time:.2f}초")

    # 토큰 사용량 추출
    if hasattr(response, 'usage_metadata'):
        usage = response.usage_metadata
        print(f"📊 토큰 사용량:")
        print(f"   - 입력 토큰: {usage.prompt_token_count}")
        print(f"   - 출력 토큰: {usage.candidates_token_count}")
        print(f"   - 총 토큰: {usage.total_token_count}")

    # JSON 응답을 CoursePriorityResponse 객체로 변환
    response_data = json.loads(response.text)

    # 정렬된 수업 데이터를 CoursePriorityData 객체 리스트로 변환
    prioritized_courses = [CoursePriorityData(**course) for course in response_data["prioritized_courses"]]

    # CoursePriorityResponse 객체 생성하여 반환
    return CoursePriorityResponse(
        prioritized_courses=prioritized_courses,
        ai_comment=response_data["ai_comment"]
    )


# 2-2. 2단계: 시간표 생성 함수
def get_timetable_json(request: TimetableRequest) -> dict:
    # API 키 설정
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("서버에 GEMINI_API_KEY가 설정되지 않았습니다.")

    genai.configure(api_key=api_key)

    # 프롬프트 내용 생성
    contents = []

    # 1. 학생 정보 및 수업 데이터
    courses_json = json.dumps(
        [course.model_dump() for course in request.courses],
        ensure_ascii=False,
        indent=2
    )

    user_prompt_text = f"""
  제공된 학생 정보:
  학년: {request.grade}학년
  학기: {request.semester}학기
  학부/전공: {request.depart}
  요청 사항: {request.requirement}
  목표 학점: {request.goal_credit}
  최대 학점: {request.max_credit}

--- 필터링된 수업 데이터 (JSON) ---
{courses_json}
--- 수업 데이터 끝 ---
"""
    contents.append(user_prompt_text)
    print(f"1. 사용자 정보 및 수업 데이터 생성 (총 {len(request.courses)}개 과목)")

    # 시스템 지시문 (간소화)
    system_instruction = """당신의 임무는 제공된 수업 데이터에서 최적의 시간표를 작성하는 것입니다.

**시간표 생성 원칙:**
1. 제공된 수업 데이터(JSON)만 사용 (없는 강의 생성 금지)
2. 수강 과목 간 시간 충돌 금지
3. 사용자의 요청 사항 최대한 반영 (단, 최적 시간표 우선)
4. 목표 학점 달성 노력, 최대 학점 초과 금지
5. 전공필수 > 교양필수 > 전공선택 우선순위
6. 시간미지정 강의는 시간 고려 없이 추가 (day="NONE", start_time=0, end_time=0)

**수업 데이터 구조:**
각 수업은 다음 필드를 가집니다:
- course_id: 시스템 고유 ID (응답에 그대로 사용)
- course_name: 과목명 (응답에 그대로 사용)
- professor: 교수명 (응답에 그대로 사용)
- credit: 학점
- course_type: 이수구분 (전공필수, 교양필수 등)
- course_times: 수업 시간 정보 배열 (이미 파싱됨)
  - 각 course_time 객체는 day(요일), start_time(시작 시간, 분 단위), end_time(종료 시간, 분 단위)를 포함
  - 예: [{"day": "THU", "start_time": 780, "end_time": 900}, {"day": "FRI", "start_time": 780, "end_time": 870}]

**course_times 사용 규칙:**
- 입력 데이터의 course_times를 그대로 사용하여 응답의 courseTimes 배열에 매핑
- 요일 변환: day 필드의 값을 dayOfWeek로 사용 (MON, TUE, WED, THU, FRI, SAT, SUN, NONE)
- 시간 변환: start_time → startTime, end_time → endTime (분 단위 그대로 사용)
- 한 수업에 여러 시간대가 있을 수 있음 (예: 목요일 2시간 + 금요일 1.5시간)

**시간 충돌 검사:**
- 각 강의의 모든 course_times를 고려하여 시간 충돌 확인
- 한 강의의 어떤 course_time도 다른 강의의 course_time과 겹치면 안 됨
- 같은 요일(day)에 시간대(start_time ~ end_time)가 겹치는지 확인

**ai_comment 작성:**
- 시간표 구성 이유를 500자 내외로 친절하게 설명
- 내부 작업 사항 노출 금지 (사용자 관점으로 작성)

**출력:**
- JSON만 출력, Structured Output 준수
"""

    # 모델 생성 및 설정
    generation_config = {
        "temperature": 0.3,
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

    client = genai.GenerativeModel(
        model_name="gemini-3-flash-preview",
        generation_config=generation_config,
        safety_settings=safety_settings,
        system_instruction=system_instruction
    )

    # API 호출
    print("2단계: 시간표 생성 API 호출 시작...")
    start_time = time.perf_counter()

    response = client.generate_content(contents)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print("2단계: 시간표 생성 응답 수신 완료")
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
# 3. FastAPI 앱 및 미들웨어 설정
# ------------------------------------
app = FastAPI(
    title="TimeWizard AI Timetable Generator",
    description="Gemini AI를 사용하여 대학생 시간표를 생성하는 API입니다."
)

@app.post("/analyze-requirements")
async def analyze_user_requirements(request: RequirementAnalysisRequest):
    """
    [1단계] 사용자의 요구사항을 분석하여 필터링 조건을 추출합니다.

    Returns:
        {
          "excludeTimeBlocks": [...],
          "includeCourses": [...],
          "excludeCourses": [...]
        }
    """
    try:
        print("=== 1단계: 요구사항 분석 시작 ===")
        result = await run_in_threadpool(analyze_requirements, request.requirement)
        print("=== 1단계: 응답 완료 ===")
        return result

    except Exception as e:
        print(f"요구사항 분석 중 오류 발생: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"요구사항 분석 중 오류가 발생했습니다: {str(e)}"
        )


@app.post("/generate-timetable")
async def create_timetable(request: TimetableRequest):
    """
    [2단계] 필터링된 수업 데이터로부터 최적의 시간표를 생성합니다.

    Args:
        - 학생 정보 (학과, 학년, 학기, 목표학점, 최대학점)
        - 필터링된 수업 데이터 목록 (courses)
        - 요구사항

    Returns:
        {
          "courses": [...],
          "ai_comment": "..."
        }
    """
    try:
        print("=== 2단계: 시간표 생성 시작 ===")
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
        print("=== 2단계: 응답 완료 ===")

        return timetable_data

    except Exception as e:
        print(f"시간표 생성 중 오류 발생: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"AI 시간표 생성 중 오류가 발생했습니다: {str(e)}"
        )


@app.post("/prioritize-courses")
async def create_course_priorities(request_data: list[CoursePriorityData]) -> CoursePriorityResponse:
    """
    수업 데이터를 우선순위 순서대로 정렬합니다.

    Args:
        - request_data: 우선순위를 정할 수업 데이터 목록 (CourseResponseDTO 형식의 배열)

    Returns:
        우선순위가 높은 순서대로 정렬된 수업 데이터 목록
    """
    try:
        print("=== 📋 수업 우선순위 정렬 시작 ===")

        # 요청 데이터 확인
        print(f"요청받은 수업 데이터 수: {len(request_data)}")
        if request_data:
            print("첫 번째 수업 데이터 샘플:")
            print(json.dumps(request_data[0].model_dump(), indent=2, ensure_ascii=False))

        result = await run_in_threadpool(prioritize_courses, request_data)
        print("=== ✅ 수업 우선순위 정렬 완료 ===")
        print(f"반환된 수업 데이터 수: {len(result.prioritized_courses)}")
        print(f"AI 코멘트: {result.ai_comment}")
        return result

    except Exception as e:
        print(f"❌ 수업 우선순위 정렬 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"수업 우선순위 정렬 중 오류가 발생했습니다: {str(e)}"
        )

# 기존의 CoursePriorityRequest 모델은 다른 용도로 사용할 수 있도록 유지


@app.get("/")
async def root():
    return {"message": "TimeWizard AI Timetable Generator API (3-Stage)", "status": "running"}