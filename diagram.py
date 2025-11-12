from diagrams import Diagram
from diagrams.aws.compute import EC2
from diagrams.aws.database import RDS
from diagrams.onprem.client import Client
from diagrams.onprem.inmemory import Redis
from diagrams.programming.framework import Fastapi

with Diagram("My System Architecture", show=False, direction="LR"):
    # 프론트엔드 (외부)
    frontend = Client("Frontend")

    # EC2 메인 서버 (백엔드)
    backend = EC2("Backend\n(EC2 Main Server)")

    # EC2에서 함께 실행되는 서비스들
    redis = Redis("Redis")
    fastapi = Fastapi("AI Model\n(FastAPI)")

    # RDS
    database = RDS("RDS Database")

    # 연결 관계
    frontend >> backend
    backend >> redis
    backend >> fastapi
    backend >> database