# 베이스: 가벼운 Python 3.11
FROM python:3.11-slim

# 시스템 기본 도구
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 워킹 디렉터리
WORKDIR /app

# 의존성 먼저 복사/설치(캐시 활용)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 앱 소스
COPY . /app

# uvicorn 실행 (개발: --reload는 compose에서 지정)
ENV PYTHONUNBUFFERED=1

# 컨테이너 기본 포트
EXPOSE 8000

# 기본 커맨드(생산용). 개발에서는 compose가 --reload로 override.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
