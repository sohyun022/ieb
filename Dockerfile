# vllm 기본 이미지를 베이스로 사용
FROM vllm/vllm-openai:v0.10.0
# 작업 디렉터리 설정
WORKDIR /app

# script.py 파일을 컨테이너로 복사
COPY ./app/requirement.txt /app/requirement.txt
RUN pip install -r requirement.txt


COPY ./app /app

# 컨테이너 시작 시 script.py를 실행하도록 설정
ENTRYPOINT ["bash", "-c"]      # 인터프리터만 고정