import os
import re
from collections import defaultdict
import shutil

def group_images_by_prefix_and_save(image_dir, target_dir):
    """
    지정된 디렉토리 내의 이미지 파일을 파일명이 시작하는 숫자에 따라 그룹화하고,
    각 그룹별로 별도의 폴더에 저장합니다.
    """
    # 그룹화된 이미지를 저장할 딕셔너리 초기화
    grouped_images = defaultdict(list)
    
    # 지정된 디렉토리의 모든 파일 목록을 순회
    for filename in os.listdir(image_dir):
        # 파일명에서 시작하는 숫자 추출
        match = re.match(r"(\d+)", filename)
        if match:
            prefix = match.group(1)  # 첫 번째 숫자 시퀀스
            # 해당 숫자로 시작하는 그룹에 파일명 추가
            grouped_images[prefix].append(filename)
    
    # 각 그룹별로 폴더 생성 및 파일 저장
    for prefix, files in grouped_images.items():
        # 저장할 그룹 폴더 경로
        group_dir = os.path.join(target_dir, f"Video_{prefix}")
        # 폴더 생성
        os.makedirs(group_dir, exist_ok=True)
        
        # 해당 그룹 폴더에 파일 복사
        for file in files:
            source_path = os.path.join(image_dir, file)
            target_path = os.path.join(group_dir, file)
            shutil.copy(source_path, target_path)
            print(f"Copied {file} to {group_dir}/")

# 이미지가 저장된 원본 디렉토리 경로
image_dir = './data/aicity/test/images'

# 그룹별 이미지를 저장할 대상 디렉토리 경로
target_dir = './data/aicity/grouped_frames_test/'

# 이미지 파일 그룹화 및 저장 실행
group_images_by_prefix_and_save(image_dir, target_dir)
