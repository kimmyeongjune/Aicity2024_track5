import os
from tqdm import tqdm  # tqdm 라이브러리 임포트

def sync_frames_and_annotations(frame_dir, annotation_dir):
    # 프레임 파일과 어노테이션 파일의 기본 이름 목록 생성
    frame_files = {f.split('.')[0] for f in os.listdir(frame_dir) if f.endswith('.jpg')}  # 확장자에 따라 조정 필요
    annotation_files = {f.split('.')[0] for f in os.listdir(annotation_dir) if f.endswith('.txt')}
    
    # 첫 번째 프레임 제목에 없는 어노테이션 파일 삭제
    to_delete = [annotation for annotation in annotation_files if annotation not in frame_files]
    for annotation in tqdm(to_delete, desc="Deleting unused annotations"):
        annotation_path = os.path.join(annotation_dir, f"{annotation}.txt")
        os.remove(annotation_path)
        print(f"Deleted annotation: {annotation_path}")
    
    # 어노테이션 제목에 없는 이미지 제목에 대한 빈 어노테이션 파일 생성
    to_create = [frame for frame in frame_files if frame not in annotation_files]
    for frame in tqdm(to_create, desc="Creating empty annotation files"):
        empty_annotation_path = os.path.join(annotation_dir, f"{frame}.txt")
        with open(empty_annotation_path, 'w') as f:  # 빈 파일 생성
            pass  # 파일 내용은 비워둠
        print(f"Created empty annotation file: {empty_annotation_path}")

# 예시 사용
frame_dir = './data/aicity/val/imgs'  # 프레임 이미지 폴더 경로를 지정하세요
annotation_dir = './data/aicity/val/anns'  # YOLO 어노테이션 파일 폴더 경로를 지정하세요

sync_frames_and_annotations(frame_dir, annotation_dir)
