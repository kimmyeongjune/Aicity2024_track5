import cv2
import os
from tqdm import tqdm  # tqdm 라이브러리 임포트

# Ground Truth 정보를 파싱하는 함수
def parse_gt_file(gt_file_path):
    gt_data = {}
    with open(gt_file_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split(',')
            if len(parts) >= 8:  # 예상되는 파라미터 개수 확인
                video_id, frame, *rest = parts
                if video_id not in gt_data:
                    gt_data[video_id] = {}
                gt_data[video_id][int(frame)] = rest
    return gt_data

# 비디오 처리 및 프레임 추출 함수
def process_video(video_path, gt_data, frames_dir, target_fps=10):
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        print(f"Could not open: {video_path}")
        return
    
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_interval = round(fps / target_fps)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # 전체 프레임 수를 얻습니다.
    
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    frame_count = 0
    saved_frame_index = 0
    with tqdm(total=total_frames, desc=f"Processing {video_id}") as pbar:  # tqdm으로 진행 상황 바를 설정
        while True:
            ret, frame = video.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frame_file_path = os.path.join(frames_dir, f"{video_id}_{saved_frame_index + 1}.jpg")
                cv2.imwrite(frame_file_path, frame)
                saved_frame_index += 1
            
            frame_count += 1
            pbar.update(1)  # 진행 상황 바를 업데이트
    
    video.release()

# 메인 코드
directory_path = './data/aicity/aicity2024_track5_train'
gt_file_path = os.path.join(directory_path, 'gt.txt')
frames_dir = os.path.join('./data/aicity/crop_data', 'frames')  # 프레임을 저장할 폴더 위치
gt_data = parse_gt_file(gt_file_path)

# 디렉토리 내 모든 비디오 파일 처리
for filename in tqdm(os.listdir(os.path.join(directory_path, 'videos')), desc="Iterating over videos"):
    if filename.endswith(".mp4"):
        video_path = os.path.join(directory_path, 'videos', filename)
        process_video(video_path, gt_data, frames_dir, target_fps=10)
