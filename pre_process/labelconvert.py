import os

def rename_files(base_dir, start_video=1, end_video=100, start_frame=1, end_frame=200):
    # Video_001부터 Video_100까지 순회
    for video_idx in range(start_video, end_video + 1):
        video_folder = os.path.join(base_dir, f'Video_{video_idx:03d}')
        # 각 비디오 폴더 내의 파일 이름 변경
        for frame_idx in range(start_frame, end_frame + 1):
            old_filename = os.path.join(video_folder, f'{video_idx:03d}_frame{frame_idx}.jpg')
            new_filename = os.path.join(video_folder, f'{video_idx:03d}_{frame_idx}.jpg')
            # 파일 이름 변경
            if os.path.exists(old_filename):
                os.rename(old_filename, new_filename)
                print(f'Renamed: {old_filename} -> {new_filename}')
            else:
                print(f'File not found: {old_filename}')

# 사용 예시
base_dir = './data/aicity/train/images/'  # 기본 디렉토리 경로를 적절히 설정하세요.
rename_files(base_dir)
