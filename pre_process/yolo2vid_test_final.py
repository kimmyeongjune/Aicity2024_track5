import os
import cv2
import json
from tqdm import tqdm
import argparse
import re

# ArgumentParser를 사용하여 커맨드라인 인자를 파싱
parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', default='./data/aicity/test/', type=str, help="root path of images and labels, include ./images and classes.txt")
parser.add_argument('--save_path', type=str, default='./data/aicity/test', help="if not split the dataset, give a path to a json file")
parser.add_argument('--save_name', type=str, default='cocovid_test_fast.json', help="if not split the dataset, give a path to a json file without annotations")
args = parser.parse_args()

# 자연스러운 정렬을 위한 키 함수
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

# 비디오 정보 추출 함수
def extract_video_info(filename):
    match = re.match(r"Video_(\d+)", filename)
    if match:
        return match.group(1)
    return None

# YOLO 형식에서 COCOVID 형식으로 변환하는 함수(어노테이션 제외)
def yolo2cocovid(args):
    print("Loading data from ", args.root_dir)

    assert os.path.exists(args.root_dir)
    origin_images_dir = os.path.join(args.root_dir, 'fast_image')
    with open(os.path.join('./data/aicity/aicity2024_track5_train/classes_9.txt')) as f:
        classes = f.read().strip().split('\n')

    # 어노테이션을 제외한 데이터셋 구조 생성
    dataset = {'categories': [], 'videos': [], 'images': []}
    for i, cls in enumerate(classes):
        dataset['categories'].append({'id': i, 'name': cls})

    img_id = 1
    video_folders = sorted(filter(lambda x: os.path.isdir(os.path.join(origin_images_dir, x)), os.listdir(origin_images_dir)), key=natural_sort_key)

    for video_folder in video_folders:
        video_id = extract_video_info(video_folder)
        if video_id and int(video_id) not in [vid['id'] for vid in dataset['videos']]:
            dataset['videos'].append({'id': int(video_id), 'name': f"Video_{video_id}"})

        video_path = os.path.join(origin_images_dir, video_folder)
        images = sorted(os.listdir(video_path), key=natural_sort_key)
        frame_id = 0
        for image_file in images:
            img_path = os.path.join(video_path, image_file)
            im = cv2.imread(img_path)
            if im is None:
                print(f"Warning: Failed to load image at {img_path}")
                continue
            height, width, _ = im.shape

            # 이미지 정보만 포함하여 dataset에 추가
            dataset['images'].append({
                'file_name': os.path.join(video_folder, image_file),
                'height': height,
                'width': width,
                'id': img_id,
                'video_id': int(video_id),
                'frame_id': frame_id
            })

            img_id += 1
            frame_id += 1

    # 결과 JSON 파일 저장
    os.makedirs(args.save_path, exist_ok=True)
    json_name = os.path.join(args.save_path, args.save_name)
    with open(json_name, 'w') as f:
        json.dump(dataset, f, indent=4)
        print('Save annotation to', json_name)

if __name__ == "__main__":
    yolo2cocovid(args)
