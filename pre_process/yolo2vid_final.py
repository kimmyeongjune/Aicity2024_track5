import os
import cv2
import json
from tqdm import tqdm
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', default='./data/aicity/train/', type=str, help="root path of images and labels, include ./images and ./labels and classes.txt")
parser.add_argument('--save_path', type=str, default='./data/aicity/train', help="if not split the dataset, give a path to a json file")
parser.add_argument('--save_name', type=str, default='cocovid_train.json', help="if not split the dataset, give a path to a json file")
args = parser.parse_args()

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def extract_video_info(filename):
    match = re.match(r"Video_(\d+)", filename)
    if match:
        return match.group(1)
    return None

def yolo2cocovid(args):
    print("Loading data from ", args.root_dir)

    assert os.path.exists(args.root_dir)
    origin_labels_dir = os.path.join(args.root_dir, 'labels')
    origin_images_dir = os.path.join(args.root_dir, 'images')
    with open(os.path.join('./data/aicity/aicity2024_track5_train/classes_9.txt')) as f:
        classes = f.read().strip().split('\n')

    dataset = {'categories': [], 'videos': [], 'images': [], 'annotations': []}
    for i, cls in enumerate(classes):
        dataset['categories'].append({'id': i , 'name': cls})

    img_id = 1
    ann_id_cnt = 1
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

            dataset['images'].append({
                'file_name': os.path.join(video_folder, image_file),
                'height': height,
                'width': width,
                'id': img_id,
                'video_id': int(video_id),
                'frame_id': frame_id
            })

            txt_file = image_file.replace('.jpg', '.txt')
            label_path = os.path.join(origin_labels_dir, video_folder, txt_file)
            if os.path.exists(label_path):
                with open(label_path, 'r') as fr:
                    label_list = fr.readlines()
                    for label in label_list:
                        label = label.strip().split()
                        x, y, w, h = map(float, label[1:])
                        x1 = (x - w / 2) * width
                        y1 = (y - h / 2) * height
                        bbox_width = w * width
                        bbox_height = h * height
                        cls_id = int(label[0]) 

                        dataset['annotations'].append({
                            'id': ann_id_cnt,
                            'image_id': img_id,
                            'video_id': int(video_id),
                            'category_id': cls_id,
                            'bbox': [x1, y1, bbox_width, bbox_height],
                            'area': bbox_width * bbox_height,
                            "is_vid_train_frame": True
                        })
                        ann_id_cnt += 1
            img_id += 1
            frame_id += 1

    os.makedirs(args.save_path, exist_ok=True)
    json_name = os.path.join(args.save_path, args.save_name)
    with open(json_name, 'w') as f:
        json.dump(dataset, f, indent=4)
        print('Save annotation to', json_name)

if __name__ == "__main__":
    yolo2cocovid(args)
