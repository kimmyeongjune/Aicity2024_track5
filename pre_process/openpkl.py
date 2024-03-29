import pickle
import os
import re

### submission할 one txt file 만드는 코드~~~~


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

data_path = "./data/aicity/test/images"

# .pkl 파일 경로 지정
pkl_file_path = './outputs/last_troi/last_troi.pkl'

one_txt_dir = './outputs/Troi_original.txt'

# .pkl 파일 열기 및 내용 읽기
with open(pkl_file_path, 'rb') as file:
    data = pickle.load(file)

det_bboxes = data['det_bboxes'] # 19,972개의 test image -> len(det_bboxes) = 19,972

frame_counts_before_nth_video = []
frame_counts_per_video = []
frame_counts = 0
# data_path 내의 모든 폴더를 순회

folder_names = sorted(os.listdir(data_path), key=natural_sort_key)

for folder_name in folder_names:
    # 각 폴더의 전체 경로
    folder_path = os.path.join(data_path, folder_name)
    
    # 폴더 내의 파일들을 순회하며 jpg 파일만 세기
    if os.path.isdir(folder_path):
        count = len([file for file in os.listdir(folder_path) if file.endswith('.jpg')])
        frame_counts_before_nth_video.append(frame_counts)
        frame_counts_per_video.append(count)
        frame_counts += count

classes_num = len(det_bboxes[0])

with open(one_txt_dir, 'w') as file:
    for i in range(len(folder_names)):
        current_det_bboxes = det_bboxes[frame_counts_before_nth_video[i]:frame_counts_before_nth_video[i]+frame_counts_per_video[i]] # (200, 9)
        for j in range(len(current_det_bboxes)):
            for k in range(classes_num):
                for proposal in current_det_bboxes[j][k]:
                    bbox_coord, score = proposal[:4], proposal[4]
                    left, top, w, h = bbox_coord[0], bbox_coord[1], bbox_coord[2]-bbox_coord[0], bbox_coord[3]-bbox_coord[1]
                    if score > 0.5:
                        file.write(f"{i+1},{j+1},{left},{top},{w},{h},{k+1},{score}\n")



