import cv2
import numpy as np
import argparse
import glob
import os
    
def load_video(args):
    videos = glob.glob(args.data_path + '/*.mp4')
    for video in videos:
        save_dir = os.path.join(args.save_path, video.split('/')[-1].replace('.mp4',''))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        extract_frame(video, save_dir)

def extract_frame(video_path, save_path):
    extract_video = cv2.VideoCapture(video_path)
    frame_count = 0
    print(video_path)
    while True:
        success, image = extract_video.read()
        if success:
            image_name = os.path.join(save_path, str(frame_count))
            cv2.imwrite(f'{image_name}.jpg', image)
            frame_count += 1
        else:
            break
    extract_video.release()

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path'
    )
    parser.add_argument(
        '--save_path'
    )

    args = parser.parse_args()
    load_video(args)
