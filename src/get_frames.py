import cv2

output_folder = '/detectron-api/data/frames'
output_file = '/detectron-api/data/frames.txt'
video_folder = '/detectron-api/data/'
video_name = 'out1080.mp4'

import os
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

print("Extracting frames...")
vidcap = cv2.VideoCapture(video_folder + video_name)
success,image = vidcap.read()
count = 0
with open(output_file, 'a') as file:
    while success:
        print("%s/%s-%s.jpg" % (output_folder, video_name, str(count).zfill(5)))
        cv2.imwrite("%s/%s-%s.jpg" % (output_folder, video_name, str(count).zfill(5)), image)
        file.write("%s-%s.jpg \n" % (video_name, str(count).zfill(5)))
        success,image = vidcap.read()
        count += 1