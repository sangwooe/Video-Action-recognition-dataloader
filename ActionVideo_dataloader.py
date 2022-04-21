import numpy as np
import random
import cv2
import glob
import os

'''
    <directory shape>
        - path_to_dataset
            --> action_clssses1
                -->video1
                    -->frame1.jpg
                    -->frame2.jpg
                    --> ...
                    -->frameN.jpg
                -->video2
                    -->frame1.jpg
                    -->frame2.jpg
                    --> ...
                    -->frameN.jpg

            --> action_classes2
            --> action_classes3
'''


class action_dataset(Dataset):
    
    '''
    Data Argumentation : Random crop, horizontal flip
    Input : Every frames in the videos
    Ouptut : [ number of segment, number of frame in the segment, image height, image width, channel ]

    The last frames not composed in a segment are deleted.
    '''

    def __init__ (self, root_dir, img_size = (256,256), crop_size=(224,224), crop=False, flip=True, flip_prob=0.5, frame_len=16):

        self.root_dir = root_dir
        self.img_width, self.img_height = img_size[0], img_size[1]
        self.crop_width, self.crop_height = crop_size[0], crop_size[1]
        self.frame_len = frame_len
        self.label_list = os.listdir(self.root_dir)
        self.video_list = glob.glob(self.root_dir+'/*/*')
        self.crop = crop
        self.flip = flip
        self.flip_prob = flip_prob

    def load_image(self, image_path):

        img = cv2.imread(image_path)
        img = cv2.resize(img, dsize=(self.img_height, self.img_width), interpolation=cv2.INTER_AREA)
        if self.crop: # random crop
            img = img[self.crop_position[0]:self.crop_position[0] + self.crop_height, self.crop_position[1]:self.crop_position[1] + self.crop_width]
        if self.flip and self.useflip == 1:
            img = cv2.flip(img, 1) # horizontal flip

        img = np.array(img).astype('float32')

        return img

    def make_segment(self, frame_list):

        frame_length = len(frame_list)
        segment_group = np.empty([int(frame_length//self.frame_len), self.frame_len, self.img_height, self.img_width,3], np.dtype('float32'))
        
        if self.crop:
            segment_group = np.empty([int(frame_length//self.frame_len), self.frame_len, self.crop_height, self.crop_width,3], np.dtype('float32'))
            self.crop_position = [random.randint(0, self.img_height-self.crop_height), random.randint(0, self.img_width-self.crop_width)]
        
        if self.flip:
            self.useflip = random.choices(range(0,2), weights=[1-self.flip_prob, self.flip_prob])

        for i in range(int(frame_length//self.frame_len)):
            segment = np.empty([self.frame_len, self.img_height, self.img_width,3], np.dtype('float32'))
            if self.crop:
                segment = np.empty([self.frame_len, self.crop_height, self.crop_width,3], np.dtype('float32'))
            start_frame_num = i * self.frame_len
            end_frame_num = (i+1) * self.frame_len 

            for j, k in enumerate(range(start_frame_num, end_frame_num)):
                segment[j] = self.load_image(frame_list[k])
            self.to_tensor(segment)
            segment_group[i] = segment

        return torch.from_numpy(segment_group)

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):

        frame_list = glob.glob(self.video_list[index] + "/*.jpg") # sort 진행해야됨
        label = frame_list[0].split('/')[-3]
        label = self.label_list.index(label)
        segment = self.make_segment(frame_list)

        return segment, label

if __name__ == '__main__':
    dataset = action_dataset(root_dir ='/workspace/action_recognition/Data')

    for i, (segment,label) in enumerate(dataset):
        print(segment.size(), label)
