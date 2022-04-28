# Video-Action-recognition-dataloader
Dataloader for video/action-aware machine learning. 
Supports RGB Images only.


### For frame extraction

```bash
python extract_frame.py --data_path /path2video/video_group1 --save_path /path/to/path2extracted_frame/
```



Input
```Shell
/path2video/
   video_group1/
    video1.mp4 
    video2.mp4 
``` 

Result 
```Shell
/path2extracted_frame/
   video_group1/
    video1/
      frame0.jpg
      frame1.jpg
      frame2.jpg
          ...
      frameN.jpg
    video2/
      frame0.jpg
      frame1.jpg
          ...
      frameN.jpg
```   

### For dataloader

Input : Whole frames in the videos

Ouptut : [ number of segment, number of frame in the segment, image height, image width, channel ]

Data Argumentation : Random crop, horizontal flip

```bash
 python action_dataset.py
```
