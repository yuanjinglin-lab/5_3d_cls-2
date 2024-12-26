from moviepy.editor import VideoFileClip


video_path = "data/normal第二批/40782614/Image85.avi"
left_path = "A1.avi"
right_path = "B1.avi"
 
# 读取视频
clip = VideoFileClip(video_path)
print(clip.size)

# 裁剪左边和右边的视频
clip_left = clip.crop(x1=0, y1=0, x2=508, y2=708)
clip_right = clip.crop(x1=508, y1=0, x2=1016, y2=708)
print(clip_left.size, clip_right.size)
 
# 输出裁剪后的视频
clip_left.write_videofile(left_path, codec='png')
clip_right.write_videofile(right_path, codec='png')