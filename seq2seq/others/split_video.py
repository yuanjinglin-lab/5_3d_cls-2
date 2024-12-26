import os

from moviepy.editor import VideoFileClip, clips_array
from tqdm import tqdm


def fun(input_video_path):
    # 输入视频文件路径
    # input_video_path = 'video(14).mp4'
    name = input_video_path.split('.')[0]
    # 加载视频剪辑
    video_clip = VideoFileClip(input_video_path)

    # 获取视频的一半宽度
    half_width = video_clip.size[0] // 2

    # 裁剪左右两边
    left_clip = video_clip.crop(x1=0, y1=0, x2=half_width, y2=video_clip.size[1])
    right_clip = video_clip.crop(x1=half_width, y1=0, x2=video_clip.size[0], y2=video_clip.size[1])

    # 输出左右两个视频文件
    output_left_video_path = f'{name}_color.mp4'
    output_right_video_path = f'{name}_black.mp4'
    left_clip.write_videofile(output_left_video_path, codec="libx264", audio_codec="aac")
    right_clip.write_videofile(output_right_video_path, codec="libx264", audio_codec="aac")

if __name__ == '__main__':
    root = 'huanzhe'

    dirs = os.listdir(root)
    for dir in tqdm(dirs):
        vides = os.listdir(root + "/" + dir)
        for vide in vides:
            if vide.endswith('MP4') or vide.endswith('AVI') or vide.endswith('WMV'):
                fun(root + "/" + dir + "/" + vide)