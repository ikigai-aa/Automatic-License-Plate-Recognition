import argparse
import os
import subprocess
import time
import sys
import ipdb
import re


def main(args):
    # Parameters from the args
    dir, h, w, fps, extension = args.dir, args.height, args.width, args.fps, args.extension

    # avi dir
    dir_split = dir.split('/')
    avi_dir = dir_split[-1]
    root_dir = '/'.join(dir_split[:-1])
    new_avi_dir = "{}_{}x{}_{}".format(avi_dir, w, h, fps)
    new_dir = os.path.join(root_dir, new_avi_dir)
    os.makedirs(new_dir, exist_ok=True)

    # Get the video filenames
    list_video_fn = get_all_videos(dir, extension)

    print("{} videos to uncompressed in total".format(len(list_video_fn)))

    # Loop over the video and extract
    op_time = AverageMeter()
    start = time.time()
    list_error_fn = []
    for i, video_fn in enumerate(list_video_fn):
        try:
            # Rescale
            rescale_video(video_fn, w, h, fps, dir, new_dir, extension)

            # Log
            duration = time.time() - start
            op_time.update(duration, 1)
            print("{}/{} : {time.val:.3f} ({time.avg:.3f}) sec/video".format(i + 1, len(list_video_fn), time=op_time))
            sys.stdout.flush()
            start = time.time()
        except:
            print("Impossible to rescale video for {}".format(video_fn))
            list_error_fn.append(video_fn)

    print("\nDone")
    print("\nImpossible to extract frames for {} videos: \n {}".format(len(list_error_fn), list_error_fn))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_duration(file):
    """Get the duration of a video using ffprobe. -> https://stackoverflow.com/questions/31024968/using-ffmpeg-to-obtain-video-durations-in-python"""
    cmd = 'ffprobe -i {} -show_entries format=duration -v quiet -of csv="p=0"'.format(file)
    output = subprocess.check_output(
        cmd,
        shell=True,  # Let this run in the shell
        stderr=subprocess.STDOUT
    )
    # return round(float(output))  # ugly, but rounds your seconds up or down
    return float(output)


def rescale_video(video_fn, w, h, fps, dir, new_dir, extension):
    """ Rescale a video according to its new width, height an fps """

    # Output video_name
    video_dir = video_fn.split(dir)[1]
    video_dir = video_dir.split('/')
    video_dir, video_name_rescaled = '/'.join(video_dir[:-1])[1:], video_dir[-1]

    # Create the dir
    video_dir_to_create = os.path.join(new_dir, video_dir)
    os.makedirs(video_dir_to_create, exist_ok=True)
    video_fn_rescaled = os.path.join(new_dir, video_dir, video_name_rescaled)

    # Run a subprocess using ffmepg
    subprocess.call('ffmpeg -i {video_input} -vf scale={w}:{h} -r {fps} -y {video_output} -loglevel panic'.format(
        video_input=video_fn,
        h=h,
        w=w,
        fps=fps,
        video_output=video_fn_rescaled
    ), shell=True)

    # Get the duration of the new video (in sec)
    duration_sec = get_duration(video_fn_rescaled)
    duration_frames = int(duration_sec * fps)

    # Update the name of the file
    video_name_rescaled_dur = video_name_rescaled.split('.')[0] + '_{}.{}'.format(duration_frames, extension)
    video_fn_rescaled_dur = os.path.join(new_dir, video_dir, video_name_rescaled_dur)
    os.rename(video_fn_rescaled, video_fn_rescaled_dur)

    return video_fn_rescaled


def get_all_videos(dir, extension='mp4'):
    """ Return a list of the video filename from a directory and its subdirectories """

    list_video_fn = []
    for dirpath, dirnames, filenames in os.walk(dir):
        for filename in [f for f in filenames if f.endswith(extension)]:
            fn = os.path.join(dirpath, filename)
            list_video_fn.append(fn)

    return list_video_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract frames')
    parser.add_argument('--dir', metavar='DIR',
                        default='/Users/fabien/Downloads/avi',
                        help='path to avi dir')
    parser.add_argument('--width', default=256, type=int,
                        metavar='W', help='Width')
    parser.add_argument('--height', default=256, type=int,
                        metavar='H', help='Height')
    parser.add_argument('--fps', default=30, type=int,
                        metavar='FPS',
                        help='Frames per second for the extraction, -1 means that we take the fps from the video')
    parser.add_argument('--extension', metavar='E',
                        default='mp4',
                        help='Extension of the video files')

    args = parser.parse_args()

    main(args)
