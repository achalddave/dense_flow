__author__ = 'yjxiong'

import cv2
import glob
import os
from multiprocessing import Pool, current_process
from functools import partial

import argparse
out_path = ''


def dump_frames(vid_path):
    video = cv2.VideoCapture(vid_path)
    vid_name = vid_path.split('/')[-1].split('.')[0]
    out_full_path = os.path.join(out_path, vid_name)

    fcount = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    try:
        os.mkdir(out_full_path)
    except OSError:
        pass
    file_list = []
    for i in xrange(fcount):
        ret, frame = video.read()
        assert ret
        cv2.imwrite('{}/{:06d}.jpg'.format(out_full_path, i), frame)
        access_path = '{}/{:06d}.jpg'.format(vid_name, i)
        file_list.append(access_path)
    print '{} done'.format(vid_name)
    return file_list


def run_optical_flow(vid_item, step=1, dev_id=0, warp=False):
    vid_path = vid_item[0]
    vid_id = vid_item[1]
    vid_name = vid_path.split('/')[-1].split('.')[0]
    out_full_path = os.path.join(out_path, vid_name)
    try:
        os.mkdir(out_full_path)
    except OSError:
        pass

    current = current_process()
    dev_id = int(current._identity[0]) - 1
    image_path = '{}/img'.format(out_full_path)
    flow_x_path = '{}/flow_x'.format(out_full_path)
    flow_y_path = '{}/flow_y'.format(out_full_path)

    if warp:
        cmd = './build/extract_warp_gpu -f={} -x={} -y={} -b=20 -t=1 -d={} -s={} -o=zip'.format(vid_path, flow_x_path, flow_y_path, dev_id, step)
    else:
        cmd = './build/extract_gpu -f={} -x={} -y={} -i={} -b=20 -t=1 -d={} -s={} -o=zip'.format(vid_path, flow_x_path, flow_y_path, image_path, dev_id, step)

    os.system(cmd)
    if warp:
        print 'warp on {} {} done'.format(vid_id, vid_name)
    else:
        print '{} {} done'.format(vid_id, vid_name)
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="extract optical flows")
    parser.add_argument("src_dir")
    parser.add_argument("out_dir")
    parser.add_argument("--num_worker", type=int, default=8)
    parser.add_argument("--flow_type", type=str, default='tvl1', choices=['tvl1', 'warp_tvl1'])
    parser.add_argument("--step", type=int, default=1)

    args = parser.parse_args()

    out_path = args.out_dir
    src_path = args.src_dir
    num_worker = args.num_worker
    flow_type = args.flow_type


    vid_list = glob.glob(src_path+'/*.mp4')
    print len(vid_list)
    pool = Pool(num_worker)

    warp_mode = flow_type == 'warp_tvl1'
    pool.map(
        partial(run_optical_flow,
                step=args.step,
                warp=warp_mode),
        zip(vid_list, xrange(len(vid_list))))
