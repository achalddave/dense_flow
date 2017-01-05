__author__ = 'yjxiong'

import glob
import os
import random
from multiprocessing import Pool, current_process
from functools import partial

import argparse
out_path = ''


def run_optical_flow(vid_item, step=1, num_gpus=4, warp=False):
    vid_path = vid_item[0]
    vid_id = vid_item[1]
    vid_name = vid_path.split('/')[-1].split('.')[0]
    out_full_path = os.path.join(out_path, vid_name)
    try:
        os.mkdir(out_full_path)
    except OSError:
        pass

    current = current_process()
    dev_id = random.choice(range(num_gpus))
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
    parser.add_argument("--num_gpus", type=int, default=4)

    args = parser.parse_args()

    out_path = args.out_dir
    src_path = args.src_dir
    num_worker = args.num_worker
    flow_type = args.flow_type


    vid_list = glob.glob(src_path + '/*.mp4')
    pool = Pool(num_worker)

    warp_mode = flow_type == 'warp_tvl1'
    pool.map(
        partial(run_optical_flow,
                step=args.step,
                warp=warp_mode,
                num_gpus=args.num_gpus),
        zip(vid_list, xrange(len(vid_list))))
