#!/usr/bin/env python3
"""Publish recorded .npy frames as a PointCloud2 topic, to test the node without sensors.

    python3 realtime/replay_frames.py data/real/labE_2026-08-27/*.npy --rate 5
"""

import argparse
import os
import sys

import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from realtime.segmentation_node import xyz_label_to_cloud  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--topic", default="/merged/cloud")
    ap.add_argument("--rate", type=float, default=5.0, help="frames per second")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--frame-id", default="os_sensor")
    a = ap.parse_args(rospy.myargv()[1:])

    rospy.init_node("replay_frames")
    pub = rospy.Publisher(a.topic, PointCloud2, queue_size=1)
    rate = rospy.Rate(a.rate)
    rospy.sleep(1.0)                      # let subscribers connect
    while not rospy.is_shutdown():
        for f in a.files:
            pts = np.load(f)[:, :3].astype(np.float32)
            pub.publish(xyz_label_to_cloud(pts, np.zeros(len(pts), dtype=np.int64),
                                           a.frame_id, rospy.Time.now()))
            rate.sleep()
            if rospy.is_shutdown():
                return
        if not a.loop:
            return


if __name__ == "__main__":
    main()
