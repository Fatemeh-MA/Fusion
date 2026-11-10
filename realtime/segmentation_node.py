#!/usr/bin/env python3
"""ROS 1 node: segment the merged LiDAR cloud in real time.

    rosrun / python3 realtime/segmentation_node.py --config configs/realtime.yaml

Subscribes to the merged cloud of both LiDARs and publishes:
    <ns>/segmented_point_cloud        PointCloud2 (x, y, z, rgba) of the dynamic classes
    <ns>/segmented_point_cloud_label  Float64MultiArray, flattened [x, y, z, label] rows
    <ns>/Wall, <ns>/floor, <ns>/Table PointCloud2, published once (static scene)

Topic names are those of the original node. The checkpoint carries its own
normalization; only the crop box and voxel size come from the config.
"""

import argparse
import os
import sys

import numpy as np
import rospy
import yaml
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Float64MultiArray, Header
from std_srvs.srv import Empty, EmptyResponse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sim2real.labels import CLASS_COLORS, CLASS_NAMES  # noqa: E402
from sim2real.realtime import Segmenter  # noqa: E402

STATIC = {"Wall": 2, "floor": 3, "Table": 5}
_DTYPES = {PointField.FLOAT32: "<f4", PointField.FLOAT64: "<f8"}


def cloud_to_xyz(msg):
    """PointCloud2 -> (N, 3) float32, without ros_numpy."""
    fields = {f.name: f for f in msg.fields}
    dtype = np.dtype({"names": ["x", "y", "z"],
                      "formats": [_DTYPES[fields[c].datatype] for c in "xyz"],
                      "offsets": [fields[c].offset for c in "xyz"],
                      "itemsize": msg.point_step})
    arr = np.frombuffer(msg.data, dtype=dtype, count=msg.width * msg.height)
    return np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(np.float32)


def xyz_label_to_cloud(points, labels, frame_id, stamp):
    """(N, 3) points + labels -> PointCloud2 with a packed rgba field."""
    rgb = (np.asarray(CLASS_COLORS) * 255).astype(np.uint32)
    c = rgb[labels]
    rgba = (255 << 24) | (c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]
    data = np.empty(len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgba", "<u4")])
    data["x"], data["y"], data["z"], data["rgba"] = points[:, 0], points[:, 1], points[:, 2], rgba
    fields = [PointField("x", 0, PointField.FLOAT32, 1), PointField("y", 4, PointField.FLOAT32, 1),
              PointField("z", 8, PointField.FLOAT32, 1), PointField("rgba", 12, PointField.UINT32, 1)]
    return PointCloud2(header=Header(frame_id=frame_id, stamp=stamp), height=1, width=len(points),
                       fields=fields, is_bigendian=False, point_step=16, row_step=16 * len(points),
                       data=data.tobytes(), is_dense=True)


class SegmentationNode:
    def __init__(self, cfg):
        self.cfg = cfg
        self.seg = Segmenter(cfg["checkpoint"], crop_box=cfg.get("crop_box"),
                             voxel_size=cfg.get("voxel_size", 0.048),
                             max_points=cfg.get("max_points", 20000))
        rospy.loginfo("loaded %s (%r)", cfg["checkpoint"], self.seg.normalizer)
        self.frame_id = cfg.get("frame_id", "os_sensor")
        self.static_sent = set()
        self.pub_cloud = rospy.Publisher("segmented_point_cloud", PointCloud2, queue_size=1)
        self.pub_label = rospy.Publisher("segmented_point_cloud_label", Float64MultiArray, queue_size=1)
        self.pub_static = {name: rospy.Publisher(name, PointCloud2, queue_size=1, latch=True)
                           for name in STATIC}
        rospy.Service("reset_floor", Empty, self.reset_floor)
        rospy.Subscriber(cfg.get("input_topic", "/merged/cloud"), PointCloud2, self.callback,
                         queue_size=1, buff_size=2 ** 24)

    def reset_floor(self, _):
        """Call after moving the sensors: the floor is fitted again on the next frame."""
        self.seg.reset_floor()
        self.static_sent.clear()
        return EmptyResponse()

    def callback(self, msg):
        points, labels = self.seg(cloud_to_xyz(msg))
        if len(points) == 0:
            return
        stamp = msg.header.stamp
        for name, cls in STATIC.items():
            if name not in self.static_sent:
                m = labels == cls
                self.pub_static[name].publish(xyz_label_to_cloud(points[m], labels[m], self.frame_id, stamp))
                self.static_sent.add(name)
        dynamic = ~np.isin(labels, list(STATIC.values()))
        self.pub_cloud.publish(xyz_label_to_cloud(points[dynamic], labels[dynamic], self.frame_id, stamp))
        self.pub_label.publish(Float64MultiArray(
            data=np.column_stack([points, labels]).astype(np.float64).ravel().tolist()))
        t = self.seg.timing
        counts = np.bincount(labels, minlength=len(CLASS_NAMES))
        rospy.loginfo_throttle(2.0, "%d pts  pre %.1f ms  net %.1f ms  | %s" % (
            t["points"], 1e3 * t["preprocess"], 1e3 * t["inference"],
            " ".join("%s:%d" % (n, c) for n, c in zip(CLASS_NAMES, counts) if c)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs",
                                                     "realtime.yaml"))
    args, _ = ap.parse_known_args(rospy.myargv()[1:])
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    rospy.init_node("semantic_segmentation_node")
    SegmentationNode(cfg)
    rospy.spin()


if __name__ == "__main__":
    main()
