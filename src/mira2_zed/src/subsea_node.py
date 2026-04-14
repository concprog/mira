import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
import struct
import ctypes

from point_cloud_util import (
    is_yellow_lime_green,
    filter_points_by_color,
    cluster_with_hdbscan,
    segment_plane_ransac,
    segment_multiple_planes,
)


_DATATYPES = {}
_DATATYPES[PointField.INT8] = ("b", 1)
_DATATYPES[PointField.UINT8] = ("B", 1)
_DATATYPES[PointField.INT16] = ("h", 2)
_DATATYPES[PointField.UINT16] = ("H", 2)
_DATATYPES[PointField.INT32] = ("i", 4)
_DATATYPES[PointField.UINT32] = ("I", 4)
_DATATYPES[PointField.FLOAT32] = ("f", 4)
_DATATYPES[PointField.FLOAT64] = ("d", 8)


def read_points(cloud, field_names=None, skip_nans=False, uvs=[]):
    import math

    assert isinstance(cloud, PointCloud2), "cloud is not a sensor_msgs.msg.PointCloud2"
    fmt = _get_struct_fmt(cloud.is_bigendian, cloud.fields, field_names)
    width, height, point_step, row_step, data, isnan = (
        cloud.width,
        cloud.height,
        cloud.point_step,
        cloud.row_step,
        cloud.data,
        math.isnan,
    )
    unpack_from = struct.Struct(fmt).unpack_from

    if skip_nans:
        if uvs:
            for u, v in uvs:
                p = unpack_from(data, (row_step * v) + (point_step * u))
                has_nan = False
                for pv in p:
                    if isnan(pv):
                        has_nan = True
                        break
                if not has_nan:
                    yield p
        else:
            for v in range(height):
                offset = row_step * v
                for u in range(width):
                    p = unpack_from(data, offset)
                    has_nan = False
                    for pv in p:
                        if isnan(pv):
                            has_nan = True
                            break
                    if not has_nan:
                        yield p
                    offset += point_step
    else:
        if uvs:
            for u, v in uvs:
                yield unpack_from(data, (row_step * v) + (point_step * u))
        else:
            for v in range(height):
                offset = row_step * v
                for u in range(width):
                    yield unpack_from(data, offset)
                    offset += point_step


def _get_struct_fmt(is_bigendian, fields, field_names=None):
    import sys

    fmt = ">" if is_bigendian else "<"
    offset = 0
    for field in (
        f
        for f in sorted(fields, key=lambda f: f.offset)
        if field_names is None or f.name in field_names
    ):
        if offset < field.offset:
            fmt += "x" * (field.offset - offset)
            offset = field.offset
        if field.datatype not in _DATATYPES:
            print(
                "Skipping unknown PointField datatype [%d]" % field.datatype,
                file=sys.stderr,
            )
        else:
            datatype_fmt, datatype_length = _DATATYPES[field.datatype]
            fmt += field.count * datatype_fmt
            offset += field.count * datatype_length
    return fmt


def extract_rgb_bitwise(rgb_float):
    s = struct.pack(">f", rgb_float)
    i = struct.unpack(">l", s)[0]
    pack = ctypes.c_uint32(i).value
    r = (pack & 0x00FF0000) >> 16
    g = (pack & 0x0000FF00) >> 8
    b = pack & 0x000000FF
    return r, g, b


class SubseaNode(Node):
    def __init__(self):
        super().__init__("subsea_node")

        self.get_logger().info("Subsea node initialized")

        self.pcd_subscriber = self.create_subscription(
            PointCloud2,
            "/zed/zed_node/point_cloud/fused_cloud_registered",
            self.listener_callback,
            10,
        )

        self.fused_publisher = self.create_publisher(
            PointCloud2, "/subsea/fused_cloud_result", 10
        )

        self.points = None
        self.colors = None

    def listener_callback(self, msg):
        self.get_logger().info(f"Received PointCloud2: {msg.width}x{msg.height} points")

        gen = read_points(msg, skip_nans=True, field_names=["x", "y", "z", "rgb"])
        data = list(gen)

        if not data:
            self.get_logger().warn("No valid points received")
            return

        points_array = np.array(data, dtype=np.float32)
        self.points = points_array[:, :3]
        self.colors = np.zeros((len(self.points), 3), dtype=np.uint8)

        for idx, point in enumerate(data):
            r, g, b = extract_rgb_bitwise(point[3])
            self.colors[idx] = [r, g, b]

        self.get_logger().info(f"Extracted {len(self.points)} points with colors")

        self.process_pointcloud()

    def process_pointcloud(self):
        if self.points is None or len(self.points) == 0:
            return

        r, g, b = self.colors[:, 0], self.colors[:, 1], self.colors[:, 2]
        yellow_lime_mask = is_yellow_lime_green(r, g, b)
        yellow_lime_count = np.sum(yellow_lime_mask)
        self.get_logger().info(
            f"Yellow/lime green points: {yellow_lime_count}/{len(self.points)}"
        )

        if yellow_lime_count > 0:
            yellow_lime_points = self.points[yellow_lime_mask]
            yellow_lime_colors = self.colors[yellow_lime_mask]

            try:
                planes = segment_multiple_planes(
                    yellow_lime_points,
                    distance_threshold=0.02,
                    ransac_n=3,
                    num_iterations=1000,
                    min_inliers=50,
                )
                self.get_logger().info(
                    f"Detected {len(planes)} planes in yellow/lime green points"
                )
                for i, (plane_eq, inliers) in enumerate(planes):
                    self.get_logger().info(
                        f"  Plane {i}: {plane_eq}, inliers: {len(inliers)}"
                    )
            except Exception as e:
                self.get_logger().warn(f"RANSAC plane segmentation failed: {e}")

        try:
            labels, features = cluster_with_hdbscan(
                self.points,
                self.colors,
                min_cluster_size=10,
                min_samples=5,
                color_weight=0.3,
            )
            unique_labels = np.unique(labels)
            self.get_logger().info(
                f"HDBSCAN clusters: {len(unique_labels[unique_labels >= 0])}"
            )
        except Exception as e:
            self.get_logger().warn(f"HDBSCAN clustering failed: {e}")

        self.publish_result()

    def publish_result(self):
        if self.points is None or len(self.points) == 0:
            return

        result_msg = PointCloud2()
        result_msg.header.stamp = self.get_clock().now().to_msg()
        result_msg.header.frame_id = "map"
        result_msg.width = len(self.points)
        result_msg.height = 1
        result_msg.is_bigendian = False
        result_msg.is_dense = False

        result_msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        result_msg.point_step = 16
        result_msg.row_step = 16 * len(self.points)

        data = np.zeros(len(self.points) * 4, dtype=np.float32)
        data[0::4] = self.points[:, 0]
        data[1::4] = self.points[:, 1]
        data[2::4] = self.points[:, 2]

        rgb_uint32 = (
            (self.colors[:, 0].astype(np.uint32) << 16)
            | (self.colors[:, 1].astype(np.uint32) << 8)
            | self.colors[:, 2].astype(np.uint32)
        )
        data[3::4] = rgb_uint32.view(np.float32)

        result_msg.data = data.tobytes()
        self.fused_publisher.publish(result_msg)
        self.get_logger().info("Published fused cloud result")


def main(args=None):
    rclpy.init(args=args)
    node = SubseaNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
