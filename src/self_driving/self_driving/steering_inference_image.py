#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import torch
import torchvision.transforms as T
import numpy as np
import threading, queue
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from PIL import Image
from self_driving.model import ImageOnlySteerNet   # use your new model
import time
"""
run:
  ros2 run steering_inference steering_image --ros-args -p model_path:=/home/abhi/turtle_pure_regression/ckpt_best.pt -p device:=cuda -p linear_vel:=0.01 -p flip_sign:=True
"""

class SteeringNode(Node):
    def __init__(self):
        super().__init__('steering_inference_node')

        # ---- Parameters ----
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('model_path', 'model.pt')
        self.declare_parameter('linear_vel', 0.01)
        self.declare_parameter('rate_hz', 5.0)
        self.declare_parameter('flip_sign', False)

        self.device = torch.device(
            self.get_parameter('device').get_parameter_value().string_value
        )
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.linear_vel = float(self.get_parameter('linear_vel').get_parameter_value().double_value)
        self.rate_hz = float(self.get_parameter('rate_hz').get_parameter_value().double_value)
        self.flip_sign = bool(self.get_parameter('flip_sign').get_parameter_value().bool_value)

        # ---- Load model ----
        self.get_logger().info(f"Loading ImageOnlySteerNet from {model_path} on {self.device}")
        ckpt = torch.load(model_path, map_location=self.device)
        self.mu = float(ckpt.get("mu", 0.0))
        self.sigma = float(ckpt.get("sigma", 1.0))
        self.omega_sign = ckpt.get("omega_sign", 1.0)
        if self.flip_sign:
            self.omega_sign *= -1.0

        self.model = ImageOnlySteerNet(out_len=1, pretrained=False).to(self.device)
        state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        self.get_logger().info("Model loaded and ready for inference.")

        # ---- ROS Interfaces ----
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.bridge = CvBridge()
        self.sub = self.create_subscription(
            CompressedImage, '/image_raw/compressed', self.image_callback, qos_profile=qos)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 1)

        # ---- Timer-driven publisher ----
        self.latest_omega = 0.0
        self.timer = self.create_timer(1.0 / self.rate_hz, self.publish_cmd)

        # ---- Preprocessing (matches training) ----
        self.tf = T.Compose([
            T.ToPILImage(),
            T.Lambda(lambda im: im.crop((0, int(im.height * 0.20), im.width, im.height))),
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        # ---- Runtime smoothing ----
        self.omega_f = 0.0
        self.omega_prev = 0.0

        # ---- Frame queue + background thread ----
        self.frame_q = queue.Queue(maxsize=1)
        self.worker = threading.Thread(target=self.worker_loop, daemon=True)
        self.worker.start()

        self.get_logger().info(f"Ready — publishing /cmd_vel at {self.rate_hz:.1f} Hz")

    # ---------------------------------------------------
    def image_callback(self, msg: CompressedImage):
        """Store latest image without blocking."""
        if self.frame_q.full():
            try:
                self.frame_q.get_nowait()
            except queue.Empty:
                pass
        try:
            self.frame_q.put_nowait(msg)
        except queue.Full:
            pass

    # ---------------------------------------------------
    def worker_loop(self):
        """Background thread: runs inference on latest frame."""
        while rclpy.ok():
            try:
                msg = self.frame_q.get(timeout=1.0)
            except queue.Empty:
                continue

            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_image is None:
                continue

            # Preprocess
            img_tensor = self.tf(cv_image).unsqueeze(0).to(self.device, non_blocking=True)

            # Inference
            with torch.inference_mode():
                y_std = self.model(img_tensor)
                y_raw = y_std * self.sigma + self.mu
                omega = float(self.omega_sign * y_raw.item())

            # Optional smoothing & clamp
            # alpha = 0.3
            # self.omega_f = alpha * omega + (1 - alpha) * self.omega_f
            # max_slew = 0.4
            # delta = self.omega_f - self.omega_prev
            # delta = np.clip(delta, -max_slew, max_slew)
            # self.omega_prev += delta
            # omega_cmd = np.clip(self.omega_prev, -0.6, 0.6)
            #
            omega_cmd = omega
            # Optional gain scaling (tunable)
            self.latest_omega = float(1.2* omega_cmd)

            #self.get_logger().info_throttle(
            #    1.0,
            #    f"ω_cmd={omega_cmd:+.3f} rad/s  (raw={omega:+.3f})",
            #)
            # Simple periodic print every second
            now = time.time()
            if not hasattr(self, "_last_log_t") or now - self._last_log_t > 1.0:
                self._last_log_t = now
                self.get_logger().info(f"ω_cmd={omega_cmd:+.3f} rad/s  (raw={omega:+.3f})")


    # ---------------------------------------------------
    def publish_cmd(self):
        """Timer callback: publish latest steering command at fixed rate."""
        twist = Twist()
        twist.linear.x = self.linear_vel
        twist.angular.z = self.latest_omega
        self.pub.publish(twist)

# -------------------------------------------------------
from rclpy.executors import MultiThreadedExecutor

def main():
    rclpy.init()
    node = SteeringNode()
    executor = MultiThreadedExecutor(num_threads=2)
    try:
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

