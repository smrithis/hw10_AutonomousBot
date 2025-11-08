import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import threading
import time
import cv2

class NonBlockingImageSubscriber(Node):
    def __init__(self):
        super().__init__('non_blocking_image_subscriber')

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            qos_profile
        )

        self.bridge = CvBridge()
        self.latest_image_lock = threading.Lock()
        self.latest_image = None

        # Background processing thread
        self.processing_thread = threading.Thread(target=self.process_images, daemon=True)
        self.processing_thread.start()

        self.get_logger().info('Non-blocking image subscriber started.')

    def image_callback(self, msg):
        # Store only the latest image
        with self.latest_image_lock:
            self.latest_image = msg

    def process_images(self):
        while rclpy.ok():
            local_image = None
            with self.latest_image_lock:
                if self.latest_image is not None:
                    local_image = self.latest_image
                    self.latest_image = None

            if local_image is not None:
                try:
                    cv_image = self.bridge.imgmsg_to_cv2(local_image, 'bgr8')
                    # Example: process or display
                    # cv2.imshow("Camera Feed", cv_image)
                    # cv2.waitKey(1)
                    self.get_logger().info('Processed one image frame.')
                except Exception as e:
                    self.get_logger().error(f"Error processing image: {e}")

            # Sleep to avoid busy-waiting
            time.sleep(1.0 / 30.0)  # 30 Hz loop

def main(args=None):
    rclpy.init(args=args)
    node = NonBlockingImageSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

