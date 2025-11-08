from setuptools import setup

package_name = 'self_driving'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'torch', 'torchvision', 'opencv-python', 'numpy'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@cmu.edu',
    description='ROS2 node for steering angle regression using camera input.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'steering_image = self_driving.steering_inference_image:main',
        ],
    },
)
