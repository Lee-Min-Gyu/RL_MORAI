from glob import glob
from setuptools import find_packages, setup


package_name = "morai_rl"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", glob("config/*.toml")),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="mglee",
    maintainer_email="mglee@example.com",
    description="ROS2 MORAI reinforcement learning environment.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "train_ppo = morai_rl.scripts.train_ppo:main",
            "check_reset = morai_rl.scripts.check_reset:main",
            "check_reset_drive = morai_rl.scripts.check_reset_drive:main",
            "check_step_loop = morai_rl.scripts.check_step_loop:main",
            "run_simple_driver = morai_rl.scripts.run_simple_driver:main",
        ],
    },
)
