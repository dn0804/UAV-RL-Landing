from setuptools import setup, find_packages

setup(
    name="rl_uav_package",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "train_ppo=rl_uav_package.train_ppo:main",
        ],
    },
)
