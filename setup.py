# setup.py

import setuptools

# Read the contents of your README file for the long description
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.0"

REPO_NAME = "End-to-End-Chest-Cancer-Classification-using-MLflow-and-DVC"
AUTHOR_USER_NAME = "AlyyanAhmed21" # Change to your GitHub username
SRC_REPO = "cnnClassifier" # This is the name of your main source folder under src/
AUTHOR_EMAIL = "alyyanawan19@gmail.com" # Change to your email

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for CNN app",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    # This is the crucial part!
    # It tells setuptools to look for packages in the 'src' directory.
    package_dir={"": "src"},
    # This finds all packages automatically within the directory specified above.
    packages=setuptools.find_packages(where="src")
)