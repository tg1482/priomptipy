from setuptools import setup, find_packages

# Read the contents of your requirements file
with open("requirements.txt") as f:
    requirements = f.read().splitlines()


setup(
    name="priomptipy",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,  # Include the requirements
    author="The Quarkle Dev Team",
    author_email="samarth@quarkle.ai, tanmay@quarkle.ai",
    description="A library for creating smarter prompts for LLMs by adding priority to components.",
)
