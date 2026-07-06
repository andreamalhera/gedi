from setuptools import setup, find_packages
import os

with open("README.md", "r") as fh:
    long_description = fh.read()

version_string = os.environ.get("VERSION_PLACEHOLDER", "1.9.0")
print(version_string)
version = version_string

setup(
        name = 'gedi',
        version = str(version),
        packages = find_packages(),
        description = 'Generating Event Data with Intentional Features for Benchmarking Process Mining',
        author = 'Andrea Maldonado',
        author_email = 'andreamalher.works@gmail.com',
        license = 'MIT',
        url='https://github.com/andreamalhera/gedi.git',
        long_description=long_description,
        long_description_content_type="text/markdown",

        python_requires='>=3.9',

        include_package_data=True,
        install_requires=[
            'pandas==2.2.3',
            'numpy>=1.26.4',
            'ConfigSpace==1.2.0',
            'feeed>=2.0.0',
            'smac==2.4.0',
            'seaborn==0.13.2',
            'scipy~=1.14.1',
            'scikit-learn>=1.6.1,<1.9.0',
            'tqdm~=4.65.0',
            'matplotlib>=3.10.9',
            'pm4py>=2.7.2',
            'imbalanced-learn~=0.14.2',
            'pytest~=8.3.4',
            ],
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Science/Research',
            'Topic :: Software Development',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3.12',
            "Programming Language :: Python :: 3.14",
    ],
)
