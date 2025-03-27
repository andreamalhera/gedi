from setuptools import setup, find_packages
import os

with open("README.md", "r") as fh:
    long_description = fh.read()

version_string = os.environ.get("VERSION_PLACEHOLDER", "1.0.8")
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
        url='https://github.com/lmu-dbs/gedi.git',
        long_description=long_description,
        long_description_content_type="text/markdown",
        include_package_data=True,
        install_requires=[
            'pandas==2.2.3',
            'numpy==1.26.4',
            'ConfigSpace==1.2.0',
            'feeed==1.3.2',
            'smac==2.2.0',
            'imblearn~=0.0',
            'seaborn==0.13.2',
            'scipy~=1.14.1',
            'scikit-learn~=1.5.2',
            'tqdm~=4.65.0',
            'matplotlib~=3.9.2',
            'pm4py~=2.7.2',
            'imbalanced-learn~=0.12.4',
            'pytest~=8.3.4',
            ],
        classifiers=[
            'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
            'Intended Audience :: Science/Research',      # Define that your audience are developers
            'Topic :: Software Development',
            'License :: OSI Approved :: MIT License',   # Again, pick a license
            'Programming Language :: Python :: 3.12',
    ],
)
