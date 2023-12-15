from setuptools import setup, find_packages

setup(
    name = 'stKeep',
    version = '0.0.1',
    keywords='TME',
    description = 'stKeep for processing spatially transcriptomics data',
    license = 'MIT License',
    url = 'https://github.com/cmzuo11/stKeep',
    author = 'cmzuo',
    author_email = 'cmzuo@dhu.edu.cn',
    packages = find_packages(),
    include_package_data = True,
    platforms = 'any',
    install_requires = [
        'numpy==1.22.4',
        'pandas==2.0.3',
        'scipy==1.8.1',
        'scikit-learn',
        'torch==1.13.0',
        'tqdm==4.51.0',
        'scanpy==1.9.6',
        'seaborn',
        'matplotlib==3.7.3',
        'glob2',
        'Pillow==9.5.0',
        'anndata==0.9.2',
        'argparse==1.1',
        'pathlib',
        'opencv-python==4.8.1.78',
        'torchvision==0.14.0'
        ],
        )
