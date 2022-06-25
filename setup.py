from codecs import open

from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()

setup(
    name='lbbutils',

    version='0.0.17',

    description='lbbutils - Some tools for neural network and image fusion.',
    long_description=long_description,

    url='https://github.com/littlebaba/lbbutils',

    author='Heng',
    author_email='littlebaba0304@163.com',

    license='MIT',

    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],

    keywords='deep-learning multi-focus',

    packages=['lbbutils','lbbutils/metrics'],
    install_requires=[
        'numpy',
    ]
)
