from setuptools import setup

setup(
   name='team-crux-p1',
   version='1.0.0-dev',
   description='Scalable document classification with PySpark - CSCI 8360 @ UGA',
   url='https://github.com/dsp-uga/team-crux-p1',
   maintainer='Zach Jones',
   maintainer_email='zach.dean.jones@gmail.com',
   license='MIT',
   packages=['src', 'src.classifiers', 'src.utilities']
)