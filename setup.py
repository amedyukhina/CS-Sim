from setuptools import setup

import cs_sim

setup(
    name='cs-sim',
    version=cs_sim.__version__,
    url='https://github.com/amedyukhina/CS-Sim',
    author="Anna Medyukhina",
    author_email='anna.medyukhina@gmail.com',
    packages=['cs_sim',
              'cs_sim.synth',
              'cs_sim.corrupt',
              'cs_sim.batch'
              ],
    license='MIT',
    include_package_data=True,

    test_suite='cs_sim.tests',

    install_requires=[
        'scipy',
        'numpy',
        'pytest',
        'joblib',
        'scikit-image',
    ],
)
