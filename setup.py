from setuptools import setup

setup(
    name='dtia',
    version='0.1.0',
    description='Decision Tree Insight Analysis Tool',
    url='https://github.com/karim/dtia',
    author='Karim Hossny',
    author_email='k.hossny@kth.se',
    license='BSD 2-clause',
    packages=['dtia'],
    # install_requires=['scikit-learn',
    #                   'numpy',
    #                   'tqdm',
    #                   'graphviz',
    #                   'pandas'],

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux :: Windows :: MacOS',
        'Programming Language :: Python :: 3.8',
    ],
)
