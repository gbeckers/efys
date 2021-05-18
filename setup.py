import sys
import versioneer
import setuptools

if sys.version_info < (3,6):
    print("efys requires Python 3.6 or higher please upgrade")
    sys.exit(1)

long_description = \
"""
efys is a Python science library that enables you to work with 
electrofysiology data that is sampled uniformly in the time domain.

Efys is currently pre-1.0, still undergoing significant development.

"""

setuptools.setup(
    name='efys',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=['efys', 'efys.tests'],
    url='https://github.com/gbeckers/efys',
    license='BSD-3',
    author='Gabriel J.L. Beckers',
    author_email='gabriel@gbeckers.nl',
    description='A library for working with electrofysiology data',
    python_requires='>=3.6',
    install_requires=['numpy'],
    data_files = [("", ["LICENSE"])],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Education',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
    ],
    project_urls={  # Optional
        'Source': 'https://github.com/gbeckers/efys',
    },
)
