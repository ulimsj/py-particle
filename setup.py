import setuptools

setuptools.setup(
    name='lmj.particle',
    version='0.1',
    install_requires=['numpy'],
    namespace_packages=['lmj'],
    packages=setuptools.find_packages(),
    author='Leif Johnson',
    author_email='leif@leifjohnson.net',
    description='',
    long_description=open('README.md').read(),
    license='MIT',
    keywords=('particle '
              'filter '
              'probabilistic '
              'modeling '
              'machine '
              'learning'),
    url='http://github.com/lmjohns3/py-particle/',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ],
    )
