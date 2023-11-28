from setuptools import setup, find_packages

setup(
    name='project2',
    use_scm_version=True,
    description='A Python package for parsing XML sitemaps, fetching data from APIs, and scraping web content.', 
    author='Qing Dou', 
    author_email='qdou@mail.yu.edu', 
    license='MIT', 
    url='https://github.com/qdou14/project2_movie_analysis', 
    packages=find_packages(), 
    install_requires=[
        'matplotlib>=3.8.2',
        'numpy>=1.26.1',
        'pandas>=2.1.1',
        'seaborn>=0.13.0',
        'requests>=2.25.1', 
        'beautifulsoup4>=4.9.3'
    ],
    setup_requires=['setuptools_scm'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Education',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.9',
)