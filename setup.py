import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='TNAgen',
    author=['Seamus Staite', 'Jade Powell'],
    author_email=['seamus.staite@gmail.com', 'jade.powell@ligo.org'],
    description='A package that generates transient noise artifacts in gravitational-wave detector data.',
    keywords='gravitational-wave, noise artifacts, GANs',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/sStaite/TNAgen',
    project_urls={
        'Documentation': '',
        'Bug Reports':
        'https://github.com/sStaite/TNAgen/issues',
        'Source Code': 'https://github.com/sStaite/TNAgen',
    },
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src'),
    classifiers=[
        'Development Status :: 3 - Alpha',
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8.5',
)