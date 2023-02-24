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
        'Documentation': 'https://tnagen.readthedocs.io/en/latest/',
        'Source Code': 'https://github.com/sStaite/TNAgen',
    },
    include_package_data=True,
    package_dir={'': 'TNAgen'},
    packages=setuptools.find_packages(where='TNAgen'),
    classifiers=[
        'Development Status :: 3 - Alpha',
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10.8',
)