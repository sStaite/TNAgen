Installation
============

To use TNAgen, a Conda environment must be used - this is because python-framel (the python package that is used to write data to .gwf files) can only be managed by Conda. 
Annoyingly, torchGAN can only be installed via pip. This means the requirements are stored in a requirements.yml file. 

Conda:

.. code-block:: console

   $ conda install TNAgen

From source: 

.. code-block:: console

   $ git clone https://github.com/sStaite/TNAgen
   $ cd TNAgen
   $ python setup.py install



