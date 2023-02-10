Examples
========

First, we need to import the Generator module and instatiate a Generator object.

.. code-block:: python

    from TNAgen import Generator
    generator = Generator.Generator()

Generating and saving 50 Koi Fish glitches as images of spectrograms:

.. code-block:: python

    generator.generate("Koi_Fish", 50)
    generator.save_as_png("path/to/file", clear_queue=True)

Generating and saving 2 of each glitch and saving as time-series in a hdf5 file:

.. code-block:: python

    generator.generate_all(2)
    generator.save_as_hdf5("path/to/file", "name", SNR=12, clear_queue=True)

Generating and saving 96 Blip glitches and saving as time-series in a gwf file:

.. code-block:: python

    generator.generate("Blip", 96)
    generator.save_as_hdf5("path/to/file", "name", clear_queue=True)


Generating raw data (numpy arrays) for 50 Chirp glitches:

.. code-block:: python

    numpy_array, label_list = generator.generate("Chirp", 50)
    generator.clear_queue()

Generating raw data (numpy arrays) for 25 Tomte glitches and 25 Blip glitches and saving them to a hdf5 file:

.. code-block:: python

    generator.generate("Tomte", 25)
    generator.generate("Blip", 25)
    generator.save_as_array("path/to/file", "name")
    