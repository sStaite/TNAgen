Examples
========

First, we need to import the Generator module and instatiate a Generator object.

.. code-block:: python

    from TNAgen import Generator
    generator = Generator.Generator()

To find a list of the names of glitches that can be generated, along with example spectrograms of each glitch, go to the Glitch Names tab.

Generating and saving 50 Koi Fish glitches as images of spectrograms:

.. code-block:: python

    generator.generate("Koi_Fish", 50, SNR = 10)
    generator.save_as_png("path/to/file", clear_queue=True)

Generating and saving 2 of each glitch, each with a SNR of 12, and saving as time-series in a gwf file:

.. code-block:: python

    generator.generate_all(2)
    generator.save_as_hdf5("path/to/file", "name", SNR=12, format="gwf", clear_queue=True)

Generating and saving 96 Blip glitches and saving as time-series in a gwf file:

.. code-block:: python

    generator.generate("Blip", 96)
    generator.save_as_hdf5("path/to/file", "name", format="hdf5", clear_queue=True)


Generating raw data (numpy arrays) for 50 Chirp glitches:

.. code-block:: python

    numpy_array, label_list = generator.generate("Chirp", 50)
    generator.clear_queue()

Generating raw data (numpy arrays) for 25 Tomte glitches and 25 Blip glitches and saving them to a hdf5 file:

.. code-block:: python

    generator.generate("Tomte", 25)
    generator.generate("Blip", 25)
    generator.save_as_array("path/to/file", "name")

    