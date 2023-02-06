Examples
========

First, we need to import the Generator module and instatiate a Generator object.

.. code-block:: python

    from TNAgen import Generator
    generator = Generator()

Generating and saving 50 Koi Fish glitches as images of spectrograms:

.. code-block:: python

    generator.generate("Koi_Fish", 50)
    generator.save_as_png("path/to/file", clear_queue=True)

Generating and saving 2 of each glitch and saving as time-series in a hdf5 file:

.. code-block:: python

    generator.generate_all(2)
    generator.save_as_hdf5("path/to/file", "name", clear_queue=True)

Generating raw data (numpy arrays) for 50 Blip glitches.

.. code-block:: python

    numpy_array, label_list = generator.generate("Blip", 50)
    generator.clear_queue()

Generating 10 Blip glitches, 10 Koi Fish glitches and saving to one hdf5 file, then generating 50 Chirp glitches and saving in another hdf5 file.

.. code-block:: python

    generator.generate("Blip", 10)
    generator.generate("Koi Fish", 10)
    generator.save_as_hdf5("path/to/file", "file1", clear_queue=True)
    generator.generate("Chirp", 50)
    generator.save_as_hdf5("path/to/file", "file2", clear_queue=True)