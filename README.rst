.. image:: https://travis-ci.org/datasciencebr/rosie.svg?branch=master
   :target: https://travis-ci.org/datasciencebr/rosie
   :alt: Travis CI build status (Linux)

.. image:: https://readthedocs.org/projects/rosie/badge/?version=latest
   :target: http://rosie.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://landscape.io/github/datasciencebr/rosie/master/landscape.svg?style=flat
   :target: https://landscape.io/github/datasciencebr/rosie/master
   :alt: Code Health


Rosie, the robot
================

A Python application reading receipts from the Quota for Exercising
Parliamentary Activity (aka CEAP, from the Brazilian Chamber of
Deputies) and outputs, for each of the receipts, a "probability of
corruption" and a list of reasons why is considered this way.

-  [x] Fetch CEAP dataset from Chamber of Deputies
-  [x] Convert XML to CSV
-  [x] Translate CSVs to English
-  [x] Read CEAP dataset from CSV into Pandas
-  [ ] Process in the "corruption pipeline"

   -  [ ] Monthly limits - quota
   -  [x] Monthly limits - subquota
   -  [ ] Machine Learning models using scikit-learn

-  [ ] Task to push to Jarbas via API

Setup
-----

.. code:: console

    $ cd rosie
    $ conda update conda
    $ conda create --name serenata_rosie python=3
    $ source activate serenata_rosie
    $ ./setup

Running
-------

.. code:: console

    $ python rosie/main.py

A ``/tmp/serenata-data/irregularities.xz`` file will be created. It's a
compacted CSV with all the irregularities Rosie is able to find.

A `/tmp/serenata-data/irregularities.xz` file will be created. It's a compacted CSV with all the irregularities Rosie is able to find.


Full Documentation
------------------

https://rosie.readthedocs.io

Build documentation locally
---------------------------

You will need sphinx installed in your machine

::

  $ cd docs
  $ make clean;make rst;rm source/modules.rst;make html
  

Run Unit Test suite
-------------------

.. code:: console

    $ python -m unittest discover tests


Source Code
-----------

Feel free to fork, evaluate and contribute to this project.

Source: https://github.com/datasciencebr/rosie

License
-------

MIT licensed.

