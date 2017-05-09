========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis|
        | |coveralls|
        | |codeclimate|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|

.. |docs| image:: https://readthedocs.org/projects/rosie/badge/?style=flat
    :target: https://readthedocs.org/projects/rosie
    :alt: Documentation Status

.. |travis| image:: https://travis-ci.org/datasciencebr/rosie.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/datasciencebr/rosie

.. |coveralls| image:: https://coveralls.io/repos/datasciencebr/rosie/badge.svg?branch=master&service=github
    :alt: Coverage Status
    :target: https://coveralls.io/r/datasciencebr/rosie

.. |codeclimate| image:: https://codeclimate.com/github/datasciencebr/rosie/badges/gpa.svg
   :target: https://codeclimate.com/github/datasciencebr/rosie
   :alt: CodeClimate Quality Status

.. |version| image:: https://img.shields.io/pypi/v/rosie.svg
    :alt: PyPI Package latest release
    :target: https://pypi.python.org/pypi/rosie

.. |commits-since| image:: https://img.shields.io/github/commits-since/datasciencebr/rosie/v0.1.0.svg
    :alt: Commits since latest release
    :target: https://github.com/datasciencebr/rosie/compare/v0.1.0...master

.. |wheel| image:: https://img.shields.io/pypi/wheel/rosie.svg
    :alt: PyPI Wheel
    :target: https://pypi.python.org/pypi/rosie

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/rosie.svg
    :alt: Supported versions
    :target: https://pypi.python.org/pypi/rosie

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/rosie.svg
    :alt: Supported implementations
    :target: https://pypi.python.org/pypi/rosie


.. end-badges

A Python application reading receipts from the `Quota for Exercising Parliamentary Activity (aka CEAP)`_ from the
Brazilian Chamber of Deputies and outputs, for each of the receipts, a **probability of corruption** and a list of reasons
why it was considered this way.

.. _`Quota for Exercising Parliamentary Activity (aka CEAP)`: https://github.com/datasciencebr/serenata-de-amor/blob/master/CONTRIBUTING.md#more-about-the-quota-for-exercising-parliamentary-activity-ceap


* Free software: MIT license

Installation
============

::

    pip install rosie

Documentation
=============

https://rosie.readthedocs.io/

Development
===========

To run the all tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
