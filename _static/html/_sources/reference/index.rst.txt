{{ header }}

.. _api:

=============
API reference
=============

This page gives an overview of all public pandas objects, functions and
methods. All classes and functions exposed in ``pandas.*`` namespace are public.

Some subpackages are public which include ``pandas.errors``,
``pandas.plotting``, and ``pandas.testing``. Public functions in
``pandas.io`` and ``pandas.tseries`` submodules are mentioned in
the documentation. ``pandas.api.types`` subpackage holds some
public functions related to data types in pandas.

.. warning::

    The ``pandas.core``, ``pandas.compat``, and ``pandas.util`` top-level modules are PRIVATE. Stable functionality in such modules is not guaranteed.

.. If you update this toctree, also update the manual toctree in the
   main index.rst.template

.. toctree::
   :maxdepth: 2

   measures
   models