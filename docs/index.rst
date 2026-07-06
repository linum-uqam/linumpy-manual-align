linumpy-manual-align
====================

**linumpy-manual-align** is the manual alignment widget and CLI companion to `linumpy <https://github.com/linum-uqam/linumpy>`_, for serial optical coherence tomography (S-OCT) slice-pair alignment and transform export.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Operator Guide
      :link: cli
      :link-type: doc

      CLI usage, data package layout, the napari widget walkthrough, transform semantics, and Nextflow pipeline handoff.

   .. grid-item-card:: CLI Reference
      :link: cli
      :link-type: doc

      Every ``linumpy-manual-align`` command-line option, generated from the argparse definition.

   .. grid-item-card:: Library API
      :link: api/index
      :link-type: doc

      Auto-generated reference for the ``linumpy_manual_align`` Python package.

   .. grid-item-card:: Architecture
      :link: architecture
      :link-type: doc

      Module boundaries, mixin responsibilities, napari hot paths, and the incremental refactor sequence.

.. toctree::
   :maxdepth: 1
   :hidden:

   cli
   architecture
   api/index
