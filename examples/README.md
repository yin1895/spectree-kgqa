# Mira running example

The example graph follows the named entities and relation paths appearing in Figures 1 and 2. `c1` can correspond to Mira → University of Oxford → Oxford → United Kingdom; `c2` uses the partner-university branch to Canada; and `c3` uses Mira's birthplace branch to France.

The example also includes the optional `graduated_from` fact. Search results are assigned stable IDs by actual completion order; the figure's c1/c2/c3 labels are illustrative references to these paths, not a hard-coded runtime order.

Run with `spectree-demo` or `python scripts/run_kgqa.py ... --backend demo`.
