# Test fixtures

Hand-made, tiny stand-ins for the real datasets. They are small enough that
every expected number in the tests can be worked out by hand, and they let CI
run without downloading anything.

- `hemibrain/` has the same columns as Janelia hemibrain v1.2
  `exported-traced-adjacencies-v1.2`: 7 neurons and 10 directed connections.
  Three pairs are reciprocal (1<->4, 2<->4, 5<->6), so edge reciprocity is 6/10.
  Neuron 7 has no type.
- `flywire/neurons.tsv` has the FlyWire annotation columns the code uses
  (`root_id`, `super_class`, `top_nt`, `side`, `cell_type`): 6 neurons, with
  missing values in every column and mixed-case neurotransmitter names.
