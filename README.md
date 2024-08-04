# tencheck

`tencheck` provides a simple set of utilities for analyzing and validating the properties of layers of deep neural nets.

It's typically quite difficult to validate that a layer "behaves properly" (oftentimes, the final barometer is simply how well a model performs with the layer included), and many brittle unit tests have been written involving randomly instantiated tensors and `is_close` checks.  We believe a good "first line of defense" for neural nets is to create a suite of properties that can be asserted about a layer, while requiring minimal effort per layer in order to do so.

We think there are two aspects of property-based testing that are quite useful to take inspiration from:

- Automatically generating inputs (and generating inputs of variable sizes and values to elucidate properties of interest).
- Evaluating properties based on the maintenance of invariants instead of attempting to exactly match values (which is particularly difficult to interpret in deep neural nets).

However, an important difference is that the properties of interest are generally fairly generic and often shared between layers, while the input generation strategies are pretty similar (they're all tensors).  So the focus of `tencheck` is to provide:

- An (attempted) universal input generation harness.
- A variety of interesting properties.
- Three modalities: assertion, analysis, and profiling.

The following requirements need to be met for `tencheck` to work:

- Your layers are implemented in torch.
- The `.forward()` method is annotated with [jaxtyping](https://github.com/patrick-kidger/jaxtyping).


## Assertion

The assertion modality is simple:

- For each layer:
  - Introspect the arguments of the forward pass and automatically generate matching tensors.
  - Run a forward and backward pass with a trivial loss.
  - Assert properties based on the resultant outputs and tensor states.


## Properties

- Smoke-test: The layer runs forward and backward with a trivial loss without error.
- Shape-check: Turn on [jaxtyping's pytest beartype integration](https://docs.kidger.site/jaxtyping/api/runtime-type-checking/#pytest-hook) to shape check your outputs and intermediates.


## Backlog

- Represent properties explicitly.
  - Analysis modality.
- Profiling modality and properties.
- Scaling input sizes to produce performance curves.
- Tensor container types include more options like dataclasses.
- Auto-generate simple hyperparameters for layer instantiation.
- Refine dtype mapping and coherence.
