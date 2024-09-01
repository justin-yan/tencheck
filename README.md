# tencheck

`tencheck` provides a simple set of utilities for analyzing and validating the properties of layers of deep neural nets.

It's typically quite difficult to validate that a layer "behaves properly" (oftentimes, the final barometer is simply how well a model performs with the layer included), and many brittle unit tests have been written involving randomly instantiated tensors and `is_close` checks.  We believe a good "first line of defense" for neural nets is to create a suite of properties that can be asserted about a layer, while requiring minimal effort per layer in order to do so.

There are two aspects of property-based testing that are quite useful to take inspiration from:

- Automatically generating inputs (and generating inputs of variable sizes and values to elucidate properties of interest).
- Evaluating properties based on the maintenance of invariants instead of attempting to exactly match values (which is particularly difficult to interpret in deep neural nets).

However, an important difference is that the properties of interest are generally fairly generic and often shared between layers, while the input generation strategies are pretty similar (they're all tensors).  So the focus of `tencheck` is to provide:

- An (attempted) universal input generation harness.
- A variety of interesting properties.
- Three modalities: assertion, analysis, and profiling.

The following requirements need to be met for `tencheck` to work:

- Your layers are implemented in torch.
- The `.forward()` method is annotated with [jaxtyping](https://github.com/patrick-kidger/jaxtyping).

## Usage

Create a unit test like the following and define the modules you'd like to have tested:

```python
from tencheck.harness import check_layers

def test_check_layers(self):
    LAYERS_TO_TEST = [
        MyTorchModule(c=4),
        MyTorchModule2(a=4, b=2),
    ]    
    check_layers(LAYERS_TO_TEST)
```

The `.forward()` method can be type annotated with the following "tencheck"-table types:

- Jaxtyped Tensors
- Dataclasses where each attribute is "tencheck"-able
- `list[tencheckable]`
  - Tencheck will always generate a list of length 2.
- `Optional[tencheckable]`
  - Tencheck will always generate the non-null value.
