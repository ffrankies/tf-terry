
$z$ - update gate

$r$ - reset gate

$h$ - hidden state

$s_t$ - GRU state at timestep t (Output vector)

$x$ - Input vector

$\odot$ - Dot product

$U^{z|r|h}$ - Weights from input layer to update gate | reset gate | hidden state

$W^{z|r|h}$ - Weights from hidden layer to update gaet | reset gate | hidden state

$$z = \sigma(x \odot U^z + s_{t-1} * W^z)$$

$$r = \sigma(x \odot U^r + s_{t-1} * W^r)$$

$$h = \tanh(x \odot U^h + (s_{t-1} * r) \odot W^h)$$

$$s_t = (1 - z) * h + z * s_{t-1}$$


```python

```
