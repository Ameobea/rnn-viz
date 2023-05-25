Need to implement post layers

## OPTIMIZATIONS

We need to prune connected inputs that are not themselves somehow connected to a root input.

The simplest example of this is a node with no inputs connected to another node. That node can automatically be pruned by taking its bias and adding it to the bias of the node it's connected to.

Things get more complicated when there are loops to consider. We can have a graph like this:

a -> b -> a
b -> c

In this case, node a has no inputs that reach to the root inputs, so a can be pruned. This will result in b now having no inputs, so we can prune it as well.

Here's an excellent test case that can be pruned down to an optimal solution:

```
digraph "RNN" {
subgraph "cluster_outputs" {
  "output_0";
  "output_1";
}

// subgraph "cluster_layer_0" {
// subgraph "cluster_state" {
  "layer_0_state_1";
  "layer_0_state_2";
  "layer_0_state_3";
  "layer_0_state_4";
  "layer_0_state_5";
  "layer_0_state_6";
// }

// subgraph "cluster_recurrent" {
  "layer_0_recurrent_0";
  "layer_0_recurrent_1";
  "layer_0_recurrent_2";
  "layer_0_recurrent_3";
  "layer_0_recurrent_4";
  "layer_0_recurrent_5";
  "layer_0_recurrent_6";
  "layer_0_recurrent_7";
// }

// subgraph "cluster_output" {
  "layer_0_output_0";
  "layer_0_output_1";
// }

// }

subgraph "cluster_inputs" {
    color=lightblue;
  "input_0";
}

  "layer_0_output_0";
  "output_0";
  "layer_0_state_4";
  "layer_0_recurrent_4";
  "layer_0_state_6";
  "layer_0_recurrent_6";
  "layer_0_output_1";
  "output_1";
  "input_0";
  "layer_0_state_2";
  "layer_0_recurrent_2";
  "layer_0_state_3";
  "layer_0_recurrent_3";
  "layer_0_state_1";
  "layer_0_recurrent_1";
  "layer_0_state_5";
  "layer_0_recurrent_5";
  "layer_0_output_0" -> "output_0" [ label = "1" ];
  "layer_0_state_4" -> "layer_0_output_0" [ label = "1" ];
  "layer_0_recurrent_4" -> "layer_0_state_4" [ label = "1" ];
  "layer_0_state_6" -> "layer_0_output_0" [ label = "1" ];
  "layer_0_recurrent_6" -> "layer_0_state_6" [ label = "1" ];
  "layer_0_state_6" -> "layer_0_recurrent_6" [ label = "-1" ];
  "layer_0_output_1" -> "output_1" [ label = "1" ];
  "input_0" -> "layer_0_output_1" [ label = "1" ];
  "layer_0_state_2" -> "layer_0_output_1" [ label = "-1" ];
  "layer_0_recurrent_2" -> "layer_0_state_2" [ label = "1" ];
  "layer_0_state_3" -> "layer_0_recurrent_2" [ label = "-1" ];
  "layer_0_recurrent_3" -> "layer_0_state_3" [ label = "1" ];
  "layer_0_state_1" -> "layer_0_recurrent_3" [ label = "-1" ];
  "layer_0_recurrent_1" -> "layer_0_state_1" [ label = "1" ];
  "input_0" -> "layer_0_recurrent_1" [ label = "1" ];
  "layer_0_state_5" -> "layer_0_recurrent_3" [ label = "-1" ];
  "layer_0_recurrent_5" -> "layer_0_state_5" [ label = "1" ];
  "layer_0_state_6" -> "layer_0_recurrent_5" [ label = "1" ];
}
```

Weights:

```
my_rnn_MyRNN1/RNNCell_0/output_tree (2) [9, 2]
print.ts:34 Tensor
    [[-0.0013225, 0.9985099 ],
     [0.0007498 , -0.0020642],
     [-0.0007498, 0.0008217 ],
     [-0.0021982, -1.0128093],
     [0.0008968 , 0.0007873 ],
     [1.0006403 , -0.0029936],
     [-0.000896 , 0.0006999 ],
     [0.9987024 , -0.0005651],
     [-0.0004978, 0.0037821 ]]
+page.svelte:218 my_rnn_MyRNN1/RNNCell_0/output_bias [2]
print.ts:34 Tensor
    [0.9989777, -1.0013669]
+page.svelte:218 my_rnn_MyRNN1/RNNCell_0/recurrent_tree (2) [9, 8]
print.ts:34 Tensor
    [[-0.9997373, 0.9956539 , 0.000069  , 0.0019652 , 0.0013265 , -0.0015785, 0.0008119 , 0.0022041 ],
     [-0.0005561, -0.0006266, -0.0019011, -0.0019507, -0.0019737, 0.0016918 , -0.0011887, -0.0007545],
     [0.0021989 , 0.0005424 , 0.0001253 , -1.0082515, 0.0013196 , 0.0022336 , 0.0008588 , -0.0013228],
     [-0.9955506, 0.0004425 , 0.0011675 , -0.000748 , -0.0019669, 0.000623  , 0.0002603 , -0.0019642],
     [0.0005577 , 0.0006392 , -1.0231484, -0.0029067, 0.0013267 , 0.0015787 , 0.0000826 , -0.0004972],
     [-0.00075  , 0.0002005 , 0.000058  , -0.000007 , 0.0005476 , -0.0003582, -0.0029444, -0.0008827],
     [-0.0018699, -0.0003611, 0.0006322 , -0.9921619, 0.0005009 , -0.0028259, 0.0001542 , -0.0009095],
     [0.000497  , 0.0005881 , 0.0013203 , 0.0008414 , 0.000553  , 1.0018713 , -0.9985552, 0.9983892 ],
     [-0.0007047, 0.0027038 , 0.0031361 , -0.001058 , 0.000491  , 0.0008989 , 0.0034093 , -0.0013113]]
+page.svelte:218 my_rnn_MyRNN1/RNNCell_0/recurrent_bias [8]
print.ts:34 Tensor
    [1.0007997, -0.0004942, 0.0028694, 1.0000962, -1.0016478, 1.002045, -0.0014043, -0.9999518]
+page.svelte:218 my_rnn_MyRNN1/initial_state_weights_0 [8]
print.ts:34 Tensor
    [0.6103195, -0.1184926, -0.9936724, -0.9714772, -0.2415891, 0.144228, -0.0315875, 1.354445]
+page.svelte:218 dense_Dense1/kernel (2) [2, 1]
print.ts:34 Tensor
    [[-0.0013149],
     [-1.0012252]]
```

One thing we might have to add special handling for is initial states... Or do we? I think we might. It may only become clear if we unroll the RNN graph through time.

initial_state -> a -> state[0] -> a -> state[1] -> a -> ...
b -> a
a -> c

```
digraph "RNN" {
  initial_state -> a0 -> state0 -> a1 -> state1 -> a2 -> state3
  b0 -> a0
  b1 -> a1
  b2 -> a2
  a0 -> c0
  a1 -> c1
  a2 -> c2

  c0 -> out
  c1 -> out
  c2 -> out
}
```

To start off, we can prune B since it has _no_ inputs. In this way, the separation of state from neurons helps us out since there is no state involved at all - just the bias.

That leaves us with this:

```
digraph "RNN" {
  initial_state -> a0 -> state0 -> a1 -> state1 -> a2 -> state3
  a0 -> c0
  a1 -> c1
  a2 -> c2

  c0 -> out
  c1 -> out
  c2 -> out
}
```

Or:

```
digraph x {
  a -> state -> a
  a -> c
}
```

Hmmm... Yeah I don't think we can prune this. At least not in all cases.

Consider a case where `a` inverts the state every time through. This could implement a circuit that alternates between two values every step of the sequence.

Because of that, we can't automatically prune it. We'll have to treat initial state as root nodes as well.

The exception is if the weight is -1. For the Ameo activation function, a weight of -1 makes it the identity function. This makes the loop the equivalent of a constant value, and we can merge it into the bias of nodes it's connected to.

OK well that's interesting. Given that, we should still be able to prune that example graph down to optimal.

We should also handle loops with more than two nodes. Might be more difficult or even impossible, though. Yeah how the hell can we even do that?

```
digraph x {
  a -> a_state -> b -> b_state -> a
  a -> c
}
```

The only way we can prune the loop is if we can prove that it will output the same value from a to c every step of the sequence from the start forever.

I'm not going to worry about this for now. I think we can get some decent mileage out of just pruning 1-loops and non-loops.

### More Optimizations

Here's a new one:

```
digraph "RNN" {
// subgraph "cluster_outputs" {
  "output_0";
// }

// subgraph "cluster_layer_0" {
// subgraph "cluster_state" {
  "layer_0_state_0";
  "layer_0_state_1";
  "layer_0_state_3";
  "layer_0_state_4";
  "layer_0_state_5";
  "layer_0_state_6";
// }

// subgraph "cluster_recurrent" {
  "layer_0_recurrent_0";
  "layer_0_recurrent_1";
  "layer_0_recurrent_2";
  "layer_0_recurrent_3";
  "layer_0_recurrent_4";
  "layer_0_recurrent_5";
  "layer_0_recurrent_6";
  "layer_0_recurrent_7";
// }

// subgraph "cluster_output" {
  "layer_0_output_0";
  "layer_0_output_1";
// }

// }

subgraph "cluster_post_layer_0" {
  "post_layer_output_0";
}

subgraph "cluster_inputs" {
  "input_0";
}

  "post_layer_output_0";
  "output_0";
  "layer_0_output_1";
  "input_0";
  "layer_0_state_0";
  "layer_0_recurrent_0";
  "layer_0_state_3";
  "layer_0_recurrent_3";
  "layer_0_state_1";
  "layer_0_recurrent_1";
  "layer_0_state_5";
  "layer_0_recurrent_5";
  "post_layer_output_0" -> "output_0" [ label = "1" ];
  "layer_0_output_1" -> "post_layer_output_0" [ label = "-1" ];
  "input_0" -> "layer_0_output_1" [ label = "-1" ];
  "layer_0_state_0" -> "layer_0_output_1" [ label = "1" ];
  "layer_0_recurrent_0" -> "layer_0_state_0" [ label = "1" ];
  "layer_0_state_3" -> "layer_0_recurrent_0" [ label = "-1" ];
  "layer_0_recurrent_3" -> "layer_0_state_3" [ label = "1" ];
  "layer_0_state_1" -> "layer_0_recurrent_3" [ label = "1" ];
  "layer_0_recurrent_1" -> "layer_0_state_1" [ label = "1" ];
  "input_0" -> "layer_0_recurrent_1" [ label = "-1" ];
  "layer_0_state_5" -> "layer_0_output_1" [ label = "1" ];
  "layer_0_recurrent_5" -> "layer_0_state_5" [ label = "1" ];
  "layer_0_state_3" -> "layer_0_recurrent_5" [ label = "-1" ];
}
```

a -> b -> c
a -> d -> c

Should be able to merge these in some or all cases. It will be trivial to merge in the case that the weights are the same (just add the final weights together and remove one).

Might be a bit tricker to merge things with different weights.

Just had an idea. We can find "equivalent nodes" in the network. Like if we track from the source and we have two nodes like this:

x -> x -> !x -> a
x -> !x -> x -> b

We can say that those end nodes are equivalent. We can delete one of them and connect all its outputs to the other one. I quite like that.

We'll have to add special handling for if a and b are actually the same node though. God this is starting to get complex...

We can track a sort of lineage that way. I really don't know how to implement that cleanly, but we can certainly do it.
