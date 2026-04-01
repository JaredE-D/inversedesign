In this repo I used inverse design algorithms to optimize for different beamsplitting power ratios in a small 2x2 $\mu m$ area. 90:10 and 60:40 splitting ratios were also optimized for high power transmittance in a small footprint.

The code uses a multi-step beta approach where we use a Tanh projection of the weights to model the physical constraints of keeping features above 50nm. This is done to explore the design space while
achieving physical designs and iteratively decreasing feature size.
Simulations were performed with MEEP the open source FDTD solver. 
An example 60:40 beamsplitter simulation is shown here:

![fieldgif](./optresult\(6\)_field.gif)

