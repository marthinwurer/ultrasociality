# Things to look into

Use `rasterstats` to aggregate the data.

Use raw `rasterio` to load the data.

Use `shapely` to build boxes to do run stats on.

Plot a shapely polygon:
https://stackoverflow.com/a/56140178/3000741







## Model parameters

Grid size:
- Many people tend to walk at about 1.4 m/s (5.0 km/h; 3.1 mph; 4.6 ft/s)
- Towns in colonial america are about 10 miles apart (5 miles each way), 2 hours each way.
- What is the spatial frequency of polities?

Look at computing the fertility of the land to see the max population that can be supported/carrying capacity.
NPP? (net primary production)

http://www.montana.edu/hansenlab/documents/bio494/slides/Lec4.pdf
Chapin et al. 2011 Figs 6.2
Huston and Wolverton 2009 Fig 4
Light received is also important - 

Effective temperature takes into account humidity, wind, cloud cover, and air temperature to produce the way a person (or plant) actually feels.
















## Cortical model
Inputs
thalmus attention - inputs and past database queries/answers, along with statistics of the last few frames.
use that to index into database
take database info, then decide on action

database is a neural turing machine or something

All layers connect to same layer. Some connect to others for input

feedforward and feedback inputs to layers.
feedforward and feedback (up and down the hierarchy)


the hippocampus, which is at the top of cortical hierarchy
prefrontal cortex that sits near the top of the cortical hierarchy

The hippocampus sits at the very top of the cortical hierarchy.
Note that the hippocampus only stores very basic representations of objects;
their full descriptions are stored in the sensory parts of the cortex.

### Parts of neuron

as long as inputs are perfectly predicted, they are inhibited

Tuft, apical, base dendrites. Tuft are at end of apical. base are around soma.
Tuft and base have NDMA spikes.
The apical dendrite can end in different layers
The apical dendrites of excitatory cells always go upwards (towards the lower-numbered layers, or the skull) from the cell body.

Like the apical dendrite, axonal targets are also specific by layer.

sequence memory is the domain of basal dendrites – dendrites that are connected to the soma. 

Bursting occurs when there is apical input and basal input.
Calcium spikes cause neurons to burst.
Bursting means that the neuron spikes 2-4 times at 200hz

inhibitory neurons can target (and effectively “turn off ”) specific dendritic branches
as well as calcium spike initiation zones
A neuron can spike continually at a certain frequency (30 Hz, for example) for a long time.
Some inhibitory neurons fire regularly without inputs.
That can help excitatory neurons maintain frequency by 
- only allowing them to spike during certain time windows
- inhibiting them during others.
That way excitatory neurons get synchronized.

Neurons can also spontaneously discharge. (add random voltage to iaf?)
They may work, or they may not, in any given case.

### Layer connectivity

Thalmus is IO Layer.
Maps to 6, 5, and 4.
6 and 5 expand it out. 


Layer 4: input from thalmus terminates
Higher layers recieve input from lower layers here
Higher level of in-layer connectivity

Layer 6 neurons send their apical dendrites to Layer 4

Layer 3 neurons receive feedforward input to their basal dendrites from layer 4. 
Layer 3 searches for sequences in the spiking of layer 4 neurons
Layer 3 neurons’ apical dendrites extend into layer 1.

In layer 3 (or maybe others too?): Some inhibitory neurons, called Martinotti cells, respond to bursts of excitatory neurons 52 . They then inhibit nearby excitatory neurons.

Layer 3 neurons send information to a few different places.
layer 3 neurons project to layer 4 neurons in a higher level of the hierarchy
layer 5 in the same region
The third place they project to is layer 2 in the same region

Layer 2: Short term memory.
Layer 2 neurons receive local input from layer 3 and layer 4 neurons
their tuft dendrites are in layer 1
there probably exists a feedback mechanism to stop layer 2 neurons from firing
layer 2 neurons projecting to layer 2 neurons in lower hierarchical regions
Layer 2 neurons send output to tuft dendrites of layer 3 neurons.
They also send output to layer 5 neurons
An interesting feature of both layer 2 and layer 5 feedback connections is that they arrive to layer 1 of the lower cortical region.

Layer 5 is the output layer of the neocortex
Neurons in that layer extend axons to various subcortical structures, which is how they influence behavior.
Layer 5 neurons receive inputs from all layers of the cortex
Layer 5 neurons can fire regularly, as well as burst
There are also Martinotti cells in layer 5
Some layer 5 neurons send feedback information to lower regions in the hierarchy.
An interesting feature of both layer 2 and layer 5 feedback connections is that they arrive to layer 1 of the lower cortical region.
Layer 5 neurons send their apical dendrites to Layer 1

the purpose of layer 1 is clear – that’s where layer 2, 3 and 5 neurons receive feedback information to their tuft dendrites from farther up the hierarchy.


How does all of this relate to branch prediction?
If you use taken/not taken as a spike or no spike, can we use branch prediction algorithms for spiking data?

So, those are for connections.
We need to use leaky integrate-and-fire for each dendrite.
Previous * decay * weights

dendrites only connect to axons and same-neuron dendrites.
Axons will just be outputs of specific dendrites exported to different layers.

