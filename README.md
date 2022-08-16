# Neural Network Experiements + Visualizations

![A 5-second looping video showing a spinning 3D wireframe cube.  The cube's corners are labeled with truth tables in the form TTF, TFT, FFF, etc.  The cube is partially filled with solid voxels representing areas where the artificial neuron it's visualizing outputs a value greater than 0.](./static/logic-cube-header.webp)

This repository contains visualizations and demos that are used in a [blog post](https://cprimozic.net/blog/boolean-logic-with-neural-networks/) I wrote about logic in the context of neural networks. The best place to head to get more info on the stuff here is to read that!

It also includes TF.JS implementations of the Ameo activation function as introduced in that post.

Tech used to build these demos includes [uPlot](https://github.com/leeoniya/uPlot) and [eCharts](https://echarts.apache.org/) for charts + graphs, [Three.JS](https://threejs.org/) for the 3D voxel-based function plot seen above, and [TensorFlow.JS](https://github.com/tensorflow/tfjs) for implementing the Ameo activation function and training the binary additional model referenced in the last part of the blog post.
