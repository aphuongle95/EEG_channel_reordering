This is my implementation of the neural network in paper named 
"Learning from heterogeneous EEG Signals with differentiable channel reordering" 

Current stages:
- [x] implement dataset loader from left right dataset
- [x] implement basic Convolutional NN
- [x] implement convolutional reordering
- [ ] implement attentive reordering with canonical keys and values

Problems:
- when running with CHARM remapping, the loss becomes NAN after sometime
