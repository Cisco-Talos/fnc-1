<p align="center">
<img src="https://github.com/Cisco-Talos/fnc-1/blob/master/solat-in-the-swen.gif" alt="TALOS IN THE NEWS"/>
</p>
 
# SOLAT IN THE SWEN - deep\_learning\_model

## Model description:

This model applies a set of [1D Convolutional Nets](https://en.wikipedia.org/wiki/Convolutional_neural_network) on both the headline and body text, represented at the word level using the Google News pretrained vectors. The output of these two CNNs is then sent to a MLP with 4-class output (`agree`,`disagree`,`discuss`,`unrelated`) and trained end-to-end. The model was regularized using dropout (`p=.5`) in all Convolutional layers. All hyperparameters of this model were set to sensible defaults, however they were not further evaulated to find better choices. 

The final model was trained on the [FNC-1](https://github.com/FakeNewsChallenge/fnc-1) baseline training set and evaluated against the baseline validation set. The highest scoring parameters during training were saved, then applied to the final test set. This approach scores roughly 3850 on the validation set. 

For more information on model selection and further research, please view our blog post (coming soon!).

## Installation:

This model requires a `Theano` installation using the `GpuArray` backend. Additionally, it requires `Cuda` with `CuDNN` to be correctly set up on the system. Replacing `CuDNN` Conv Ops with vanilla `Theano` Conv Ops may allow this code to be run on CPU, but was not tested.
