<p align="center">
<img src="https://github.com/Cisco-Talos/fnc-1/blob/master/images/solat-in-the-swen.gif" alt="TALOS IN THE NEWS"/>
</p>
 
# SOLAT IN THE SWEN - deep\_learning\_model

## Model description:

This model applies a [1D Convolutional Net](https://en.wikipedia.org/wiki/Convolutional_neural_network) on the headline and body text, represented at the word level using the Google News pretrained vectors. The output of this CNNs is then sent to a MLP with 4-class output (`agree`,`disagree`,`discuss`,`unrelated`) and trained end-to-end. The model was regularized using dropout (`p=.5`) in all Convolutional layers. All hyperparameters of this model were set to sensible defaults, however they were not further evaulated to find better choices. 

<p align="center">
<img src="https://github.com/Cisco-Talos/fnc-1/blob/master/images/diagrams_light/deep_model_light.png" alt="Deep Model Diagram"/>
</p>

The final model was trained on the [FNC-1](https://github.com/FakeNewsChallenge/fnc-1) baseline training set and evaluated against the baseline validation set. The highest scoring parameters during training were saved, then applied to the final test set. This approach scores roughly 3850 on the validation set. 

For more information on model selection and further research, please view our blog post (coming soon!).

## Installation:

This model requires a `Theano` installation using the `GpuArray` backend. Additionally, it requires `Cuda` with `CuDNN` to be correctly set up on the system. Replacing `CuDNN` Conv Ops with vanilla `Theano` Conv Ops may allow this code to be run on CPU, but was not tested.

 <!--
   Copyright 2017 Cisco Systems, Inc.
  
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
  
     http://www.apache.org/licenses/LICENSE-2.0
  
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
 -->
