<p align="center">
<img src="https://github.com/Cisco-Talos/fnc-1/blob/master/solat-in-the-swen.gif" alt="TALOS IN THE NEWS"/>
</p>

# Fake News Challenge - Team SOLAT IN THE SWEN

In the below directories, you can find code used by Team SOLAT IN THE SWEN to perform stance detection on a number of news headlines and article text. Our model is based on an weighted average between a deep convolutional neural network and a gradient-boosted decision trees. 

<p align="center">
<img src="https://github.com/Cisco-Talos/fnc-1/blob/master/images/diagrams_light/final_prediction_light.png" alt="Our ensemble used a 50/50 weighting" width="50%"/>
</p>

Both `tree_model` and `deep_learning_model` contain their own `README.md` files detailing their model and providing instructions for running and installation. The model averaging process in described in `tree_model`

For those interested, `tree_model/README.md` has detailed information on how to run our models to duplicate our results.

**Primary Authors:**     Doug Sibley (dosibley@cisco.com), [Yuxi Pan](https://www.linkedin.com/in/yuxipanucla) (yuxpan@cisco.com)

**Primary Organizer:**  [Sean Baird](https://www.linkedin.com/in/seanrichardbaird/) (seanrichardbaird@gmail.com)

See [Fake News Challenge Official Website](http://www.fakenewschallenge.org/) for more information.
Thank you to Wendy for the logo, and to Joel and Luci for helping to open source our solution.

To learn more about how Talos forces the bad guys to innovate, visit [talosintelligence.com](https://talosintelligence.com/about)

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
