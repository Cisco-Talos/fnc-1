<p align="center">
<img src="https://github.com/Cisco-Talos/fnc-1/blob/master/images/solat-in-the-swen.gif" alt="TALOS IN THE NEWS"/>
</p>

# Fake News Challenge - Team SOLAT IN THE SWEN

**Team Members:**     [Yuxi Pan](https://www.linkedin.com/in/yuxipanucla) (yuxpan@cisco.com), Doug Sibley (dosibley@cisco.com), [Sean Baird](https://www.linkedin.com/in/seanrichardbaird/) (sebaird@cisco.com)

In the below directories, you can find code used by Team SOLAT IN THE SWEN to perform stance detection on a number of news headlines and article text. Our model is based on an weighted average between gradient-boosted decision trees and a deep convolutional neural network. 

<p align="center">
<img src="https://github.com/Cisco-Talos/fnc-1/blob/master/images/diagrams_light/final_prediction_light.png" alt="Our ensemble used a 50/50 weighting" width="50%"/>
</p>

Both `tree_model` and `deep_learning_model` contain their own `README.md` files detailing their model and providing instructions for running and installation. The model averaging process is described in `tree_model`.

To learn more about how stance detection can help to detect fake news and disinformation campaigns, please visit our blog post on [blog.talosintelligence.com/2017/06/talos-fake-news-challenge.html](https://blog.talosintelligence.com/2017/06/talos-fake-news-challenge.html).

For those interested, `tree_model/README.md` has detailed information on how to run our models to duplicate our results.

See [Fake News Challenge Official Website](http://www.fakenewschallenge.org/) for more information.
Thank you to Wendy, Melissa, and the entire Talos art team for the graphics, and to Joel and Luci for helping to open source our solution.  Big thank you to our leadership team, as well, for allowing us the time to work on this important problem.

**Interested in learning how Talos forces the bad guys to innovate?**  Visit [talosintelligence.com](https://talosintelligence.com/about)

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
