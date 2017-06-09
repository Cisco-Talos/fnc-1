from theano import function
import numpy as np
# The prediction function for pylearn2.models.*
def predict(model, x, U = 1000):
    a = model.get_input_space().make_theano_batch()
    b = model.fprop(a)
    f = function([a], b)
    n = x.shape[0]
    yhat = []
    for i in range(n // U + 1):
        yhat.append(f(x[(i * U):((i + 1) * U),:]))
    return np.vstack(yhat)


 #   Copyright 2017 Cisco Systems, Inc.
 #  
 #   Licensed under the Apache License, Version 2.0 (the "License");
 #   you may not use this file except in compliance with the License.
 #   You may obtain a copy of the License at
 #  
 #     http://www.apache.org/licenses/LICENSE-2.0
 #  
 #   Unless required by applicable law or agreed to in writing, software
 #   distributed under the License is distributed on an "AS IS" BASIS,
 #   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 #   See the License for the specific language governing permissions and
 #   limitations under the License.
