# hackday
This script contains a basic U-net, coded using Keras Functional API, that can be used as a jumping off point for more advanced deep learning.

To improve the model, and increase your understanding of deep learning, I recommend trying the following:
<ul>
<li>Change the loss function (https://keras.io/api/losses/, and maybe try coding a custom function - there's lots of help online for that)</li>
<li>Play with different optimizers and learning rates (https://keras.io/api/optimizers/)</li>
<li>Try adding an exponential decay to your learning rate using a learning rate scheduler (https://keras.io/api/optimizers/learning_rate_schedules/exponential_decay/)</li>
<li>Add callbacks such as ModelCheckpoint, which saves your model at regular intervals during training (https://keras.io/api/callbacks/)</li>
<li>Add batch normalization (rescales inputs for more consistency between batches) (https://keras.io/api/layers/normalization_layers/batch_normalization/)</li>
<li>Try different activation functions for you convolutional layers (https://keras.io/api/layers/activations/)</li>
<li>Look at other regularization layers (eg. different types of dropout) https://keras.io/api/layers/regularization_layers/</li>
<li>Install Tensorboard in your environment and add a tensorboard callback (https://keras.io/api/callbacks/tensorboard/) to your model so you can visualize training with nice graphs (https://www.tensorflow.org/tensorboard/get_started)</li>
<li>Read this intro on the keras website for a better understanding about how tensorflow and keras works: https://keras.io/getting_started/intro_to_keras_for_researchers/</li>
<li>Try adding more blocks of convolutional layers, but make sure the sizes match up in the encoding and decoding branches (use print(model.summary()) to see the sizes of each layer)</li>
<\ul>
