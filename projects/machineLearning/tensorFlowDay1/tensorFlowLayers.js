//Importing only works using parcel
import * as tf from "@tensorflow/tfjs";

const model = tf.sequential();

//Outputs from one layer are inputs to another
// .dense means the layer is generic, no shenanigans
// 1 unit = 1 "neuron" in the brain, results passed down to 64 "neurons", then results from that passed down to single "result" neuron
//inputShape represents number of dimensions i.e. 64 neurons along x axis as opposed to x,y,z, etc
model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
model.add(tf.layers.dense({ units: 64, inputShape: [1] }));
model.add(tf.layers.dense({ units: 1, inputShape: [64] }));

model.compile({ loss: "meanSquaredError", optimizer: "sgd" });

//5 samples of 1 element each
//if x = 1, y = 2 ... x = 3, y = 6 ... and so on
const xs = tf.tensor2d([1, 2, 3, 4, 5], [5, 1]);
const ys = tf.tensor2d([2, 4, 6, 8, 10], [5, 1]);

//This is asking, given training set xs, what will the next value of ys be?
//the model is fitting xs to xy
//SO... you'd assume, given x = 6, y will equal 12.
//Code below tests this, outputting something around 11.9
model.fit(xs, ys, { epochs: 150 }).then(function() {
  //predict y given x = 6
  model.predict(tf.tensor2d([6], [1, 1])).print();
  model.predict(tf.tensor2d([200], [1, 1])).print();
});
