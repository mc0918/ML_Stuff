//SEE WEBSITE TO DOWNLOAD PACKAGE IF I WANT TO USE NODE.JS

//Dependencies
import "bootstrap/dist/css/bootstrap.css";

import * as tf from "@tensorflow/tfjs";

const one = tf.tensor1d([1, 2, 3, 4]);
const two = tf.tensor1d([10, 20, 30, 40]);

tf.add(one, two).print(); //same as: a.add(b).print();

const y = tf.tidy(() => {
  // aa, b, and two will be cleaned up when the tidy ends.
  const two = tf.scalar(2);
  const aa = tf.scalar(2);
  const b = aa.square();

  console.log("numTensors (in tidy): " + tf.memory().numTensors);

  // The value returned inside the tidy function will return
  // through the tidy, in this case to the variable y.
  return b.add(two);
});

console.log("numTensors (outside tidy): " + tf.memory().numTensors);
y.print();
