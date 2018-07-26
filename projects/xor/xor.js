const tf = require('@tensorflow/tfjs');

require('@tensorflow/tfjs-node');

xs = tf.tensor2d([[1, 0], [0, 1], [0, 0], [1, 1]])
xy = tf.tensor2d([[1], [1], [0], [0]]);




const model = tf.sequential();

const hiddenLayer = tf.layers.dense({
    units: 2,
    inputShape: [2],
    activation: "relu",
    kernelInitializer: 'leCunNormal',
    useBias: true,
    biasInitializer: 'randomNormal'
    
});

const outputLayer = tf.layers.dense({
    units: 1,
    activation: "sigmoid"
});

model.add(hiddenLayer)
model.add(outputLayer)

const optimizer = tf.train.sgd(0.1)
model.compile({
    loss: tf.losses.meanSquaredError,
    optimizer: optimizer
});


const modelFit = async () => {

    for (let i = 1; i <= 200; i++) {

        const history = await model.fit(xs, xy, {
            epochs: 3,
            shuffle:true
        });

        console.log("Loss after Epoch " + i + " : "+history.history.loss[0]);
    }


};

modelFit().then(() => {
    console.log("Training completed");

    const result=model.predict(tf.tensor2d([[0,1]])).print();
    console.log("Prediction completed");

}).catch((err)=>{
    console.log(err);
})

