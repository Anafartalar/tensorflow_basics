const tf = require('@tensorflow/tfjs');
const Path = require("path");
const Data = require("./data");
const Utils = require("../../utils");

require('@tensorflow/tfjs-node');

const dataFilePath = Path.join(__dirname, "dataset", "Churn_Modelling.csv");


const main = async () => {

    let dataFrame = await Data.loadFromCSV(dataFilePath);

    dataFrame = Utils.normalizeFeatures(dataFrame, ["CreditScore", "Age", "Tenure",
        "Balance", "NumOfProducts", "EstimatedSalary"]);

    dataFrame = Utils.oneHotEncoder(dataFrame, ["Geography", "Gender"]);

    // turn text value into float
    dataFrame = Utils.castValuesToFloat(dataFrame, ["HasCrCard", "IsActiveMember"]);

    let features = dataFrame.drop("Exited").toArray();// get futures
    let labels = dataFrame.select("Exited").toArray();// get labels

    // prepare traning and test data, 80% traning
    let [xs_train, xs_test] = tf.split(tf.tensor2d(features), [8000, 2000]);
    let [ys_train, ys_test] = tf.split(tf.tensor2d(labels), [8000, 2000]);


    // Build the model
    const model = tf.sequential();

    const hiddenLayer = tf.layers.dense({
        units: 7, // number of neurons
        inputShape: [13],
        activation: "relu"
    });

    const outputlayer = tf.layers.dense({
        units: 1,
        activation: "sigmoid"
    });

    // add layers to the model
    model.add(hiddenLayer);
    model.add(outputlayer);

    model.compile({
        loss: "binaryCrossentropy",
        optimizer: tf.train.adam(0.01),
        metrics:['accuracy']
    });

    model.summary();

    const config = {
        epochs: 50,
        stepsPerEpoch:null,
        shuffle: true,
        useBias: true,
        biasInitializer: 'randomNormal',
        batchSize: 32,
        callbacks:{
            onEpochEnd(num,logs){
                console.log(`Epoch : ${num}, Loss : ${logs.loss}, Accuracy : ${logs.acc}`);
            }
        }
    };

    // train the model
    const history = await model.fit(xs_train, ys_train, config);

    // make prediction
    const y_pred = model.predict(xs_test);

    // check test accuracy
    let accuracy = Utils.accuracyForBinaryResult(ys_test, y_pred);

    accuracy = await accuracy.data();
    console.log("Test Accuracy :", accuracy[0]);

};


main().then(() => {
    console.log("Training completed!");

}).catch((err) => {
    console.log(err);
})

