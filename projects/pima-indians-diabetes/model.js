const Data = require("./data");
const Path = require("path");
const tf = require('@tensorflow/tfjs');

require('@tensorflow/tfjs-node');


const accuracyForBinary=(y_true,y_pred)=>{

    const a=tf.mean(tf.equal(y_true,tf.round(y_pred)));
    return a;

};

//K.mean(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))
const accuracyForcategorical=()=>{

}


const main = async () => {

    const colNames=[
        "number_of_times_pregnant",
        "plasma_glucose_concentration",
        "diastolic_blood_pressure",
        "triceps_skin_fold_thickness",
        "2_hour_serum_insulin",
        "body_mass_index",
        "diabetes_pedigree_function",
        "age",
        "class"
    ];

    let dataFilePath = Path.join(__dirname, "dataset", "pima-indians-diabetes_No.csv");
    let dataSet = await Data.loadDataFromFile(dataFilePath);

    // get features and labels shuffling dataset
    const labelsAndFeatures=Data.separateLabelsAndFeatures(dataSet,true,colNames);
 
    // get features and labels
    let features = tf.tensor2d(labelsAndFeatures.features);
    let labels = tf.tensor2d(labelsAndFeatures.labels);

    // prepare traning and test data, 80% traning
    let [xs_train, xs_test] = tf.split(features, [648, 120])
    let [ys_train, ys_test] = tf.split(labels, [648, 120])

    features.dispose();
    labels.dispose();


    const model = tf.sequential();

    const hiddenLayer1 = tf.layers.dense({
        units: 5,
        inputShape: [8],
        activation: "relu"
    })

    const outputLayer = tf.layers.dense({
        units: 1,
        activation: "sigmoid"
    });

    model.add(hiddenLayer1)
    model.add(outputLayer)

   
    model.compile({
        loss: tf.losses.meanSquaredError,
        optimizer: tf.train.sgd(0.2)
    });

    model.summary();

    for (let i = 1; i <= 100; i++) {

        const history = await model.fit(xs_train, ys_train, {
            epochs: 2,
            shuffle:true,
            stepsPerEpoch:null,
            kernelInitializer: 'leCunNormal',
            useBias: true,
            biasInitializer: 'randomNormal',
            batchSize:10
        });

        console.log("Loss after Epoch " + i + " : "+history.history.loss[0]);
    }


    const y_pred=model.predict(xs_test);

    let accuracy=accuracyForBinary(ys_test,y_pred);

    
    accuracy=await accuracy.data();
    console.log("Accuracy :", accuracy[0]);

};


main().then(() => {
    console.log("Training completed");

}).catch((err) => {
    console.log(err);
})
