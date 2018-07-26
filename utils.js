const tf = require('@tensorflow/tfjs');


// Min-Max Normalization (Min-Max Scaling):
const normalizeFeatures = (df, cols) => {

    const minMax = {};
    let dfTemp = df;

    // go over all column names
    cols.forEach(col => {
        let min = 0, max = 0
        min = dfTemp.stat.min(col); // get the min value
        max = dfTemp.stat.max(col); // get the max value

        // apply normalization to every item in the column
        dfTemp = dfTemp.withColumn(col, (row) => {
            let val = row.get(col);
            val = ((val - min) / (max - min)); // normalizing
            return val;
        });

    });;

    return dfTemp;


};

// this function encodes categorical columns using a methot called onehot
const oneHotEncoder = (df, cols) => {

    let dfTemp = df;
    // loop column names
    cols.forEach((col) => {

        let values = dfTemp.distinct(col).toArray(); // extract unique values
        values = values.map((ary) => ary[0]);

        // loop through unique values and assign 1 or 0 
        values.forEach((val) => {
            // create new column with the value of val
            dfTemp = dfTemp.withColumn(val, (row) => {
                let cell = row.get(col);
                return cell == val ? 1 : 0; // check cell's value and assign 1 or 0
            });
        });


        dfTemp=dfTemp.drop(col); //drop the encoded column, there is no need anymore

    });

    return dfTemp;

};


const castValuesToFloat=(df,cols)=>{

    let dfTemp = df;
    cols.forEach((col) => {
        dfTemp = dfTemp.withColumn(col, (row)=>parseFloat(row.get(col)));
    });
    return dfTemp;
};

const accuracyForBinaryResult=(y_true,y_pred)=>{
    const a=tf.mean(tf.equal(y_true,tf.round(y_pred)));
    return a;
};



module.exports = {
    normalizeFeatures,
    oneHotEncoder,
    castValuesToFloat,
    accuracyForBinaryResult
}
