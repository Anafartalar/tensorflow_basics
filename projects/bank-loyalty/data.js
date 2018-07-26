const { DataFrame } = require('dataframe-js');



const loadFromCSV = async (path) => {
    let df = await DataFrame.fromCSV(path);
    return df.select("CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
        "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited").shuffle();
};

const saveToCSV = async (df, path) => {
    df.toCSV(true, path);
};


module.exports = {
    loadFromCSV,
    saveToCSV
};