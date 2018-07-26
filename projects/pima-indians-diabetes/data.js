const CSV = require("fast-csv");
const {DataFrame}  = require('dataframe-js');


const loadDataFromFile = async (path) => {

    return new Promise((resolve,reject)=>{

        const dataSet=[];

        try {

            CSV.fromPath(path,{headers: false})
            .on("data", function (row) {
                dataSet.push(row)
            })
            .on("end", function () {
                resolve(dataSet)
            });
            
        } catch (error) {
            reject(error)
        }

    });

};

const shuffle=(table,colNames)=>{
    const df=new DataFrame(table,colNames);
    return df.shuffle().toArray();
};

const separateLabelsAndFeatures = (dTable,isShuffle=true,colNames=[]) => {

    let table=dTable;

    const dataSet = {
        features:[],
        labels:[]
    };

    if(isShuffle){
        table=shuffle(table,colNames)
    }


    table.forEach(row => {
        let fe=row.slice(0,row.length-1);
        let lab=row.slice(-1);
        dataSet.features.push(fe);
        dataSet.labels.push(lab); 
    });

    
    return dataSet;

};


// Min-Max Normalization (Min-Max Scaling):
const normalizeFeatures = (xs) => {

    const max = xs.max();
    const min = xs.min();
    let norm = xs.sub(min).div(max.sub(min));
    max.print()
    return norm;

}


module.exports={
    loadDataFromFile,
    shuffle,
    separateLabelsAndFeatures
};