import * as Tensors from './prepare.js';
import * as Kernel from './kernel.js'


let xTrain, yTrain, xTest, yTest;


[xTrain, yTrain, xTest, yTest] = Tensors.getWineData(0.2);


let model = Kernel.makeModel();

let wines = [
    tf.tensor([[13.24, 2.59, 2.87, 21, 118, 2.8, 2.69, .39, 1.82, 4.32, 1.04, 2.93, 7.35]]),
    tf.tensor([[14.2, 1.76, 2.45, 15.2, 112, 3.27, 3.39, .34, 1.97, 6.75, 1.05, 2.85, 14.50]]),
    tf.tensor([[14.39, 1.87, 2.45, 14.6, 96, 2.5, 2.52, .3, 1.98, 5.25, 1.02, 3.58, 12.90]]),
    tf.tensor([[12.08, 2.08, 1.7, 17.5, 97, 2.23, 2.17, .26, 1.4, 3.3, 1.27, 2.96, 7.10]]),
    tf.tensor([[12.51, 1.73, 1.98, 20.5, 85, 2.2, 1.92, .32, 1.48, 2.94, 1.04, 3.57, 6.72]]),
    tf.tensor([[13.05, 5.8, 2.13, 21.5, 86, 2.62, 2.65, .3, 2.01, 2.6, .73, 3.1, 3.80]]),
    tf.tensor([[12.58, 1.29, 2.1, 20, 103, 1.48, .58, .53, 1.4, 7.6, .58, 1.55, 6.40]]),
    tf.tensor([[13.73, 4.36, 2.26, 22.5, 88, 1.28, .47, .52, 1.15, 6.62, .78, 1.75, 5.20]]),
    tf.tensor([[13.27, 4.28, 2.26, 20, 120, 1.59, .69, .43, 1.35, 10.2, .59, 1.56, 8.35]])
];


Kernel.trainModel(model, xTrain, yTrain, xTest, yTest).then(model => {
    wines.forEach(wine => {
        model.predict(wine).print();
    })
});
