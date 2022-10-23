import com.google.common.collect.Lists;


import java.io.DataInputStream;

import java.io.IOException;
import java.io.InputStream;

import java.util.List;
import java.util.Random;
import java.util.zip.GZIPInputStream;

public class KataMnist {

    public static final String TRAIN_IMAGE_FILE = "data/train-images-idx3-ubyte.gz";
    public static final String TRAIN_LABEL_FILE = "data/train-labels-idx1-ubyte.gz";


    public static final String TEST_IMAGE_FILE = "data/t10k-images-idx3-ubyte.gz";
    public static final String TEST_LABEL_FILE = "data/t10k-labels-idx1-ubyte.gz";


    private final int outputLayerSize = 10;
    private final int perTrainDataSize = 28 * 28;

    private final double learningRate;
    private final List<byte[]> trainDataList = Lists.newArrayList();
    private final List<Integer> trainResultList = Lists.newArrayList();

    private final List<byte[]> testDataList = Lists.newArrayList();
    private final List<Integer> testResultList = Lists.newArrayList();

    private final List<double[][]> weights = Lists.newArrayList();

    private final int[] layerUnitNums;

    public KataMnist(int[] hiddenLayerUnitNums, double learningRate) throws IOException {

        this.initTrainData(TRAIN_IMAGE_FILE, TRAIN_LABEL_FILE, trainDataList, trainResultList);
        this.initTrainData(TEST_IMAGE_FILE, TEST_LABEL_FILE, testDataList, testResultList);
        layerUnitNums = new int[hiddenLayerUnitNums.length + 2];
        layerUnitNums[0] = 28 * 28;
        for (int i = 0; i < hiddenLayerUnitNums.length; ++i) {
            layerUnitNums[i + 1] = hiddenLayerUnitNums[i];
        }
        layerUnitNums[hiddenLayerUnitNums.length + 1] = 10;
        this.learningRate = learningRate;
        initWeights();
    }


    public int feedForward(byte[] x) {
        double[] lastLayerActivation = new double[x.length + 1];
        lastLayerActivation[0] = 1;
        for (int i = 1; i < x.length; i++) {
            lastLayerActivation[i] = x[i] & 0xff;
        }

        for (int layerIndex = 1; layerIndex < layerUnitNums.length; ++layerIndex) {
            int layerUnitNum = layerUnitNums[layerIndex];
            int previewLayerUnitNum = layerUnitNums[layerIndex - 1];
            double[][] layerWeight = weights.get(layerIndex);
            double[] layerActivation = new double[layerUnitNum + 1];
            layerActivation[0] = 1;

            for (int j = 1; j <= layerUnitNum; ++j) {
                double cellOutput = 0;
                for (int i = 0; i <= previewLayerUnitNum; ++i) {
                    cellOutput += layerWeight[j][i] * lastLayerActivation[i];
                }
                cellOutput = sigmoid(cellOutput);
                layerActivation[j] = cellOutput;
            }
            lastLayerActivation = layerActivation;
        }

        int activationUnitIndex = -1;
        double maxActivation = -1;
        for (int i = 1; i < lastLayerActivation.length; ++i) {
            if (lastLayerActivation[i] > maxActivation) {
                activationUnitIndex = i;
                maxActivation = lastLayerActivation[i];
            }
        }
        return activationUnitIndex - 1;
    }

    public void gradientDescent(int miniBatchSize) {

        for (int trainDataIndex = 0; trainDataIndex < trainDataList.size(); trainDataIndex += miniBatchSize) {

            List<List<double[][]>> deltaWeightsList = Lists.newArrayList();
            for (int batchIndex = 0; batchIndex < miniBatchSize; ++batchIndex) {
                byte[] trainData = trainDataList.get(trainDataIndex + batchIndex);
                Integer trainResult = trainResultList.get(trainDataIndex + batchIndex);
                deltaWeightsList.add(backForward(trainData, trainResult));
            }

            for (int batchIndex = 0; batchIndex < miniBatchSize; ++batchIndex) {
                List<double[][]> deltaWeights = deltaWeightsList.get(batchIndex);
                for (int layerIndex = layerUnitNums.length - 1; layerIndex >= 1; --layerIndex) {
                    int layerUnitNum = layerUnitNums[layerIndex];
                    int previousLayerUnitNum = layerUnitNums[layerIndex - 1];
                    double[][] layerWeight = weights.get(layerIndex);
                    double[][] layerDeltaWeight = deltaWeights.get(layerIndex);
                    for (int j = 1; j <= layerUnitNum; ++j) {
                        for (int i = 1; i <= previousLayerUnitNum; ++i) {
                            layerWeight[j][i] += layerDeltaWeight[j][i] / miniBatchSize;
                        }
                    }
                }
            }
        }
    }


    private List<double[][]> backForward(byte[] trainData, Integer trainResult) {

        List<double[]> activations = Lists.newArrayList();
        double[] inputLayerActivation = new double[trainData.length + 1];
        inputLayerActivation[0] = 1;
        for (int i = 0; i < trainData.length; i++) {
            inputLayerActivation[i + 1] = trainData[i] & 0xff;
        }
//        System.out.println(Arrays.toString(inputLayerActivation));
//        System.exit(0);
        activations.add(inputLayerActivation);
        for (int layerIndex = 1; layerIndex < layerUnitNums.length; ++layerIndex) {
            int layerUnitNum = layerUnitNums[layerIndex];
            int previousLayerUnitNum = layerUnitNums[layerIndex - 1];
            double[][] layerWeight = weights.get(layerIndex);
            double[] lastLayerActivation = activations.get(layerIndex - 1);
            double[] layerActivation = new double[layerUnitNum + 1];
            layerActivation[0] = 1;
            for (int j = 1; j <= layerUnitNum; ++j) {
                double netj = 0;
                for (int i = 0; i <= previousLayerUnitNum; ++i) {
                    netj += layerWeight[j][i] * lastLayerActivation[i];
                }
                layerActivation[j] = sigmoid(netj);
            }
            activations.add(layerActivation);
        }

        List<double[]> layerErrorRatesList = Lists.newArrayList();
        for (int i = 0; i < layerUnitNums.length; ++i) {
            layerErrorRatesList.add(new double[0]);
        }

        // 计算输出层错误率
        int outputLayerUnitNums = layerUnitNums[layerUnitNums.length - 1];
        double[] outputLayerErrorRates = new double[outputLayerUnitNums + 1];
        double[] outputLayerActivation = activations.get(layerUnitNums.length - 1);
        for (int i = 1; i <= outputLayerUnitNums; ++i) {
            double targetResult = 0;
            if (i - 1 == trainResult) {
                targetResult = 1;
            }
            double cellActivation = outputLayerActivation[i];
            outputLayerErrorRates[i] = cellActivation * (1 - cellActivation) * (targetResult - cellActivation);
        }
        layerErrorRatesList.set(layerUnitNums.length - 1, outputLayerErrorRates);

        // 计算隐藏层错误率
        for (int layerIndex = layerUnitNums.length - 2; layerIndex >= 1; --layerIndex) {
            int layerUnitNum = layerUnitNums[layerIndex];
            int nextLayerUnitNum = layerUnitNums[layerIndex + 1];

            double[][] layerWeight = weights.get(layerIndex);
            double[] layerActivations = activations.get(layerIndex);

            double[] nextLayerErrorRate = layerErrorRatesList.get(layerIndex + 1);

            double[] layerErrorRate = new double[layerUnitNum + 1];
            for (int j = 1; j <= layerUnitNum; ++j) {
                double outputErrorRate = 0;
                for (int k = 1; k <= nextLayerUnitNum; ++k) {
                    outputErrorRate += layerWeight[k][j] * nextLayerErrorRate[k];
                }
                layerErrorRate[j] = layerActivations[j] * (1 - layerActivations[j]) * outputErrorRate;
            }
            layerErrorRatesList.set(layerIndex, layerErrorRate);
        }


        List<double[][]> deltaWeightsList = Lists.newArrayList();
        for (int i = 0; i < layerUnitNums.length; ++i) {
            deltaWeightsList.add(null);
        }

        // 计算变化的权
        for (int layerIndex = layerUnitNums.length - 1; layerIndex >= 1; --layerIndex) {
            double[] layerErrorRate = layerErrorRatesList.get(layerIndex);
            double[] previousLayerActivations = activations.get(layerIndex - 1);

            int layerUnitNum = layerUnitNums[layerIndex];
            int previousLayerUnitNum = layerUnitNums[layerIndex - 1];

            double[][] deltaWeights = new double[layerUnitNum + 1][previousLayerUnitNum + 1];

            for (int j = 1; j <= layerUnitNum; ++j) {
                for (int i = 1; i <= previousLayerUnitNum; ++i) {
                    deltaWeights[j][i] = learningRate * layerErrorRate[j] * previousLayerActivations[i];
                }
            }
            deltaWeightsList.set(layerIndex, deltaWeights);
        }

        return deltaWeightsList;
    }


    public double runTest() {
        int correctResultNum = 0;
        for (int i = 0; i < testDataList.size(); ++i) {
            int actualResult = feedForward(testDataList.get(i));
            int testResult = testResultList.get(i);
            if (actualResult == testResult) {
                correctResultNum++;
            }
        }
        return 1.0 * correctResultNum / testResultList.size();
    }

    public double runTrainData() {
        int correctResultNum = 0;
        for (int i = 0; i < trainDataList.size(); ++i) {
            int actualResult = feedForward(trainDataList.get(i));
            int testResult = trainResultList.get(i);
            if (actualResult == testResult) {
                correctResultNum++;
            }
        }
        return 1.0 * correctResultNum / trainDataList.size();
    }


    private void initWeights() {


        Random random = new Random();
        // input层没有weight
        weights.add(new double[0][0]);
        // 初始化weight
        for (int layerIndex = 1; layerIndex < layerUnitNums.length; ++layerIndex) {
            int previousLayerCellNum = layerUnitNums[layerIndex - 1];
            int layerCellNum = layerUnitNums[layerIndex];
            double[][] layerWeights = new double[layerCellNum + 1][previousLayerCellNum + 1];
            for (int j = 0; j < layerCellNum + 1; ++j) {
                for (int i = 0; i < previousLayerCellNum + 1; ++i) {
                    layerWeights[j][i] = random.nextGaussian();
//                                        layerWeights[j][i] = Math.random() - 0.5;

                }
            }
            weights.add(layerWeights);
        }
    }


    public double sigmoid(double x) {
        return 1.0 / (1 + Math.exp(-x));
    }


    public double derivativeSigmoid(double x) {
        return sigmoid(x) * (1 - sigmoid(x));
    }

    public void initTrainData(String inputFileName, String resultFileName, List<byte[]> inputDataList, List<Integer> resultList) throws IOException {

        InputStream in = KataMnist.class.getClassLoader().getResourceAsStream(inputFileName);
        DataInputStream dsIn = new DataInputStream(new GZIPInputStream(in));
        InputStream resultIn = KataMnist.class.getClassLoader().getResourceAsStream(resultFileName);
        DataInputStream dsResultIn = new DataInputStream(new GZIPInputStream(resultIn));
        int dataMagic = dsIn.readInt();
        int dataNums = dsIn.readInt();
        int trainDataDimX = dsIn.readInt();
        int trainDataDimY = dsIn.readInt();

        int trainResultMagic = dsResultIn.readInt();
        int trainResultNums = dsResultIn.readInt();

        for (int k = 0; k < dataNums; ++k) {
            byte[] buf = new byte[perTrainDataSize];
            dsIn.readFully(buf);
            int result = dsResultIn.readUnsignedByte();
            inputDataList.add(buf);
            resultList.add(result);
        }
        dsIn.close();
        dsResultIn.close();
    }

    public static void main(String[] args) throws IOException {
        KataMnist kataMnist = new KataMnist(new int[]{30}, 0.3);
        System.out.println("init  " + kataMnist.runTest());
        kataMnist.runTest();
        for (int i = 0; i < 100; ++i) {
            kataMnist.gradientDescent(10);
            System.out.println(i + "trainData correct percent: " + kataMnist.runTrainData());
            System.out.println(i + "testData correct percent " + kataMnist.runTest());
        }
    }

    /**
     * 70% 左右的准确率
     *
     * init  0.1004
     * 0trainData correct percent: 0.3172
     * 0testData correct percent 0.3178
     * 1trainData correct percent: 0.38666666666666666
     * 1testData correct percent 0.3787
     * 2trainData correct percent: 0.4240333333333333
     * 2testData correct percent 0.4204
     * 3trainData correct percent: 0.4641166666666667
     * 3testData correct percent 0.4628
     * 4trainData correct percent: 0.5348333333333334
     * 4testData correct percent 0.5292
     * 5trainData correct percent: 0.52225
     * 5testData correct percent 0.5267
     * 6trainData correct percent: 0.54815
     * 6testData correct percent 0.547
     * 7trainData correct percent: 0.5506333333333333
     * 7testData correct percent 0.5499
     * 8trainData correct percent: 0.56755
     * 8testData correct percent 0.5611
     * 9trainData correct percent: 0.6266833333333334
     * 9testData correct percent 0.6314
     * 10trainData correct percent: 0.5901666666666666
     * 10testData correct percent 0.587
     * 11trainData correct percent: 0.6365333333333333
     * 11testData correct percent 0.6331
     * 12trainData correct percent: 0.64355
     * 12testData correct percent 0.6366
     * 13trainData correct percent: 0.5942166666666666
     * 13testData correct percent 0.594
     * 14trainData correct percent: 0.6270166666666667
     * 14testData correct percent 0.6204
     * 15trainData correct percent: 0.6637666666666666
     * 15testData correct percent 0.6617
     * 16trainData correct percent: 0.6450166666666667
     * 16testData correct percent 0.6384
     * 17trainData correct percent: 0.66305
     * 17testData correct percent 0.6589
     * 18trainData correct percent: 0.6847666666666666
     * 18testData correct percent 0.6843
     * 19trainData correct percent: 0.6433166666666666
     * 19testData correct percent 0.6478
     * 20trainData correct percent: 0.693
     * 20testData correct percent 0.697
     * 21trainData correct percent: 0.6854333333333333
     * 21testData correct percent 0.6969
     * 22trainData correct percent: 0.68985
     * 22testData correct percent 0.694
     * 23trainData correct percent: 0.6011333333333333
     * 23testData correct percent 0.5989
     * 24trainData correct percent: 0.6254833333333333
     * 24testData correct percent 0.6269
     * 25trainData correct percent: 0.6348666666666667
     * 25testData correct percent 0.6279
     * 26trainData correct percent: 0.6988
     * 26testData correct percent 0.7072
     * 27trainData correct percent: 0.69905
     * 27testData correct percent 0.7029
     * 28trainData correct percent: 0.7013833333333334
     * 28testData correct percent 0.6926
     * 29trainData correct percent: 0.6619833333333334
     * 29testData correct percent 0.66
     * 30trainData correct percent: 0.6804166666666667
     * 30testData correct percent 0.6771
     * 31trainData correct percent: 0.6894166666666667
     * 31testData correct percent 0.6854
     * 32trainData correct percent: 0.6670166666666667
     * 32testData correct percent 0.6669
     * 33trainData correct percent: 0.7251666666666666
     * 33testData correct percent 0.7191
     * 34trainData correct percent: 0.73115
     * 34testData correct percent 0.7236
     * 35trainData correct percent: 0.7140166666666666
     * 35testData correct percent 0.7128
     * 36trainData correct percent: 0.69195
     * 36testData correct percent 0.6928
     * 37trainData correct percent: 0.6827666666666666
     * 37testData correct percent 0.6794
     * 38trainData correct percent: 0.66055
     * 38testData correct percent 0.6594
     * 39trainData correct percent: 0.6750166666666667
     * 39testData correct percent 0.6723
     * 40trainData correct percent: 0.7024833333333333
     * 40testData correct percent 0.7026
     * 41trainData correct percent: 0.711
     * 41testData correct percent 0.7099
     * 42trainData correct percent: 0.7033
     * 42testData correct percent 0.7013
     * 43trainData correct percent: 0.7053166666666667
     * 43testData correct percent 0.7014
     * 44trainData correct percent: 0.6974333333333333
     * 44testData correct percent 0.6885
     * 45trainData correct percent: 0.6548166666666667
     * 45testData correct percent 0.6545
     * 46trainData correct percent: 0.6897
     * 46testData correct percent 0.6858
     * 47trainData correct percent: 0.6658
     * 47testData correct percent 0.6585
     * 48trainData correct percent: 0.6926
     * 48testData correct percent 0.685
     * 49trainData correct percent: 0.7068833333333333
     * 49testData correct percent 0.6986
     * 50trainData correct percent: 0.6659166666666667
     * 50testData correct percent 0.6599
     * 51trainData correct percent: 0.6720333333333334
     * 51testData correct percent 0.6647
     * 52trainData correct percent: 0.6874333333333333
     * 52testData correct percent 0.681
     * 53trainData correct percent: 0.6556333333333333
     * 53testData correct percent 0.6436
     * 54trainData correct percent: 0.6477
     * 54testData correct percent 0.6394
     * 55trainData correct percent: 0.6730666666666667
     * 55testData correct percent 0.6699
     * 56trainData correct percent: 0.6843333333333333
     * 56testData correct percent 0.6799
     * 57trainData correct percent: 0.6598
     * 57testData correct percent 0.6584
     * 58trainData correct percent: 0.6893
     * 58testData correct percent 0.6869
     * 59trainData correct percent: 0.6804333333333333
     * 59testData correct percent 0.6758
     * 60trainData correct percent: 0.676
     * 60testData correct percent 0.6755
     * 61trainData correct percent: 0.68085
     * 61testData correct percent 0.6774
     * 62trainData correct percent: 0.6904833333333333
     * 62testData correct percent 0.693
     * 63trainData correct percent: 0.7052333333333334
     * 63testData correct percent 0.7073
     * 64trainData correct percent: 0.7011
     * 64testData correct percent 0.6984
     * 65trainData correct percent: 0.7238166666666667
     * 65testData correct percent 0.7267
     * 66trainData correct percent: 0.7122666666666667
     * 66testData correct percent 0.7122
     * 67trainData correct percent: 0.7035833333333333
     * 67testData correct percent 0.7028
     * 68trainData correct percent: 0.7092166666666667
     * 68testData correct percent 0.7074
     * 69trainData correct percent: 0.6883166666666667
     * 69testData correct percent 0.6866
     * 70trainData correct percent: 0.7049833333333333
     * 70testData correct percent 0.7015
     * 71trainData correct percent: 0.7085666666666667
     * 71testData correct percent 0.7052
     * 72trainData correct percent: 0.6666666666666666
     * 72testData correct percent 0.664
     * 73trainData correct percent: 0.6936666666666667
     * 73testData correct percent 0.6921
     * 74trainData correct percent: 0.6900666666666667
     * 74testData correct percent 0.6858
     * 75trainData correct percent: 0.6914166666666667
     * 75testData correct percent 0.6894
     * 76trainData correct percent: 0.6854833333333333
     * 76testData correct percent 0.6883
     * 77trainData correct percent: 0.6939666666666666
     * 77testData correct percent 0.6964
     * 78trainData correct percent: 0.70715
     * 78testData correct percent 0.7068
     * 79trainData correct percent: 0.6994166666666667
     * 79testData correct percent 0.6969
     * 80trainData correct percent: 0.6508166666666667
     * 80testData correct percent 0.6511
     * 81trainData correct percent: 0.6972166666666667
     * 81testData correct percent 0.6934
     * 82trainData correct percent: 0.6951166666666667
     * 82testData correct percent 0.6917
     * 83trainData correct percent: 0.7005166666666667
     * 83testData correct percent 0.6941
     * 84trainData correct percent: 0.7038833333333333
     * 84testData correct percent 0.6979
     * 85trainData correct percent: 0.69555
     * 85testData correct percent 0.6893
     * 86trainData correct percent: 0.6997333333333333
     * 86testData correct percent 0.6922
     * 87trainData correct percent: 0.7047166666666667
     * 87testData correct percent 0.6988
     * 88trainData correct percent: 0.7023
     * 88testData correct percent 0.7043
     * 89trainData correct percent: 0.70795
     * 89testData correct percent 0.7093
     * 90trainData correct percent: 0.7124666666666667
     * 90testData correct percent 0.7154
     * 91trainData correct percent: 0.7332
     * 91testData correct percent 0.7359
     * 92trainData correct percent: 0.7160166666666666
     * 92testData correct percent 0.7125
     * 93trainData correct percent: 0.7128833333333333
     * 93testData correct percent 0.7079
     * 94trainData correct percent: 0.7157833333333333
     * 94testData correct percent 0.7124
     * 95trainData correct percent: 0.7265333333333334
     * 95testData correct percent 0.7191
     * 96trainData correct percent: 0.7353
     * 96testData correct percent 0.7333
     * 97trainData correct percent: 0.6996166666666667
     * 97testData correct percent 0.6917
     * 98trainData correct percent: 0.69065
     * 98testData correct percent 0.6822
     * 99trainData correct percent: 0.7137166666666667
     * 99testData correct percent 0.7051
     */
}