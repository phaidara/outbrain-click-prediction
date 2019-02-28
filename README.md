# Outbrain Click Prediction
This repository contains the work that I have done in the Outbrain Click Prediction Kaggle challenge
(https://www.kaggle.com/c/outbrain-click-prediction).

### Running the scripts
The whole project has been developed in `Python 3.6` using
`Pyspark 2.4.0`. 

To make it work, the original csv files (https://www.kaggle.com/c/outbrain-click-prediction/data) must be put in the `data/csv` directory.

The `scripts directory contains 4 Pyspark applications that do the following:
- `01_csv_to_parquet.py` : reads all the Kaggle csv files and rewrites them in Parquet format
- `02_train_dataset.py` : consolidates the data and builds features for training a model
- `03_data_pipeline.py` : fits a Spark data transformation pipeline and applies it to the train and test datasets
- `04_light_gbm_model.py` : fits a simple Light GBM model over the train dataset and generates a Kaggle submission on the test dataset

To run each one of the 3 first scripts, run on a shell :
```bash
spark-submit [script name] [spark config options]
```
The `04_light_gbm_model.py` needs the MMLSpark package (https://github.com/Azure/mmlspark):
```bash
spark-submit 04_light_gbm_model.py --packages Azure:mmlspark:0.15 [spark config options]
```

### Results
The final submission gives a Kaggle MAP12 score of **0.65530**.

If submitted on time during the competition, it would have ranked **167th** over 979.

### Possible improvements
The Light GBM model has been fitted with default parameters which could be fine-tuned to achieve better results.
It could also be interesting to test Deep learning models as well.

The model has been trained a sampled dataset which accounts for 2,5% of the whole train dataset. 
We could gain more power by training it on more observations.