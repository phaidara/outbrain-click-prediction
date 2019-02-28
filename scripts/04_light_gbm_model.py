from itertools import chain
from pyspark.sql.types import *
from pyspark.sql import functions as f
from pyspark.sql.window import Window
from mmlspark import LightGBMClassifier

DIR_DATA_PARQUET = "./data/parquet/"
DIR_DATA_CSV = "./data/csv/"

if __name__ == "__main__":

    spark = SparkSession \
        .builder \
        .appName('LightGBM') \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    # Reading train_test dataset with features vector column
    train_test = spark.read.parquet(DIR_DATA_PARQUET + 'train_test_final')

    print("Fitting model on train sample")
    train_model = train_test.filter("stage = 2")
    lgbm = LightGBMClassifier(labelCol='clicked', objective="binary")
    lgbm_model = lgbm.fit(train_model)

    # printing feature importance
    print("Feature importance")
    feature_imp = lgbm_model.getFeatureImportances()
    feature_imp_order = list(reversed(sorted(range(len(feature_imp)), key=lambda k: feature_imp[k])))
    feature_names = sorted((attr["idx"], attr["name"]) for attr in
                           (chain(*train_model.schema["features"].metadata["ml_attr"]["attrs"].values())))

    for t in [(feature_names[i], feature_imp[i]) for i in feature_imp_order if feature_imp[i] > 0.0]:
        print(t[0], ":", str(t[1]))

    # Computing map_k on the validation data set
    print("Computing map_k on validation")

    # User defined function to extract the probability of click from the model probability array
    extract_proba_udf = f.udf(lambda x: float(x[1]), DoubleType())

    # Map k function
    def map_k(df, k, grp_col="display_id", proba_col="probability", label_col="clicked"):
        """
        Compute the map_k on a validation dataset
        :param df: DataFrame with grouping column, probability array, and label column
        :param k: k
        :param grp_col: grouping column
        :param proba_col: Array containing the probabilities
        :param label_col: label column
        :return: map k computed from the input DataFrame
        """
        df_new = df.withColumn(proba_col, extract_proba_udf(proba_col))
        window = Window.partitionBy(df_new[grp_col]).orderBy(df_new[proba_col].desc())
        window_full = window.rangeBetween(Window.unboundedPreceding, Window.unboundedFollowing)
        # Computing rank and precision per group
        df_new = df_new.withColumn("rank", f.when(f.first(proba_col).over(window_full) ==
                                                  f.last(proba_col).over(window_full),  # if all probabilities have
                                                  #  the same value, we take the average rank (number of ads / 2)
                                                  f.count(grp_col).over(window_full)/2)
                                            .otherwise(f.rank().over(window)))\
                       .withColumn("precision",
                                   f.when((f.col(label_col) == 1) & (f.col("rank") <= k),
                                          1/f.col("rank"))
                                    .otherwise(0.0))
        mapk = df_new.filter(f.col(label_col) == 1).select(f.avg("precision").alias("precision")).collect()
        return mapk[0]['precision']

    train_val = train_test.filter("stage = 4")
    train_val_transform = lgbm_model.transform(train_val)
    train_val_transform.show()
    print("Validation Map_k ", map_k(train_val_transform, 12))

    # Kaggle submission
    print("Computing Kaggle submission csv file")

    def gen_submission(df, model, path):
        """
        Writes a csv files with kaggle submission format
        :param df: Kaggle test dataset with display_id, ad_id and features columns
        :param model: Spark model to compute probabilities with
        :param path: Output path for csv file
        """
        submission = model.transform(df)
        submission = submission.select("display_id", "ad_id", extract_proba_udf("probability").alias("probability"))
        w = Window.partitionBy('display_id').orderBy(f.desc('probability'))
        submission = submission.withColumn('ad_id', f.collect_list('ad_id').over(w)) \
            .groupBy('display_id') \
            .agg(f.max('ad_id').alias('ad_id'))
        submission = submission.withColumn("ad_id", f.concat_ws(" ", "ad_id"))
        submission.select("display_id", "ad_id").repartition(1).write.mode("overwrite").csv(path, header=True)

    submission_df = train_test.filter("stage=0")
    gen_submission(submission_df, lgbm_model, DIR_DATA_CSV + "submission_lgbm")
