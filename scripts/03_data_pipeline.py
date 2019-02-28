from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, ChiSqSelector, Imputer
from pyspark.ml import Pipeline

DIR_DATA_PARQUET = "./data/parquet/"

if __name__ == "__main__":

    spark = SparkSession \
        .builder \
        .appName('csvToParquetApp') \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    train_test = spark.read.parquet(DIR_DATA_PARQUET + 'train_test_v5')
    train = train_test.filter("train = 1")

    # Splitting display ids in 4 groups (pipeline building, train sample, train and validation)
    print("Splitting displays id")
    display_ids = train.select("display_id").distinct().randomSplit([0.025, 0.025, 0.65, 0.3])
    display_id_sel = display_ids[0].withColumn("stage", f.lit(1))
    display_id_train_1 = display_ids[1].withColumn("stage", f.lit(2))
    display_id_train_2 = display_ids[2].withColumn("stage", f.lit(3))
    display_id_val = display_ids[3].withColumn("stage", f.lit(4))
    display_id_all = display_id_sel.union(display_id_train_1).union(display_id_train_2).union(display_id_val)

    train_test = train_test.join(display_id_all, "display_id", "left_outer").fillna(0, "stage")

    # Filling missing values
    print("Filling missing values")
    fill_na_0 = ['click_ratio_ad_id', 'click_ratio_doc_id_ad', 'click_ratio_cpg_id', 'click_ratio_adv_id',
                 'click_ratio_src_id_ad', 'click_ratio_pub_id_ad', 'click_ratio_doc_id_ad_event',
                 'click_ratio_doc_id_ad_cty', 'click_ratio_doc_id_ad_day', 'click_ratio_doc_id_ad_moment',
                 'click_ratio_doc_id_ad_platform', 'click_ratio_pub_id_ev_doc_id_ad', 'gap_start_session_minutes',
                 'nb_pages_session', 'max_interaction_cat_ad_ev', 'count_interaction_cat_ad_ev',
                 'cosine_distance_cat_ad_ev', 'max_interaction_top_ad_ev', 'count_interaction_top_ad_ev',
                 'cosine_distance_top_ad_ev', 'max_interaction_cat_ad_hist', 'count_interaction_cat_ad_hist',
                 'cosine_distance_cat_ad_hist', 'max_interaction_top_ad_hist', 'count_interaction_top_ad_hist',
                 'cosine_distance_top_ad_hist']

    train_test = train_test.fillna(0, fill_na_0)
    train_test = train_test.fillna(False, "is_weekend")

    # Building pipeline
    print("Fitting pipeline")
    train_select = train_test.filter("stage = 1")
    # One hot encoding
    cols_dummies = ['platform', 'moment_of_day', 'day_of_week', 'country_region', 'hour_of_day', 'traffic_source']
    stages_si = [StringIndexer(inputCol=c, outputCol=c + "_index", handleInvalid="keep") for c in cols_dummies]
    stages_ohe = [OneHotEncoder(inputCol=c + '_index', outputCol=c + '_dummies') for c in cols_dummies]
    # Chi square selection
    cols_select = ['country_region_dummies']
    stages_chiSq = [ChiSqSelector(featuresCol=c, outputCol=c + "_select", labelCol="clicked",
                                  selectorType="fpr", fpr=0.5) for c in cols_select]
    # Imputer
    cols_imp = ['doc_event_published_since', 'doc_ad_published_since', 'gap_previous_timestamp_hours',
                'time_diff_pub_ad_doc']
    imputer = Imputer(inputCols=cols_imp, outputCols=[c + "_imp" for c in cols_imp]).setStrategy("median")
    # Fitting pipeline
    pipeline = Pipeline(stages=stages_si + stages_ohe + stages_chiSq + [imputer])
    pipeline_model = pipeline.fit(train_select)

    # Assembling features on the whole train_test dataset
    train_test_pipeline = pipeline_model.transform(train_test)

    # Columns to assemble into the feature vector
    train_col = ['is_weekend', 'click_ratio_doc_id_ad', 'click_ratio_cpg_id',
                 'click_ratio_adv_id', 'click_ratio_src_id_ad', 'click_ratio_pub_id_ad', 'click_ratio_doc_id_ad_event',
                 'click_ratio_doc_id_ad_cty', 'click_ratio_doc_id_ad_day', 'click_ratio_doc_id_ad_moment',
                 'click_ratio_pub_id_ev_doc_id_ad', 'click_ratio_doc_id_ad_platform', 'gap_start_session_minutes',
                 'nb_pages_session', 'max_interaction_cat_ad_ev', 'count_interaction_cat_ad_ev',
                 'cosine_distance_cat_ad_ev', 'max_interaction_top_ad_ev', 'count_interaction_top_ad_ev',
                 'cosine_distance_top_ad_ev', 'max_interaction_cat_ad_hist', 'count_interaction_cat_ad_hist',
                 'cosine_distance_cat_ad_hist', 'max_interaction_top_ad_hist', 'count_interaction_top_ad_hist',
                 'cosine_distance_top_ad_hist', 'count_ads_in_display', 'rank_click_ratio_doc_id_ad',
                 'rank_click_ratio_cpg_id', 'rank_click_ratio_adv_id', 'rank_click_ratio_src_id_ad',
                 'rank_click_ratio_pub_id_ad', 'rank_click_ratio_doc_id_ad_event', 'rank_click_ratio_doc_id_ad_cty',
                 'rank_click_ratio_doc_id_ad_day', 'rank_click_ratio_doc_id_ad_moment',
                 'rank_click_ratio_doc_id_ad_platform', 'rank_click_ratio_pub_id_ev_doc_id_ad',
                 'rank_cosine_cat_ad_event', 'rank_cosine_top_ad_event', 'platform_dummies', 'moment_of_day_dummies',
                 'day_of_week_dummies', 'hour_of_day_dummies', 'traffic_source_dummies',
                 'country_region_dummies_select', 'doc_event_published_since_imp', 'doc_ad_published_since_imp',
                 'gap_previous_timestamp_hours_imp', 'gap_previous_timestamp_hours_imp', 'time_diff_pub_ad_doc_imp']

    assembler_train = VectorAssembler(inputCols=train_col, outputCol="features")
    train_test_final = assembler_train.transform(train_test_pipeline).select('display_id', 'ad_id', "stage",
                                                                             'clicked', 'features')
    print("Writing final train_test")
    train_test_final.write.mode("overwrite").partitionBy("stage").parquet(DIR_DATA_PARQUET + "train_test_final")
