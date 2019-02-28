from pyspark.sql.types import *
from pyspark.sql import SparkSession, functions as f
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.window import Window

DIR_DATA_CSV = "./data/csv/"
DIR_DATA_PARQUET = "./data/parquet/"
FIRST_TIMESTAMP = 1465876799998  # First recorded timestamp in the DataSet in ms
THRESHOLD_CLICK_RATIO = 25  # If a click ratio is computed with less measures than that threshold, it will be set to 0
MAX_GAP_MINUTES = 120  # For a same user and document id,  a page view row and an event row are considered equal if
# the time difference between the two is less than that value
GAP_SESSION_MINUTES = 30  # Session gap duration (a new user session starts after this value)

if __name__ == "__main__":

    spark = SparkSession \
        .builder \
        .appName("TrainDataset") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    # ############################################# #
    # Concatenation of clicks_train and clicks_test #
    # ############################################# #
    print("Union of clicks_train and clicks_test")
    clicks_train = spark.read.parquet(DIR_DATA_PARQUET + "/clicks_train")
    clicks_test = spark.read.parquet(DIR_DATA_PARQUET + "/clicks_test")
    clicks_train = clicks_train.withColumn("train", f.lit(1))
    clicks_test = clicks_test.withColumn("clicked", f.lit(None)).withColumn("train", f.lit(0))
    clicks_all = clicks_train.union(clicks_test)
    print('clicks_all - number of lines:', "{:,}".format(clicks_all.count()))
    clicks_all.write.mode("overwrite").parquet(DIR_DATA_PARQUET + '/clicks_all')

    # ############################# #
    # Joining events and page views #
    # ############################# #
    print("Joining events and page views")
    events = spark.read.parquet(DIR_DATA_PARQUET + 'events')
    page_views = spark.read.parquet(DIR_DATA_PARQUET + 'page_views')
    
    # Union of  page views and events
    page_views = page_views.withColumn("display_id", f.lit(None)) \
                           .withColumn("page_views", f.lit(1)) \
                           .select("display_id", "uuid", "document_id", "timestamp", "traffic_source", "page_views")
    
    events_pv = events.withColumn("traffic_source", f.lit(None)) \
                      .withColumn("page_views", f.lit(0)) \
                      .select("display_id", "uuid", "document_id", "timestamp", "traffic_source", "page_views")
    
    page_views_events = page_views.union(events_pv.select("display_id", "uuid", "document_id", 
                                                          "timestamp", "traffic_source", "page_views"))
    
    # Matching events and page views by uuid, document id and close enough timestamp (gap < MAX_GAP_MINUTES)
    window_uuid_doc = Window.partitionBy(page_views_events['uuid'], page_views_events["document_id"]) \
                            .orderBy(page_views_events['timestamp'])
    
    page_views_events = page_views_events.withColumn("previous_ts", f.lag("timestamp").over(window_uuid_doc)) \
                                         .withColumn("next_ts", f.lead("timestamp").over(window_uuid_doc))
    
    page_views_events = page_views_events \
                     .withColumn("gap_prev_ts", f.col("timestamp") - f.col("previous_ts")) \
                     .withColumn("gap_next_ts", f.col("next_ts") - f.col("timestamp")) \
                     .withColumn("display_id",
                                 f.when(f.col("gap_next_ts") == 0, f.lead("display_id").over(window_uuid_doc))
                                 .when(f.col("gap_prev_ts") == 0, f.lag("display_id").over(window_uuid_doc))
                                 .when((f.least(f.col("gap_next_ts"), f.col("gap_prev_ts")) == f.col("gap_next_ts")) &
                                       (f.col("gap_next_ts") < MAX_GAP_MINUTES * 60 * 1000),
                                       f.lead("display_id").over(window_uuid_doc))
                                 .when((f.least(f.col("gap_next_ts"), f.col("gap_prev_ts")) == f.col("gap_prev_ts")) & 
                                     (f.col("gap_next_ts") < MAX_GAP_MINUTES * 60 * 1000),
                                       f.lag("display_id").over(window_uuid_doc)))
                
    # page_views rows with matching event display_id
    page_views_events = page_views_events.filter("page_views = 1")
    
    # Creating sessions per user
    window_uuid = Window.partitionBy(page_views_events['uuid']).orderBy(page_views_events['timestamp'])
    page_views_events = page_views_events.withColumn("first_timestamp_sess",
                                                      f.when((f.row_number().over(window_uuid) == 1) |
                                                             (f.col("timestamp") - f.lag("timestamp").over(window_uuid)
                                                              > GAP_SESSION_MINUTES * 60 * 1000), f.col("timestamp")))
    
    page_views_events = page_views_events.withColumn("first_timestamp_sess",
                                                      f.last("first_timestamp_sess", ignorenulls=True)
                                                      .over(window_uuid.rowsBetween(Window.unboundedPreceding, 0)))
    
    # Time difference between two consecutive page views and since the start of the session, and all documents seen
    page_views_events = page_views_events.withColumn("gap_previous_timestamp_hours",
                                                      (f.col('timestamp') - f.lag("timestamp").over(window_uuid))
                                                      / 1000 / 60 / 60) \
                                          .withColumn("gap_start_session_minutes",
                                                      (f.col('timestamp') - f.col("first_timestamp_sess"))
                                                      / 1000 / 60 / 60) \
                                          .withColumn("list_documents_all", f.collect_list("document_id")
                                                      .over(window_uuid.rowsBetween(Window.unboundedPreceding, -1)))
    
    # Number of pages viewed in session
    window_session = Window.partitionBy(page_views_events['uuid'], page_views_events['first_timestamp_sess']) \
                           .orderBy(page_views_events['timestamp'])
    page_views_events = page_views_events.withColumn("nb_pages_session", f.row_number().over(window_session))
    
    # Writing page_views with a non-null display_id (=> that match an event)
    page_views_events.filter(page_views_events.display_id.isNotNull()) \
                     .write.mode("overwrite") \
                     .parquet(DIR_DATA_PARQUET + 'page_views_events')

    # ##################### #
    # Reading parquet files #
    # ##################### #
    categories = spark.read.parquet(DIR_DATA_PARQUET + 'documents_categories')
    topics = spark.read.parquet(DIR_DATA_PARQUET + 'documents_topics')
    meta = spark.read.parquet(DIR_DATA_PARQUET + 'documents_meta')
    content = spark.read.parquet(DIR_DATA_PARQUET + 'promoted_content')
    clicks_all = spark.read.parquet(DIR_DATA_PARQUET + 'clicks_all')
    page_views_events = spark.read.parquet(DIR_DATA_PARQUET + 'page_views_events')

    # ################# #
    # Adding event data #
    # ################# #
    print("Adding event data")
    # Splitting geo_location column
    events_new = events.withColumn('geo_location', f.split('geo_location', '>')) \
                       .withColumn('geo_location_struct',
                                   f.when(f.size("geo_location") > 1,
                                          f.struct(f.col("geo_location").getItem(0).alias("country"),
                                                   f.col("geo_location").getItem(1).alias("region")))
                                    .otherwise(f.struct(f.col("geo_location").getItem(0).alias("country"),
                                                        f.lit("Unknown").alias("region"))))

    # Computing the real timestamp (with regards to timezones)
    fields_map_tz = [StructField("country", StringType(), True),
                     StructField("region", StringType(), True),
                     StructField("timezone", StringType(), True)]

    map_tz = spark.read.load(DIR_DATA_CSV + "/tz_countries.csv",
                             format='com.databricks.spark.csv',
                             header='true',
                             schema=StructType(fields_map_tz),
                             delimiter=";")

    map_tz = map_tz.withColumn('geo_location_struct', f.struct("country", "region"))
    events_new = events_new.join(f.broadcast(map_tz.select("geo_location_struct", "timezone")),
                                 "geo_location_struct", "left_outer")

    events_new = events_new.withColumn("timestamp_utc", f.from_unixtime((f.col("timestamp") + FIRST_TIMESTAMP)/1000)) \
                           .withColumn("timestamp_local",
                                       f.when(f.col("timezone").isNotNull(),
                                              f.from_utc_timestamp(f.col("timestamp_utc"), f.col("timezone")))
                                        .otherwise(f.col("timestamp_utc")))

    # Adding new variables: hour, day of week, week end flag, moment of day, country (+ region if in US and CA)
    events_new = events_new.withColumn('hour_of_day', f.hour("timestamp_local")) \
                           .withColumn('day_of_week', f.date_format("timestamp_local", 'E')) \
                           .withColumn('is_weekend', f.when(f.col("day_of_week").isin(["Sat", "Sun"]), True)
                                                      .otherwise(False)) \
                           .withColumn("moment_of_day", f.when((f.col("hour_of_day") >= 7) &
                                                               (f.col("hour_of_day") < 12), "morning")
                                                         .when((f.col("hour_of_day") >= 12) &
                                                               (f.col("hour_of_day") < 18), "afternoon")
                                                         .when((f.col("hour_of_day") >= 18) &
                                                               (f.col("hour_of_day") < 23), "evening")
                                                         .when((f.col("hour_of_day") == 23) |
                                                               (f.col("hour_of_day") < 7), "night")) \
                           .withColumn("country_region", f.when(f.col("geo_location_struct.country").isin(["US", "CA"]),
                                                                f.concat_ws("_", f.col("geo_location_struct.country"),
                                                                f.col("geo_location_struct.region")))
                                                          .otherwise(f.col("geo_location_struct.country")))

    # Joining clicks all and events
    cols_events = ['display_id', 'document_id', 'timestamp_utc', 'platform', 'hour_of_day', 'day_of_week',
                   'is_weekend', 'moment_of_day', 'country_region']
    cols_page_views = ["display_id", "traffic_source", "gap_previous_timestamp_hours", "gap_start_session_minutes",
                       "list_documents_all", "nb_pages_session"]
    train_test = clicks_all.join(events_new.select(cols_events), "display_id", "left_outer")
    train_test = train_test.join(page_views_events.select(cols_page_views), "display_id", "left_outer") \

    # First checkpoint
    print("Writing train_test checkpoint 1")
    train_test.write.mode("overwrite").parquet(DIR_DATA_PARQUET + "train_test_v1")

    # ##################################################### #  
    # Adding documents metadata for events and ad documents # 
    # ##################################################### #  
    print("Adding documents meta")
    train_test = train_test.join(meta, "document_id", "left_outer")

    # Adding gap time between document publication and timestamp
    train_test = train_test.withColumn("doc_event_published_since",
                                       f.datediff("timestamp_utc", "publish_time") / 365.25)

    # Renaming columns with _event suffix
    cols_to_rename = ['document_id', 'source_id', 'publisher_id', 'publish_time']
    for c in cols_to_rename:
        train_test = train_test.withColumnRenamed(c, c + '_event')

    # Joining with promoted content and adding categories and topics vectors + metadata for ad documents
    train_test = train_test.join(content, "ad_id", "left_outer")\
                           .join(meta, "document_id", "left_outer")

    train_test = train_test.withColumn("doc_ad_published_since", f.datediff("timestamp_utc", "publish_time") / 365.25) \
                           .withColumn("time_diff_pub_ad_doc",
                                       f.datediff("publish_time", "publish_time_event") / 365.25)

    # Renaming columns with _ad suffix
    for c in cols_to_rename:
        train_test = train_test.withColumnRenamed(c, c + '_ad')

    # Second checkpoint
    print("Writing train_test checkpoint 2")
    train_test.write.mode("overwrite").parquet(DIR_DATA_PARQUET + 'train_test_v2')

    # ################### #
    # Adding click ratios #
    # ################### #
    print("Adding clicks ratios")
    train_test = spark.read.parquet(DIR_DATA_PARQUET + "train_test_v2")
    
    # Computing clicks ratios for different combinations of features (on train part)
    train = train_test.filter(f.col("train") == 1)
    clicks_ratio_grp = {"doc_id_ad": ["document_id_ad"],
                        "cpg_id": ["campaign_id"],
                        "adv_id": ["advertiser_id"],
                        "src_id_ad": ["source_id_ad"],
                        "pub_id_ad": ["publisher_id_ad"],
                        "pub_id_ev_doc_id_ad": ["publisher_id_event", "document_id_ad"],
                        "doc_id_ad_event": ["document_id_ad", "document_id_event"],
                        "doc_id_ad_cty": ["document_id_ad", "country_region"],
                        "doc_id_ad_day": ["document_id_ad", "day_of_week"],
                        "doc_id_ad_moment": ["document_id_ad", "moment_of_day"],
                        "doc_id_ad_platform": ["document_id_ad", "platform"]
                        }
    
    for (col_name, grp) in clicks_ratio_grp.items():
        # Only keeping ratios where nb of case > THRESHOLD_CLICK_RATIO
        clicks_ratio = train.groupby(grp)\
                            .agg(f.mean("clicked").alias("click_ratio_" + col_name), f.count("clicked").alias("count"))\
                            .filter(f.col("count") > THRESHOLD_CLICK_RATIO)
    
        train_test = train_test.join(clicks_ratio.drop("count"), grp, "left_outer")
    
    print("Writing train_test checkpoint 3")
    train_test.write.mode("overwrite").parquet(DIR_DATA_PARQUET + 'train_test_v3')

    # ################################################################################################################ #
    # Computing interactions between categories and topic for ad doc vs event doc and ad doc and previous user average #
    # ################################################################################################################ #
    print("Computing interactions between categories and topics")
    train_test = spark.read.parquet(DIR_DATA_PARQUET + "train_test_v3")

    display_hst = page_views_events.select("display_id", f.explode("list_documents_all").alias("document_id"))

    def average_history(df_cl, col_id):
        """
        Compute average confidence level per category/topic for each user for pages seen before display id events
        :param df_cl: Spark DataFrame containing confidence level per document
        :param col_id: id column for df_info

        :return:
        """
        df = display_hst.join(df_cl, "document_id")
        window_count = Window.partitionBy(df['display_id']).orderBy(df[col_id]) \
                             .rangeBetween(Window.unboundedPreceding, Window.unboundedFollowing)
        return df.withColumn("confidence_level", f.col("confidence_level") / f.count("document_id").over(window_count))
    
    def norm_df(df_cl, id_col, conf_col):
        """
        Compute norm of confidence level vectors per document
        :param df_cl:
        :param id_col:
        :param conf_col:

        :return:
        """
        return df_cl.withColumn("confidence_square", f.pow(f.col(conf_col), 2)) \
                    .groupby(id_col) \
                    .agg(f.sum("confidence_square").alias("norm_sq")) \
                    .withColumn("norm_2", f.sqrt(f.col("norm_sq"))) \
                    .select(id_col, "norm_2")


    def compute_interaction(df, grp_cols, conf_level_cols, norms, suffix_output):
        """
        Compute cosine distance and interaction between two confidence level vectors
        :param df: Spark DataFrame which contains two confidence level columns to compute interaction per group
        :param grp_cols: List of columns to group by for computing e.g [document_id_ad, document_id_event]
        :param conf_level_cols: list of the two confidence levels columns
        :param norms: norms DataFrame (as computed by norm_df)
        :param suffix_output: suffix for computed columns

        :return: Spark DataFrame with grouping columns, maximum interaction, number of non null interaction and cosine
        distance
        """

        # Computing interaction between the two confidence level columns per group
        df_new = df.withColumn("confidence_dot", f.col(conf_level_cols[0]) * f.col(conf_level_cols[1]))
        df_new = df_new.groupBy(grp_cols) \
                       .agg(f.sum("confidence_dot").alias("confidence_dot"),
                            f.max("confidence_dot").alias("max_interaction" + suffix_output),
                            f.count(conf_level_cols[0]).alias("count_interaction" + suffix_output))

        # Adding cosine distance between confidence level vectors
        df_new = df_new.join(norms[0].withColumnRenamed("norm_2", "norm_20"), norms[0].columns[0]) \
                       .join(norms[1].withColumnRenamed("norm_2", "norm_21"), norms[1].columns[0]) \
                       .withColumn("cosine_distance" + suffix_output,
                                   f.col("confidence_dot") / (f.col("norm_20") * f.col("norm_21")))
        cols_select = grp_cols + ["max_interaction" + suffix_output, "count_interaction" + suffix_output,
                                  "cosine_distance" + suffix_output]
        return df_new.select(*cols_select)
    
    # Averaging confidence level for each display id
    avg_cat_hist = average_history(categories, "category_id")
    avg_top_hist = average_history(topics, "topic_id")
    
    # Computing confidence level vector norms per documents
    norm_categories = norm_df(categories, "document_id", "confidence_level")
    norm_topics = norm_df(topics, "document_id", "confidence_level")

    # Set of all document_id for ad and events
    doc_id_ad_event = train_test.select("document_id_ad", "document_id_event").dropDuplicates()
    # Interaction between ad and events categories
    doc_id_ad_event_cat = doc_id_ad_event.join(categories.withColumnRenamed("document_id", "document_id_ad")
                                                         .withColumnRenamed("confidence_level", "confidence_level_ad"),
                                               "document_id_ad") \
                                         .join(categories.withColumnRenamed("document_id", "document_id_event")
                                                         .withColumnRenamed("confidence_level",
                                                                            "confidence_level_event"),
                                               ["document_id_event", "category_id"])

    norms_cat = [norm_categories.withColumnRenamed("document_id", "document_id_event"),
                 norm_categories.withColumnRenamed("document_id", "document_id_ad")]

    int_ad_event_cat = compute_interaction(doc_id_ad_event_cat, ["document_id_event", "document_id_ad"],
                                           ["confidence_level_ad", "confidence_level_event"], norms_cat, "_cat_ad_ev")

    # Interaction between ad document and events topics
    doc_id_ad_event_top = doc_id_ad_event.join(topics.withColumnRenamed("document_id", "document_id_ad")
                                               .withColumnRenamed("confidence_level", "confidence_level_ad"),
                                               "document_id_ad") \
                                         .join(topics.withColumnRenamed("document_id", "document_id_event")
                                               .withColumnRenamed("confidence_level", "confidence_level_event"),
                                               ["document_id_event", "topic_id"])

    norms_top = [norm_topics.withColumnRenamed("document_id", "document_id_event"),
                 norm_topics.withColumnRenamed("document_id", "document_id_ad")]

    int_ad_event_top = compute_interaction(doc_id_ad_event_top, ["document_id_event", "document_id_ad"],
                                           ["confidence_level_ad", "confidence_level_event"], norms_top, "_top_ad_ev")

    # Interaction between ad documents and user history category
    avg_cat_hist = avg_cat_hist.groupBy("display_id", "category_id")\
                               .agg(f.sum("confidence_level").alias("confidence_level_history"))

    norm_cat_hist = norm_df(avg_cat_hist, "display_id", "confidence_level_history")

    doc_ad_hist_cat = avg_cat_hist.join(train_test.select("display_id", "ad_id", "document_id_ad"), "display_id")

    doc_ad_hist_cat = doc_ad_hist_cat.join(categories.withColumnRenamed("document_id", "document_id_ad"),
                                           ["document_id_ad", "category_id"]) \
                                     .withColumnRenamed("confidence_level", "confidence_level_document")

    int_ad_hist_cat = compute_interaction(doc_ad_hist_cat,
                                          ["document_id_ad", "display_id", "ad_id"],
                                          ["confidence_level_history", "confidence_level_document"],
                                          [norm_cat_hist,
                                           norm_categories.withColumnRenamed("document_id", "document_id_ad")],
                                          "_cat_ad_hist")

    # Interaction between ad document and user history topics
    avg_top_hist = avg_top_hist.groupBy("display_id", "topic_id")\
                               .agg(f.sum("confidence_level").alias("confidence_level_history"))
    norm_top_hist = norm_df(avg_top_hist, "display_id", "confidence_level_history")
    doc_ad_hist_top = avg_top_hist.join(train_test.select("display_id", "ad_id", "document_id_ad"),
                                        "display_id")
    doc_ad_hist_top = doc_ad_hist_top.join(topics.withColumnRenamed("document_id", "document_id_ad"),
                                           ["document_id_ad", "topic_id"]) \
                                     .withColumnRenamed("confidence_level", "confidence_level_document")
    int_ad_hist_top = compute_interaction(doc_ad_hist_top, ["document_id_ad", "display_id", "ad_id"],
                                          ["confidence_level_history", "confidence_level_document"],
                                          [norm_top_hist,
                                           norm_topics.withColumnRenamed("document_id", "document_id_ad")],
                                          "_top_ad_hist")

    # Joining interaction with train_test data set
    train_test = train_test.join(int_ad_event_cat, ["document_id_event", "document_id_ad"], "left_outer") \
                           .join(int_ad_event_top, ["document_id_event", "document_id_ad"], "left_outer") \
                           .join(int_ad_hist_cat.drop("document_id_ad"), ["display_id", "ad_id"], "left_outer") \
                           .join(int_ad_hist_top.drop("document_id_ad"), ["display_id", "ad_id"], "left_outer")

    print("Writing train_test checkpoint 4")
    train_test.write.mode("overwrite").parquet(DIR_DATA_PARQUET + "train_test_v4")

    # #################################### #
    # Adding data over the display context #
    # #################################### #
    print("Adding data over display")
    train_test = spark.read.parquet(DIR_DATA_PARQUET + "train_test_v4")
    # Adding number of ads in display
    window_display = Window.partitionBy(train_test['display_id'])
    train_test = train_test.withColumn("count_ads_in_display",
     f.count("ad_id").over(window_display.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)))
    # Computing ranks for each group of click ratios
    for col_name in clicks_ratio_grp.keys():
        train_test = train_test.withColumn("rank_click_ratio_" + col_name,
                                           f.percent_rank()
                                            .over(window_display.orderBy(f.desc_nulls_last("click_ratio_" + col_name))))

    # Computing ranks in display for cosine distances between ad and event documents
    train_test = train_test.withColumn("rank_cosine_cat_ad_event",
                                       f.percent_rank()
                                       .over(window_display.orderBy(f.desc_nulls_last("cosine_distance_cat_ad_ev"))))

    train_test = train_test.withColumn("rank_cosine_top_ad_event",
                                       f.percent_rank()
                                       .over(window_display.orderBy(f.desc_nulls_last("cosine_distance_top_ad_ev"))))

    print("Writing train_test checkpoint 5")
    train_test.write.mode("overwrite").parquet(DIR_DATA_PARQUET + "train_test_v5")

    print("Final train_test:")
    train_test = spark.read.parquet(DIR_DATA_PARQUET + "train_test_v5")
    print("Count:", train_test.count())
    print("Columns:", train_test.columns)
    train_test.show(5)
