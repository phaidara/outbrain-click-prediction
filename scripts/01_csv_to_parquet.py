from pyspark.sql.types import *
from pyspark.sql import SparkSession

"""
This script reads all the csv files and rewrites them in Parquet.
"""

DIR_DATA_CSV = "./data/csv/"
DIR_DATA_PARQUET = "./data/parquet/"


if __name__ == "__main__":

    spark = SparkSession \
        .builder \
        .appName('csvToParquetApp') \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    def csv_to_parquet(filename, schema, input_dir=DIR_DATA_CSV, output_dir=DIR_DATA_PARQUET):
        """
        Reads a comma-separated csv file (with header) and write in Parquet
        :param filename: csv filename without extension
        :param schema: Spark Schema for DataFrame reader
        :param input_dir: input directory for csv file
        :param output_dir: output directory for parquet file
        :return:
        """
        df = spark.read.load(input_dir + "/" + filename + ".csv",
                             format='com.databricks.spark.csv',
                             header='true',
                             schema=schema,
                             delimiter=',')
        print(filename + " - number of lines:", "{:,}".format(df.count()))
        df.write.mode('overwrite').parquet(output_dir + "/" + filename)


    # Reading clicks_train.csv
    fields_clicks_train = [StructField("display_id", IntegerType(), True),
                           StructField("ad_id", IntegerType(), True),
                           StructField("clicked", IntegerType(), True)]
    csv_to_parquet('clicks_train', StructType(fields_clicks_train))

    # Reading clicks_test.csv
    fields_clicks_test = [StructField("display_id", IntegerType(), True),
                          StructField("ad_id", IntegerType(), True)]
    csv_to_parquet('clicks_test', StructType(fields_clicks_test))

    # Reading events.csv
    fields_events = [StructField("display_id", IntegerType(), True),
                     StructField("uuid", StringType(), True),
                     StructField("document_id", IntegerType(), True),
                     StructField("timestamp", IntegerType(), True),
                     StructField("platform", IntegerType(), True),
                     StructField("geo_location", StringType(), True)]
    csv_to_parquet('events', StructType(fields_events))

    # Reading promoted_content
    fields_content = [StructField("ad_id", IntegerType(), True),
                      StructField("document_id", IntegerType(), True),
                      StructField("campaign_id", IntegerType(), True),
                      StructField("advertiser_id", IntegerType(), True)
                      ]
    csv_to_parquet('promoted_content', StructType(fields_content))

    # Reading document_categories
    fields_doc_cat = [StructField("document_id", IntegerType(), True),
                      StructField("category_id", IntegerType(), True),
                      StructField("confidence_level", DoubleType(), True)]
    csv_to_parquet('documents_categories', StructType(fields_doc_cat))

    # Reading document_topics
    fields_doc_topic = [StructField("document_id", IntegerType(), True),
                        StructField("topic_id", IntegerType(), True),
                        StructField("confidence_level", DoubleType(), True)]
    csv_to_parquet('documents_topics', StructType(fields_doc_topic))

    # Reading document_meta
    fields_doc_meta = [StructField("document_id", IntegerType(), True),
                       StructField("source_id", IntegerType(), True),
                       StructField("publisher_id", IntegerType(), True),
                       StructField("publish_time", TimestampType(), True)]
    csv_to_parquet('documents_meta', StructType(fields_doc_meta))

    # Reading page_views
    fields_pv = [StructField("uuid", StringType(), True),
                 StructField("document_id", IntegerType(), True),
                 StructField("timestamp", IntegerType(), True),
                 StructField("platform", IntegerType(), True),
                 StructField("geo_location", StringType(), True),
                 StructField("traffic_source", IntegerType(), True)]
    csv_to_parquet('page_views', StructType(fields_pv))
