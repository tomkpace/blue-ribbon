# Script for performing the fusion method on SageMaker.
from io import StringIO
import boto3
import numpy as np
import pandas as pd
from knowledge_graph_generator import KnowledgeGraphGenerator
from fusion import TransEFuser

BUCKET_NAME = ""
INPUT_KEY = ""
OUTPUT_KEY = ""
COUNT_LIMIT = 250


def main():
    s3 = boto3.client("s3")
    s3_resource = boto3.resource("s3")
    s3.download_file(BUCKET_NAME, f"{INPUT_KEY}kg_count_year_df.csv", "df.csv")
    full_df = pd.read_csv("df.csv")
    df = full_df.loc[full_df["arc_count"] >= COUNT_LIMIT].reset_index(
        drop=True
    )
    df = df[["entity_id", "relation", "value"]]
    kg_obj = KnowledgeGraphGenerator(known_data_list=[df])
    fuser = TransEFuser(kg_obj)
    kg_df = fuser.fuse()
    csv_buffer = StringIO()
    kg_df.to_csv(csv_buffer, index=False)
    s3_resource.Object(
        BUCKET_NAME, f"{OUTPUT_KEY}kg/count_limit_{COUNT_LIMIT}.csv"
    ).put(Body=csv_buffer.getvalue())


if __name__ == "__main__":
    main()
