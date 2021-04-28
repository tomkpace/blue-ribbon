import os
from io import StringIO
import boto3
import pandas as pd
from knowledge_graph_generator import KnowledgeGraphGenerator

BUCKET_NAME = ""
OUTPUT_BUCKET_NAME = ""
KEY = ""


def main():
    s3 = boto3.client("s3")
    s3_resource = boto3.resource("s3")
    s3.download_file(BUCKET_NAME, KEY, "df.csv")
    df = pd.read_csv("df.csv")
    for i, entity in enumerate(df["entity_id"].unique()):
        review_df = df.loc[df["entity_id"] == entity]
        generator = KnowledgeGraphGenerator(
            input_data_list=[review_df],
        )
        kg_df = generator.knowledge_graph_df.drop_duplicates().reset_index(
            drop=True
        )
        str_entity = entity.replace("/", "-")
        csv_buffer = StringIO()
        kg_df.to_csv(csv_buffer, index=False)
        s3_resource.Object(
            BUCKET_NAME, f"{OUTPUT_PREFIX}{str_entity}.csv"
        ).put(Body=csv_buffer.getvalue())


if __name__ == "__main__":
    main()
