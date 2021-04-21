import os
import io
import boto3
import pandas as pd
from knowledge_graph_generator import KnowledgeGraphGenerator

BUCKET_NAME = ""
OUTPUT_BUCKET_NAME = ""
KEY = ""


def main():
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=KEY)
    df = pd.read_csv(io.BytesIO(obj["Body"].read()), encoding="utf8")
    for i, entity in enumerate(df["entity_id"].unique()):
        review_df = df.loc[df["entity_id"] == entity]
        generator = KnowledgeGraphGenerator(
            input_data_list=[review_df],
        )
        kg_df = generator.knowledge_graph_df.drop_duplicates().reset_index(
            drop=True
        )
        str_entity = entity.replace("/", "-")
        kg_df.to_csv(f"{str_entity}.csv", index=False)
        s3.upload_file(
            f"{str_entity}.csv", OUTPUT_BUCKET_NAME, f"{str_entity}.csv"
        )
        os.remove(f"{str_entity}.csv")


if __name__ == "__main__":
    main()
