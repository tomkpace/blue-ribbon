from io import StringIO
import boto3
import numpy as np
import pandas as pd
from fusion import TransEFuser

BUCKET_NAME = ""
KEY = ""


def main():
    s3 = boto3.client("s3")
    s3_resource = boto3.resource("s3")
    s3.download_file(BUCKET_NAME, f"{KEY}00000-combined.csv", "df.csv")
    full_df = pd.read_csv("df.csv")
    entities = full_df["entity_id"].unique()
    np.random.shuffle(entities)
    for i in range(0, len(entities), 25):
        entity_group = entities[i : i + 25]
        df = full_df.loc[full_df["entity_id"].isin(entity_group)]
        fuser = TransEFuser(embed_dim=50, n_splits=5)
        kg_df = fuser.fuse(df)
        csv_buffer = StringIO()
        kg_df.to_csv(csv_buffer, index=False)
        s3_resource.Object(BUCKET_NAME, f"{KEY}kg/{i}.csv").put(
            Body=csv_buffer.getvalue()
        )


if __name__ == "__main__":
    main()
