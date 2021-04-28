from io import StringIO
import boto3
import pandas as pd
from fusion import TransEFuser

BUCKET_NAME = ""
KEY = ""


def main():
    s3 = boto3.client("s3")
    s3_resource = boto3.resource("s3")
    s3.download_file(BUCKET_NAME, f"{KEY}00000-combined.csv", "df.csv")
    df = pd.read_csv("df.csv")
    df = df.iloc[:100000]
    fuser = TransEFuser(embed_dim=3, max_epochs=2, n_splits=3)
    kg_df = fuser.fuse(df)
    csv_buffer = StringIO()
    kg_df.to_csv(csv_buffer, index=False)
    s3_resource.Object(BUCKET_NAME, f"{KEY}test.csv").put(
        Body=csv_buffer.getvalue()
    )


if __name__ == "__main__":
    main()
