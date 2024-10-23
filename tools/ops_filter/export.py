import pandas as pd
from pathlib import Path


def export_csv(dir, topic, columns, records):
    p_dir = Path(dir)
    assert p_dir.is_dir()
    p_file = p_dir / f"{topic}.csv"
    df = pd.DataFrame(records, columns=columns)
    df.to_csv(p_file, index=False)
