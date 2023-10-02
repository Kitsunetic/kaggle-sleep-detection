import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from pdb import set_trace

import h5py
import numpy as np
import pandas as pd
import torch as th
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

BATCH_SIZE = 4


class ParquetLoader(Dataset):
    def __init__(self, path, series_id_to_indices: dict):
        super().__init__()
        self.df = pd.read_parquet(path, engine="pyarrow")
        self.series_ids = list(series_id_to_indices.keys())
        self.series_id_to_indices = series_id_to_indices

    def __len__(self):
        return len(self.series_ids)

    def __getitem__(self, idx):
        series_id = self.series_ids[idx]
        indices = self.series_id_to_indices[series_id]

        data = self.df.iloc[indices]
        step = data.step.to_numpy().astype(np.int32)
        timestamp = []
        for x in data.timestamp:
            if isinstance(x, float):
                timestamp.append(x)
            else:
                timestamp.append(datetime.strptime(x, "%Y-%m-%dT%H:%M:%S%z").timestamp())
        timestamp = np.array(timestamp, dtype=np.float32)
        anglez = data.anglez.to_numpy().astype(np.float32)
        enmo = data.enmo.to_numpy().astype(np.float32)

        return series_id, step, timestamp, anglez, enmo


def export_parquet(out_file, path, df):
    series_id_to_indices = defaultdict(list)
    for i, series_id in enumerate(df.series_id):
        series_id_to_indices[series_id].append(i)
    del df

    ds = ParquetLoader(path, series_id_to_indices)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=BATCH_SIZE, collate_fn=lambda x: x)

    name = Path(path).stem
    g_main = out_file.create_group(name)
    with tqdm(total=len(ds), ncols=100, desc=name) as pbar:
        for batch in dl:
            for series_id, step, timestamp, anglez, enmo in batch:
                g = g_main.create_group(series_id)
                g.create_dataset("step", data=step)
                g.create_dataset("timestamp", data=timestamp)
                g.create_dataset("anglez", data=anglez)
                g.create_dataset("enmo", data=enmo)
                pbar.update()


class EventLoader(Dataset):
    def __init__(self, path, series_id_to_indices: dict):
        super().__init__()
        self.df = pd.read_csv(path)
        self.series_ids = list(series_id_to_indices.keys())
        self.series_id_to_indices = series_id_to_indices

    def __len__(self):
        return len(self.series_ids)

    def __getitem__(self, idx):
        series_id = self.series_ids[idx]
        indices = self.series_id_to_indices[series_id]

        data = self.df.iloc[indices]
        night = data.night.to_numpy().astype(np.int32)
        event = data.event.tolist()
        event = np.array([x == "onset" for x in event], dtype=np.uint8)
        step = data.step.to_numpy().astype(np.int32)
        timestamp = []
        for x in data.timestamp:
            if isinstance(x, float):
                timestamp.append(x)
            else:
                timestamp.append(datetime.strptime(x, "%Y-%m-%dT%H:%M:%S%z").timestamp())
        timestamp = np.array(timestamp, dtype=np.float32)

        return series_id, night, event, step, timestamp


def export_events(out_file, path, df):
    series_id_to_indices = defaultdict(list)
    for i, series_id in enumerate(df.series_id):
        series_id_to_indices[series_id].append(i)
    del df

    ds = EventLoader(path, series_id_to_indices)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=BATCH_SIZE, collate_fn=lambda x: x)

    name = Path(path).stem
    g_main = out_file.create_group(name)
    with tqdm(total=len(ds), ncols=100, desc=name) as pbar:
        for batch in dl:
            for series_id, night, event, step, timestamp in batch:
                g = g_main.create_group(series_id)
                g.create_dataset("night", data=night)
                g.create_dataset("event", data=event)
                g.create_dataset("step", data=step)
                g.create_dataset("timestamp", data=timestamp)
                pbar.update()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_only", action="store_true")
    args = parser.parse_args()

    out_file = h5py.File("data/mydata_v231001.hdf5", "w")

    # test_series
    path = "../data/kaggle-sleep-detection/test_series.parquet"
    df = pd.read_parquet(path, engine="pyarrow")
    export_parquet(out_file, path, df)

    if args.test_only:
        return

    # train_events
    path = "../data/kaggle-sleep-detection/train_events.csv"
    df = pd.read_csv(path)
    export_events(out_file, path, df)

    # train_series
    path = "../data/kaggle-sleep-detection/train_series.parquet"
    df = pd.read_parquet(path, engine="pyarrow")
    export_parquet(out_file, path, df)


if __name__ == "__main__":
    main()
