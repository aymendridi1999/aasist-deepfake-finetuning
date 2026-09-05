"""Behavioral regression tests using tiny fixtures, never benchmark results."""
from __future__ import annotations
import ast
import json
import os
import random
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import hashlib
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import librosa
import soundfile as sf
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = json.loads(next((ROOT / "notebooks").glob("*.ipynb")).read_text())
NS = dict(globals(), SEED=42, DEVICE=torch.device("cpu"), SPOOF_CLASS=0,
    SAMPLE_RATE=16000, DURATION_SECONDS=3.0, MEL_BINS=64, MFCC_BINS=40,
    N_FFT=1024, HOP_LENGTH=256, FEATURE_VERSION="regression",
    CLIP_SAMPLES=64600, ACCUMULATION_STEPS=4, BATCH_SIZE=2,
    VALIDATION_SIZE=0.2, AUDIO_EXTENSIONS={".wav", ".flac"},
    LABEL_TO_INDEX={"spoof":0,"bonafide":1},
    NON_FEATURE_COLUMNS={"id", "attack_cat", "srcip", "dstip", "stime", "ltime"},
    ID_PATTERN=re.compile(r"((pa|la|df)[_-]?e[_-]?\d+)",re.I),
    tqdm=lambda iterable, **kwargs: iterable)
for cell in NOTEBOOK["cells"]:
    source = "".join(cell["source"])
    if cell["cell_type"] != "code" or any(line.startswith(("%", "!")) for line in source.splitlines()):
        continue
    tree = ast.parse(source)
    definitions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), "notebook", "exec"), NS)

class MetricTests(unittest.TestCase):
    def test_measured_results_change_with_predictions(self):
        metric = NS.get("calculate_metrics", NS.get("metric_summary"))
        y = [0,0,1,1]
        score = [0.9,0.8,0.2,0.1] if "metric_summary" in NS else [0.1,0.2,0.8,0.9]
        good = metric(y,y,score)
        bad = metric(y,[1-x for x in y],[1-x for x in score])
        self.assertEqual(good["accuracy"],1.0)
        self.assertEqual(bad["accuracy"],0.0)
        self.assertEqual(good["auroc"],1.0)
        self.assertEqual(bad["auroc"],0.0)
        self.assertEqual(metric(y,[0,1,0,1],score)["accuracy"],0.5)
        self.assertIsNone(metric([0,0],[0,0],[0.4,0.5])["auroc"])
        with self.assertRaises(ValueError): metric([],[],[])
        with self.assertRaises(ValueError): metric(y,y,[float("nan")]*4)
        json.dumps(good, allow_nan=False)

class ProtocolTests(unittest.TestCase):
    def test_official_and_compact_cm_schemas(self):
        records = [
            ("PA", "PA_0010 PA_E_1000001 R3 M3 d4 r1 m1 s4 c4 bonafide notrim eval"),
            ("LA", "LA_0009 LA_E_9332881 alaw ita_tx A07 spoof notrim eval"),
            ("DF", "LA_0023 DF_E_2000011 nocodec asvspoof A14 spoof notrim progress traditional_vocoder - - - -"),
            ("DF", "LA_0023 DF_E_2000011 nocodec asvspoof - bonafide notrim progress bonafide - - - -"),
            ("DF", "DF_E_2000011 bonafide eval"),
            ("LA", "LA_0001 LA_E_0000001 - A07 spoof"),
        ]
        for track, row in records:
            file_id,label,subset = NS["protocol_record"](row.split(),track)
            self.assertTrue(file_id.startswith(track+"_E_"))
            self.assertEqual(label, "bonafide" if "bonafide" in row else "spoof")
        for row in ["DF_E_1 unknown eval", "LA_0001 DF_E_1 target", "PA_E_1 spoof eval"]:
            with self.assertRaises(ValueError): NS["protocol_record"](row.split(),"DF")

    def test_duplicate_and_conflicting_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"protocol.txt"
            path.write_text("DF_E_1 bonafide eval\nDF_E_1 bonafide eval\n")
            parse = (lambda p: NS["parse_protocol"](p,"DF")) if "parse_protocol" in NS else NS["load_protocol"]
            self.assertEqual(len(parse(path)),1)
            path.write_text("DF_E_1 bonafide eval\nDF_E_1 spoof eval\n")
            with self.assertRaises(ValueError): parse(path)
            path.write_text("")
            with self.assertRaises(ValueError): parse(path)

    def test_audio_decoding_and_empty_rejection(self):
        load=NS.get("load_audio",NS.get("load_waveform"))
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"fixture.wav"
            sf.write(path,np.zeros((8000,2),dtype=np.float32),8000)
            signal=load(path)
            self.assertTrue(np.isfinite(signal).all())
            self.assertEqual(len(signal),48000 if "load_audio" in NS else 16000)
            sf.write(path,np.array([],dtype=np.float32),16000)
            with self.assertRaises(ValueError): load(path)

class AASISTTests(unittest.TestCase):
    def test_clip_and_eer(self):
        clip=NS["fit_clip"](np.arange(100,dtype=np.float32),random_crop=False)
        self.assertEqual(len(clip),64600)
        np.testing.assert_array_equal(clip[:100],np.arange(100))
        with self.assertRaises(ValueError): NS["fit_clip"](np.array([]),random_crop=False)
        self.assertEqual(NS["equal_error_rate"](np.array([0,0,1,1]),np.array([.9,.8,.2,.1])),0.0)

    def test_partial_accumulation_matches_reference(self):
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__(); self.layer=nn.Linear(3,2)
            def forward(self,x,Freq_aug=False): return x,self.layer(x)
        torch.manual_seed(42)
        x=torch.randn(12,3); y=torch.tensor([0,1]*6)
        model=Tiny(); reference=Tiny(); reference.load_state_dict(model.state_dict())
        NS.update(model=model,criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.SGD(model.parameters(),lr=0.01),
            gradient_scaler=torch.amp.GradScaler("cuda",enabled=False),
            train_loader=DataLoader(TensorDataset(x,y,torch.arange(12)),batch_size=2))
        NS["train_epoch"]()
        opt=torch.optim.SGD(reference.parameters(),lr=0.01)
        for start in [0,8]:
            opt.zero_grad()
            loss=nn.CrossEntropyLoss()(reference(x[start:start+8])[1],y[start:start+8])
            loss.backward(); torch.nn.utils.clip_grad_norm_(reference.parameters(),5.0); opt.step()
        for actual,expected in zip(model.parameters(),reference.parameters()):
            torch.testing.assert_close(actual,expected)


if __name__ == "__main__":
    unittest.main()
