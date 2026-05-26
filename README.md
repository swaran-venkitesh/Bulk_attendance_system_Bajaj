# Bulk Attendance System Bajaj

## Overview
This repository contains a Python-based face-recognition attendance workflow prepared as a private code backup. It focuses on bulk attendance processing using face detection, embedding generation, and identity matching.

## Included
- `main2.py`
- `face_detector.py`
- `face_recognizer.py`
- `adaface_lib/net.py`
- `requirements.txt`
- `models/README.md`

## Excluded from Git
- attendance databases and enrolled embeddings
- runtime logs and generated outputs
- local captures and exported attendance files
- model weight files

## Setup
1. Install dependencies:

```powershell
pip install -r requirements.txt
```

2. Restore the required face-detection and recognition model files locally in the `models/` folder.

3. Restore the local attendance embedding database in the `database/` folder if needed.

4. Run:

```powershell
python main2.py
```

## Status
Private repository for code backup and controlled sharing.
