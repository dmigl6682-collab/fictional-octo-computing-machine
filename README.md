# AI Mobile App - On-Device Android Edge AI Suite

Production-ready Kivy & NumPy on-device AI Android application with automated Buildozer compilation.

## Included Files
- `main.py`: Native Kivy multi-touch UI + vectorized on-device inference (KNN, MLP Neural Network, Anomaly Detector, Linear Regressor).
- `buildozer.spec`: Target Android API 34, NDK 25b, ARM64-v8a build specification.
- `build_apk.sh`: 1-click Linux compilation script.
- `test_models.py`: PyTest vector math unit tests.
- `model_weights.json`: Pre-trained float32 tensors and calibration dataset.
- `.github/workflows/build-apk.yml`: Free cloud compilation via GitHub Actions (outputs ready-to-install `.apk`).

---

## Method 1: Free Cloud Compilation (Zero Local Setup - Recommended)
1. Create a free GitHub repository and push these files to it.
2. The included `.github/workflows/build-apk.yml` will automatically trigger.
3. Once completed (approx. 8-12 mins on first run), go to your GitHub repository's **Actions** tab.
4. Click on the latest run and download the **`aimobileapp-apk`** artifact (contains `aimobileapp-1.0.0-arm64-v8a-debug.apk`).
5. Transfer the `.apk` to your Android phone and install.

---

## Method 2: Compile Locally on Linux / Ubuntu / WSL2
### Step 1: Install Host Build Dependencies
```bash
sudo apt update
sudo apt install -y git zip unzip openjdk-17-jdk python3-pip autoconf libtool \
  pkg-config zlib1g-dev libncurses5-dev libncursesw5-dev libtinfo5 cmake \
  libffi-dev libssl-dev build-essential cython3
```

### Step 2: Install Buildozer
```bash
pip3 install --upgrade buildozer cython virtualenv
```

### Step 3: Compile the APK
```bash
buildozer android debug
```
The compiled APK will be created in the `bin/` directory:
`bin/aimobileapp-1.0.0-arm64-v8a-debug.apk`

### Step 4: Sideload & Test on USB-Connected Android Device
```bash
buildozer android deploy run logcat
```

---

## Method 3: Test on Desktop PC before compiling
Run the app directly in desktop Python:
```bash
pip install kivy numpy
python3 main.py
```
