# FishLens YOLOv11s Android Detector
**33 Bangladeshi fish species · Single-image inference · Samsung Galaxy A17 5G**  
TFLite backend · GPU-accelerated · No internet required after install

---

## Your 33 fish classes (extracted from your notebook)

| # | Class | # | Class | # | Class |
|---|-------|---|-------|---|-------|
| 0 | Batashi | 11 | Mola | 22 | Telapia |
| 1 | Bele | 12 | Mrigel | 23 | Tengra |
| 2 | Chanda | 13 | Pabda | 24 | Veda |
| 3 | Chapila | 14 | Pangas | 25 | Vetki |
| 4 | Chela | 15 | Puti | 26 | ayer |
| 5 | Foli | 16 | Rui | 27 | boyal |
| 6 | Ilish | 17 | Rupchada | 28 | chitol |
| 7 | Kajuli | 18 | Shol | 29 | kalbaus |
| 8 | Kayakanta | 19 | Sorputi | 30 | katol |
| 9 | Koi | 20 | Taki | 31 | kholshe |
| 10 | Magur | 21 | Tarabaim | 32 | shing |

The `labels.txt` file is already filled in — you do NOT need to edit it.

---

## Step 1 — Export your model (on your PC, one-time)

```bash
pip install ultralytics

python -c "
from ultralytics import YOLO
model = YOLO('best.pt')          # your trained weights
model.export(format='tflite', imgsz=640, batch=1)
"
```

Rename the output `best_float32.tflite` → `best.tflite`  
Copy it into: `YoloDetector/app/src/main/assets/best.tflite`

---

## Step 2 — Build the APK (choose ONE method)

### Option A — Android Studio (easiest, recommended)
1. Download [Android Studio](https://developer.android.com/studio) (free)
2. Open folder → wait for Gradle sync → click **Run ▶** with phone connected

---

### Option B — Command Line only (NO Android Studio UI needed)

You only need the **Android SDK command-line tools** — much smaller than full Android Studio.

#### Install on Windows
```powershell
# 1. Download command-line tools from:
#    https://developer.android.com/studio#command-line-tools-only
# 2. Unzip to C:\android-sdk\cmdline-tools\latest\

# 3. Set environment variables (run once in PowerShell):
[System.Environment]::SetEnvironmentVariable("ANDROID_HOME", "C:\android-sdk", "User")
$env:Path += ";C:\android-sdk\cmdline-tools\latest\bin;C:\android-sdk\platform-tools"

# 4. Install required SDK components:
sdkmanager "platform-tools" "platforms;android-34" "build-tools;34.0.0"
```

#### Install on Linux/Mac
```bash
# Download cmdline-tools from the same URL above, then:
export ANDROID_HOME=$HOME/android-sdk
export PATH=$PATH:$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools

sdkmanager "platform-tools" "platforms;android-34" "build-tools;34.0.0"
```

#### Build the APK
```bash
# From inside the YoloDetector/ folder:
./gradlew assembleDebug        # Linux/Mac
gradlew.bat assembleDebug      # Windows
```

The APK appears at:
```
app/build/outputs/apk/debug/app-debug.apk
```

#### Install to your Samsung A17 5G
```bash
# Enable USB Debugging on your phone:
# Settings → About phone → tap Build number 7 times
# Settings → Developer options → USB debugging ON

# Connect phone via USB, then:
adb install app/build/outputs/apk/debug/app-debug.apk
```

---

### Option C — GitHub Actions (build in the cloud, no SDK install)

Create `.github/workflows/build.yml` in your repo:
```yaml
name: Build APK
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-java@v4
        with:
          java-version: '17'
          distribution: 'temurin'
      - name: Build APK
        run: cd YoloDetector && ./gradlew assembleDebug
      - name: Upload APK
        uses: actions/upload-artifact@v4
        with:
          name: fishlens-debug
          path: YoloDetector/app/build/outputs/apk/debug/app-debug.apk
```
Push to GitHub → download the APK from the Actions tab — no local SDK needed at all.

---

### Option D — Buildozer / EAS (online build services)

[Appetize.io](https://appetize.io), [Codemagic](https://codemagic.io), and [Bitrise](https://bitrise.io)
all offer free Android builds. Upload the project, connect your repo, and download the APK.

---

## File structure

```
YoloDetector/
├── app/src/main/assets/
│   ├── best.tflite      ← COPY YOUR MODEL HERE
│   └── labels.txt       ← already filled with your 33 fish classes ✓
├── app/src/main/java/com/yolo/detector/
│   ├── ml/YoloDetector.kt        inference engine + NMS
│   └── ui/
│       ├── MainActivity.kt       camera + gallery + UI
│       └── BoundingBoxOverlay.kt drawing layer
└── README.md
```

---

## Performance on Samsung A17 5G

| Format | Expected speed |
|--------|---------------|
| float32 TFLite + GPU | ~150–300 ms |
| float32 TFLite + CPU | ~300–600 ms |
| int8 TFLite + GPU | ~80–150 ms |

The app auto-selects GPU if available, falls back to 4-thread CPU silently.

---

## Troubleshooting

**"Model not found"** — make sure the file is literally named `best.tflite` (not `best_float32.tflite`).

**All boxes say "class_N"** — labels.txt mismatch. Already pre-filled for your 33 classes.

**`gradlew: Permission denied`** (Linux/Mac) — run `chmod +x gradlew` first.

**`sdkmanager not found`** — ensure `cmdline-tools/latest/bin` is on your PATH.
