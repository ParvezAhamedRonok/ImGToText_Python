IELTS Plus — OCR Backend Project Summary
IELTS Plus — OCR Backend
Complete Setup & Current Status

Tech Stack: Python 3.11.5 + FastAPI + OpenCV + PaddleOCR + PaddlePaddle
Platform: Windows + Git Bash / MINGW64
Purpose: IELTS question/answer image → editable text

1. Project Goal

IELTS Plus-এর জন্য একটি local OCR backend তৈরি করা হয়েছে, যেখানে:

Question Image / Answer Images
→ FastAPI
→ OpenCV preprocessing
→ PaddleOCR
→ Confidence checking
→ Text cleaning
→ Next.js Frontend-এ editable text

একটি request-এ:

1টি question image
সর্বোচ্চ 4টি answer image

process করা যায়।

2. Final Architecture
Next.js Frontend
       │
       │ POST /api/ocr/batch
       ▼
FastAPI Backend
       │
       ▼
Image Validation
       │
       ▼
OpenCV Preprocessing
       │
       ▼
PaddleOCR
       │
       ▼
Confidence / Result Selection
       │
       ▼
Text Cleaning
       │
       ▼
JSON Response
       │
       ▼
Next.js Editable Text
3. Python Version

শুরুতে Python 3.14 ব্যবহার করা হয়েছিল।

কিন্তু PaddleOCR dependency installation-এর সময় PyYAML build error পাওয়া যায়:

Failed building wheel for PyYAML
Microsoft Visual C++ 14.0 or greater is required

তাই OCR project-এর জন্য Python 3.11.5 ব্যবহার করা হয়েছে।

Python 3.14 PC-তে থাকতে পারে। এটি uninstall করার প্রয়োজন নেই।

Current OCR environment:

Python 3.11.5
4. Virtual Environment

Project-specific virtual environment:

py/
└── .venv/

Git Bash-এ activate করতে হবে:

source .venv/Scripts/activate

সঠিকভাবে activate হলে:

(.venv)

দেখাবে।

Verify:

python --version

Expected:

Python 3.11.5
5. Installed OCR Stack

বর্তমানে মূল packages:

Python         3.11.5
PaddlePaddle   3.3.1
PaddleOCR      3.7.0
FastAPI        Installed
Uvicorn        Installed
OpenCV         Installed
NumPy          Installed
python-dotenv  Installed

Paddle verify:

python -c "import paddle; print(paddle.__version__)"

Expected:

3.3.1

PaddleOCR verify:

python -c "from paddleocr import PaddleOCR; print('PaddleOCR OK')"
6. PaddleOCR Models

PaddleOCR প্রথমবার চালানোর সময় official models download করেছে।

Detection:

PP-OCRv6_medium_det

Recognition:

PP-OCRv6_medium_rec

Models local cache-এ রাখা হয়েছে:

C:\Users\PARVEZ\.paddlex\official_models\

পরবর্তীতে cached model ব্যবহার হওয়ায় আবার download করার প্রয়োজন হয় না।

7. Important Paddle Compatibility Fix

Paddle/PaddleX oneDNN compatibility issue এড়ানোর জন্য PaddleOCR import-এর আগে environment configuration রাখা হয়েছে।

import os

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

Important: এই settings PaddleOCR import করার আগে থাকতে হবে।

8. OpenCV Preprocessing

OCR-এর আগে image preprocessing করা হচ্ছে।

General flow:

Original Image
      ↓
Resize / Upscale
      ↓
Grayscale
      ↓
Contrast Enhancement
      ↓
Thresholding
      ↓
Denoising
      ↓
PaddleOCR

Possible preprocessing variants:

original
gray
enhanced
otsu
adaptive
denoise_otsu

উদ্দেশ্য হলো বিভিন্ন ধরনের image-এর জন্য OCR accuracy improve করা।

9. OCR Processing

PaddleOCR মূলত দুই ধাপে কাজ করে:

Image
  ↓
Text Detection
  ↓
Text Recognition
  ↓
Text + Confidence

তারপর backend text clean করে final result তৈরি করে।

10. Confidence

OCR result-এর সঙ্গে confidence value রাখা হচ্ছে।

Confidence হলো OCR engine-এর recognition certainty।

তবে:

95% confidence

এর অর্থ exactly 95% characters correct নয়।

Actual OCR accuracy জানতে ground-truth text-এর সঙ্গে result compare করতে হবে।

11. FastAPI Backend

Backend application:

main.py

FastAPI server:

http://localhost:8000

Health endpoint:

/health

OCR health:

/api/ocr/health

Main OCR endpoint:

POST /api/ocr/batch
12. Batch OCR Request

Endpoint:

POST /api/ocr/batch

Fields:

question → single image
answers  → multiple images

Maximum:

1 question
+
4 answers

Frontend থেকে FormData হিসেবে image পাঠানো হয়।

Important: answers field-এ empty string পাঠানো যাবে না।

ভুল:

-F "answers="

সঠিকভাবে answer না থাকলে field omit করতে হবে।

Question-only request হলে শুধু:

question=image

পাঠানো উচিত।

13. File Validation

Backend image validate করে:

Allowed formats
JPG
JPEG
PNG
WEBP
Maximum size
15 MB
Validation
Empty file → error
Unsupported file type → error
File too large → error
No image → error
More than 4 answers → error
14. Environment Configuration

.env:

FRONTEND_URL=http://localhost:3000
MAX_FILE_SIZE_MB=15
PADDLE_LANG=en
MIN_CONFIDENCE=0.35

Frontend:

http://localhost:3000

Backend:

http://localhost:8000

CORS configuration-এর কারণে frontend backend-এর API call করতে পারে।

15. Next.js Frontend

Frontend OCR page-এ:

Upload
   ↓
Question Image
   ↓
Up to 4 Answer Images
   ↓
Convert to Text
   ↓
Python OCR API
   ↓
Converted Text

Converted page-এ:

Question text editable
Answer text editable
Character count
Copy button
Start over
Error message
Progress indicator

রাখা হয়েছে।

Frontend API URL:

NEXT_PUBLIC_OCR_API_URL

না থাকলে default:

http://localhost:8000
16. Important Frontend Timeout Issue

OCR processing-এর সময় browser/frontend timeout হতে পারে।

যেমন:

OCR request timed out

এর অর্থ সাধারণত:

Image অনেক বড়
OCR processing অনেক সময় নিচ্ছে
একসঙ্গে অনেক image process হচ্ছে
preprocessing-এর multiple variants বেশি সময় নিচ্ছে
backend response আসার আগেই frontend request timeout করেছে

এটি necessarily server crash বোঝায় না।

Backend যদি পরে:

POST /api/ocr/batch 200 OK

দেয়, তাহলে request backend-এ successfully process হয়েছে।

কিন্তু frontend timeout হয়ে গেলে result UI-তে পৌঁছাতে নাও পারে।

17. oneDNN Error

এক পর্যায়ে এই error পাওয়া গিয়েছিল:

ConvertPirAttribute2RuntimeAttribute not support
[pir::ArrayAttribute<pir::DoubleAttribute>]

এটি Paddle/PaddleX এবং oneDNN runtime compatibility-related issue।

এর কারণে বিভিন্ন preprocessing variant:

original
gray
enhanced
otsu
adaptive
denoise_otsu

fail করছিল।

বর্তমান setup-এ:

os.environ["FLAGS_use_mkldnn"] = "0"

দিয়ে MKL-DNN/oneDNN disable করার চেষ্টা করা হয়েছে।

Server startup-এ:

[OCR] PaddleOCR loaded successfully.
[OCR] Device: CPU
[OCR] MKL-DNN: disabled

দেখালে configuration correctly applied হয়েছে।

18. Python Environment Problem

একটি গুরুত্বপূর্ণ issue হয়েছিল যেখানে terminal-এ:

(.venv)

দেখালেও python global Python 3.14 ব্যবহার করছিল।

Check করার জন্য:

which python

এবং:

python --version

ব্যবহার করা হয়েছে।

Correct environment-এ expected:

Python 3.11.5

এবং:

python -m pip show paddleocr

করলে .venv-এর site-packages location দেখাতে হবে।

19. Correct Server Startup

Project directory:

cd ~/Desktop/files/NR-ProjectsPractices/IELTS-Plus/py

Activate:

source .venv/Scripts/activate

Verify:

python --version

Expected:

Python 3.11.5

Then start:

python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
20. Successful Server Output

সফলভাবে start হলে এরকম output দেখা যাবে:

[OCR] Initializing PaddleOCR...
[OCR] PaddleOCR loaded successfully.
[OCR] Device: CPU
[OCR] MKL-DNN: disabled

INFO: Started server process
INFO: Waiting for application startup.
INFO: Application startup complete.

এটি backend running বোঝায়।

21. Swagger Testing

FastAPI automatically API documentation তৈরি করে।

Open:

http://localhost:8000/docs

তারপর:

POST /api/ocr/batch
       ↓
Try it out
       ↓
Upload image
       ↓
Execute
       ↓
Check JSON response

Swagger test সফল হলে backend basic level-এ working।

22. Expected Response

Successful OCR response সাধারণত:

{
  "success": true,
  "question": "Question text...",
  "question_confidence": 0.91,
  "answers": [
    "Answer text..."
  ],
  "answer_text": "Answer text...",
  "answer_confidence": 0.89,
  "total_answers": 1
}

Frontend এই response থেকে:

question
answer_text

নিয়ে editable text area-তে দেখায়।

23. Project Structure

Recommended structure:

IELTS-Plus/
│
├── Frontend/
│   └── ...
│
└── py/
    │
    ├── .venv/
    ├── main.py
    ├── ocr_engine.py
    ├── requirements.txt
    ├── .env
    └── .gitignore
24. Git Ignore

.gitignore-এ অন্তত:

__pycache__/
*.pyc
.venv/
.env
*.log
.vscode/
.idea/

রাখা উচিত।

বিশেষভাবে:

.venv/
.env

GitHub-এ commit করা উচিত নয়।

Current Status
Python 3.11.5          ✅
Virtual Environment    ✅
PaddlePaddle 3.3.1    ✅
PaddleOCR 3.7.0       ✅
OCR Models             ✅
OpenCV                 ✅
FastAPI                ✅
Uvicorn                ✅
CORS                   ✅
Image Validation       ✅
Batch OCR              ✅
Preprocessing          ✅
Confidence             ✅
Text Cleaning          ✅
Swagger                ✅
Local API Server       ✅
Next.js Integration    🔄
OCR Accuracy Testing   🔄
Important Lessons / Problems Solved
Problem 1 — Git Bash activation

Wrong:

.venv\Scripts\activate

Correct:

source .venv/Scripts/activate
Problem 2 — Python 3.14 dependency issue

Python 3.14 → PyYAML build problem.

Solution:

Python 3.11.5
Problem 3 — Wrong Python environment

Always verify:

which python
python --version
python -m pip show paddleocr
Problem 4 — Paddle oneDNN compatibility

Use before PaddleOCR import:

os.environ["FLAGS_use_mkldnn"] = "0"
Problem 5 — Empty answers field

Do not send:

answers=""

If there are no answers, omit the field.

Problem 6 — Frontend timeout

Large images / multiple OCR variants can take significant time.

Need to optimize:

Image size
Preprocessing count
OCR calls
Frontend timeout
Backend processing
Next Development Phase

Installation/setup phase is mostly complete.

Next focus should be:

1. Test real IELTS question images
        ↓
2. Test IELTS essay/answer images
        ↓
3. Measure OCR accuracy
        ↓
4. Fix OCR preprocessing
        ↓
5. Reduce processing time
        ↓
6. Fix frontend timeout
        ↓
7. Improve text formatting
        ↓
8. Preserve paragraph/layout
        ↓
9. Finalize Next.js integration
        ↓
10. Production deployment
Recommended OCR Testing Dataset

কমপক্ষে এই ধরনের image test করা উচিত:

1. Clean printed IELTS text
2. Mobile camera photo
3. Low-light image
4. Slightly tilted image
5. Small font
6. Large font
7. Long paragraph
8. IELTS Writing Task 1
9. IELTS Writing Task 2
10. Handwritten answer

প্রতিটি test-এর জন্য:

Original Image
      ↓
Expected / Correct Text
      ↓
OCR Output
      ↓
Manual Comparison
      ↓
Accuracy + Processing Time

record করা উচিত।

Final Summary

IELTS Plus-এর জন্য এখন একটি Python 3.11.5 ভিত্তিক local OCR backend তৈরি হয়েছে।

Core stack:

Python 3.11.5
      +
FastAPI
      +
OpenCV
      +
PaddlePaddle 3.3.1
      +
PaddleOCR 3.7.0

System-এর মূল কাজ:

IELTS Image
    ↓
FastAPI
    ↓
Validation
    ↓
OpenCV
    ↓
PaddleOCR
    ↓
Confidence
    ↓
Text Cleaning
    ↓
JSON
    ↓
Next.js
    ↓
Editable Text

Current priority: OCR accuracy + processing speed + frontend timeout fix.

Backend:

http://localhost:8000

Swagger:

http://localhost:8000/docs

Server command:

source .venv/Scripts/activate

python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

Overall status: OCR backend setup complete; এখন optimization এবং real-image testing phase চলছে।# ImGToText_Python
