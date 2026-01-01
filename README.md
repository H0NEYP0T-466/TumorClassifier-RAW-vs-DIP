# TumorClassifier-RAW-vs-DIP

<p align="center">

  <!-- Demo -->
  [![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Visit%20Now-success?style=for-the-badge)](https://tumor-classifier-raw-vs-dip.vercel.app)

  <!-- Core -->
  ![GitHub License](https://img.shields.io/github/license/H0NEYP0T-466/TumorClassifier-RAW-vs-DIP?style=for-the-badge&color=brightgreen)  
  ![GitHub Stars](https://img.shields.io/github/stars/H0NEYP0T-466/TumorClassifier-RAW-vs-DIP?style=for-the-badge&color=yellow)  
  ![GitHub Forks](https://img.shields.io/github/forks/H0NEYP0T-466/TumorClassifier-RAW-vs-DIP?style=for-the-badge&color=blue)  
  ![GitHub Issues](https://img.shields.io/github/issues/H0NEYP0T-466/TumorClassifier-RAW-vs-DIP?style=for-the-badge&color=red)  
  ![GitHub Pull Requests](https://img.shields.io/github/issues-pr/H0NEYP0T-466/TumorClassifier-RAW-vs-DIP?style=for-the-badge&color=orange)  
  ![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen?style=for-the-badge)  

  <!-- Activity -->
  ![Last Commit](https://img.shields.io/github/last-commit/H0NEYP0T-466/TumorClassifier-RAW-vs-DIP?style=for-the-badge&color=purple)  
  ![Commit Activity](https://img.shields.io/github/commit-activity/m/H0NEYP0T-466/TumorClassifier-RAW-vs-DIP?style=for-the-badge&color=teal)  
  ![Repo Size](https://img.shields.io/github/repo-size/H0NEYP0T-466/TumorClassifier-RAW-vs-DIP?style=for-the-badge&color=blueviolet)  
  ![Code Size](https://img.shields.io/github/languages/code-size/H0NEYP0T-466/TumorClassifier-RAW-vs-DIP?style=for-the-badge&color=indigo)  

  <!-- Languages -->
  ![Top Language](https://img.shields.io/github/languages/top/H0NEYP0T-466/TumorClassifier-RAW-vs-DIP?style=for-the-badge&color=critical)  
  ![Languages Count](https://img.shields.io/github/languages/count/H0NEYP0T-466/TumorClassifier-RAW-vs-DIP?style=for-the-badge&color=success)  

  <!-- Community -->
  ![Documentation](https://img.shields.io/badge/Docs-Available-green?style=for-the-badge&logo=readthedocs&logoColor=white)  
  ![Open Source Love](https://img.shields.io/badge/Open%20Source-%E2%9D%A4-red?style=for-the-badge)  

</p>

<p align="center">
  <strong>Brain Tumor Detection: Comparing RAW vs Digital Image Processing (DIP) Pipelines</strong>
</p>

<p align="center">
A comprehensive full-stack machine learning application that compares the effectiveness of RAW and DIP preprocessing pipelines for brain tumor classification using MRI scans. Built with React, TypeScript, and Python FastAPI.
</p>

---

## 🔗 Quick Links

- [🚀 Live Demo](https://tumor-classifier-raw-vs-dip.vercel.app)
- [📖 Documentation](#-table-of-contents)
- [🐛 Report Bug](https://github.com/H0NEYP0T-466/TumorClassifier-RAW-vs-DIP/issues/new?template=bug_report.yml)
- [✨ Request Feature](https://github.com/H0NEYP0T-466/TumorClassifier-RAW-vs-DIP/issues/new?template=feature_request.yml)
- [🤝 Contributing](CONTRIBUTING.md)

---

## 📑 Table of Contents

- [🚀 Quick Start](#-quick-start)
- [📖 Abstract](#-abstract)
- [✨ Key Highlights](#-key-highlights)
- [About the Project](#about-the-project)
- [🏗️ Architecture](#-architecture)
- [✨ Features](#-features)
- [🛠 Tech Stack](#-tech-stack)
- [📦 Dependencies & Packages](#-dependencies--packages)
- [🚀 Installation](#-installation)
- [⚡ Usage](#-usage)
- [🤖 Model Training & Performance](#-model-training--performance)
- [🚀 Deployment](#-deployment)
- [📸 Screenshots & Visualizations](#-screenshots--visualizations)
- [🔬 Research & Clinical Context](#-research--clinical-context)
- [📂 Folder Structure](#-folder-structure)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)
- [🛡 Security](#-security)
- [📏 Code of Conduct](#-code-of-conduct)

---

## 🚀 Quick Start

### For Users (No Installation)

**Try the live demo**: [https://tumor-classifier-raw-vs-dip.vercel.app](https://tumor-classifier-raw-vs-dip.vercel.app)

1. Open the web application
2. Upload a brain MRI scan (JPG/PNG)
3. Click "Analyze" to get instant predictions
4. View side-by-side comparison of RAW vs DIP pipelines

### For Developers (3 Steps)

```bash
# 1. Clone and install
git clone https://github.com/H0NEYP0T-466/TumorClassifier-RAW-vs-DIP.git
cd TumorClassifier-RAW-vs-DIP
npm install

# 2. Start backend
cd backend
pip install -r requirements.txt
python -m App.main

# 3. Start frontend (new terminal)
cd ..
npm run dev
# Open http://localhost:5173
```

---

## 📖 Abstract

**TumorClassifier-RAW-vs-DIP** is an advanced medical imaging platform that investigates the impact of preprocessing on brain tumor classification accuracy. This research-oriented application compares two distinct pipelines:

**RAW Pipeline:**
- Minimal preprocessing (grayscale conversion only)
- Direct feature extraction from MRI scans
- Linear SVM classification
- Baseline performance metrics

**DIP (Digital Image Processing) Pipeline:**
- Advanced preprocessing: CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Gaussian blur for noise reduction
- Morphological operations for feature enhancement
- PCA dimensionality reduction
- Linear SVM classification with optimized features

### Research Question
**Does sophisticated image preprocessing improve brain tumor classification accuracy, or is raw data sufficient?**

The platform provides a side-by-side comparison enabling researchers and practitioners to evaluate the cost-benefit tradeoff between computational complexity and classification performance.

### Clinical Relevance
- **Early Detection**: Fast, automated screening of brain MRI scans
- **Decision Support**: Assists radiologists with preliminary tumor detection
- **Accessibility**: Web-based interface requires no specialized software
- **Transparency**: Visual comparison of preprocessing steps builds trust

⚠️ **Important**: This is a research/educational tool. Clinical use requires regulatory approval and medical expert validation.

---

## ✨ Key Highlights

- 🔬 **Dual-Pipeline Architecture** - Compare RAW vs DIP preprocessing side-by-side
- 🧠 **Brain Tumor Classification** - Binary classification (tumor vs no tumor)
- ⚡ **Linear SVM Classifier** - Fast, interpretable machine learning model
- 📊 **Real-time Predictions** - Upload MRI, get instant results
- 🖼️ **Preprocessing Visualization** - Step-by-step DIP pipeline visualization
- 📈 **Performance Metrics** - Accuracy, precision, recall, F1-score comparisons
- 🌐 **Modern Web Interface** - React 19 + TypeScript + TailwindCSS v4
- 🚀 **Production-Ready API** - FastAPI backend with OpenAPI documentation
- 🔧 **Modular Design** - Easy to extend with new preprocessing techniques
- 📦 **Deployable** - Vercel-ready frontend, scalable backend

---

## About the Project

**TumorClassifier-RAW-vs-DIP** is an advanced medical imaging application designed to classify brain MRI scans as either containing a tumor or being tumor-free. The project implements and compares two distinct preprocessing approaches:

1. **RAW Pipeline**: Minimal preprocessing - direct classification on grayscale MRI images
2. **DIP Pipeline**: Advanced Digital Image Processing with CLAHE, Gaussian blur, and morphological operations

The application features a modern React frontend with real-time prediction capabilities and a high-performance FastAPI backend powered by Support Vector Machine (SVM) classifiers.

### 🎯 Key Objectives

- Compare effectiveness of RAW vs DIP preprocessing for medical image classification
- Provide an intuitive web interface for real-time brain tumor detection
- Demonstrate production-ready ML deployment with FastAPI
- Visualize preprocessing steps and model predictions side-by-side

---

## 🏗️ Architecture

### System Overview

```
┌──────────────────────────────────────────────────────────┐
│              Frontend (React + TypeScript)               │
│  ┌────────────┐ ┌────────────┐ ┌───────────────────┐    │
│  │  Landing   │ │   Upload   │ │    Results        │    │
│  │    Page    │ │   Image    │ │  Comparison       │    │
│  └────────────┘ └────────────┘ └───────────────────┘    │
│        │              │                  │               │
│        └──────────────┴──────────────────┘               │
│                       │                                  │
│              ┌────────▼────────┐                         │
│              │   API Client     │                         │
│              └────────┬────────┘                         │
└───────────────────────┼──────────────────────────────────┘
                        │ HTTP/REST
┌───────────────────────▼──────────────────────────────────┐
│              Backend (FastAPI + Python)                  │
│  ┌─────────────────────────────────────────────────┐     │
│  │           API Routes (main.py)                  │     │
│  │  • /health - Health check                       │     │
│  │  • /api/v1/predict - Single model prediction    │     │
│  │  • /api/v1/predict/compare - Dual comparison   │     │
│  └──────────────────┬──────────────────────────────┘     │
│                     │                                    │
│  ┌──────────────────▼──────────────────────────────┐     │
│  │         Preprocessing Pipeline                  │     │
│  │  ┌──────────┐        ┌──────────────┐          │     │
│  │  │   RAW    │        │     DIP      │          │     │
│  │  │ Pipeline │        │   Pipeline   │          │     │
│  │  └──────────┘        └──────────────┘          │     │
│  │      ↓                       ↓                  │     │
│  │  Grayscale        CLAHE + Gaussian +           │     │
│  │   Only           Morphology + PCA              │     │
│  └──────────────────┬──────────────────────────────┘     │
│                     │                                    │
│  ┌──────────────────▼──────────────────────────────┐     │
│  │            SVM Classifiers                      │     │
│  │  ┌──────────────┐  ┌──────────────┐            │     │
│  │  │  RAW SVM     │  │   DIP SVM    │            │     │
│  │  │   Model      │  │    Model     │            │     │
│  │  └──────────────┘  └──────────────┘            │     │
│  └──────────────────┬──────────────────────────────┘     │
│                     │                                    │
│  ┌──────────────────▼──────────────────────────────┐     │
│  │          Prediction Results                     │     │
│  │  • Class: Tumor / No Tumor                      │     │
│  │  • Confidence Score                             │     │
│  │  • Preprocessing Steps (DIP)                    │     │
│  │  • Performance Metrics                          │     │
│  └─────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Image Upload**: User uploads brain MRI scan (JPG/PNG)
2. **Preprocessing**:
   - **RAW**: Convert to grayscale → Extract features → Predict
   - **DIP**: Grayscale → CLAHE → Gaussian blur → Morphology → PCA → Predict
3. **Classification**: Both SVMs predict tumor presence independently
4. **Results**: Side-by-side comparison with confidence scores and preprocessing visualization

### Pipeline Comparison

| Stage | RAW Pipeline | DIP Pipeline |
|-------|--------------|--------------|
| **1. Grayscale** | ✅ Convert to grayscale | ✅ Convert to grayscale |
| **2. CLAHE** | ❌ Skipped | ✅ Contrast enhancement |
| **3. Gaussian Blur** | ❌ Skipped | ✅ Noise reduction |
| **4. Morphology** | ❌ Skipped | ✅ Feature enhancement |
| **5. PCA** | ❌ Skipped | ✅ Dimensionality reduction |
| **6. Classification** | ✅ Linear SVM | ✅ Linear SVM |
| **Complexity** | 🟢 Low | 🟡 Medium |
| **Speed** | 🟢 Fast (~50ms) | 🟡 Moderate (~150ms) |
| **Accuracy** | 📊 Baseline | 📊 Enhanced |

---

## ✨ Features

### 🔬 Core Functionality
- **Dual-Model Architecture**: Compare predictions from both RAW and DIP pipelines simultaneously
- **Real-Time Prediction**: Upload MRI scans and get instant tumor classification results
- **Preprocessing Visualization**: View step-by-step image transformations in the DIP pipeline
- **Model Performance Metrics**: Access detailed accuracy, precision, and recall statistics

### 💻 Frontend
- Modern, responsive UI built with React 19 and TypeScript
- TailwindCSS v4 for sleek, professional styling
- Real-time image upload and preview
- Side-by-side model comparison view
- Interactive preprocessing step visualization

### 🔧 Backend
- RESTful API built with FastAPI
- Dual SVM model architecture (RAW + DIP)
- Advanced image preprocessing pipeline (CLAHE, Gaussian blur, morphological operations)
- PCA dimensionality reduction for optimal performance
- Comprehensive logging and error handling
- CORS-enabled for seamless frontend integration

### 🛡 Production-Ready
- Type-safe TypeScript implementation
- Comprehensive error handling and validation
- Health check and status monitoring endpoints
- Structured logging for debugging and monitoring
- Scalable architecture for future enhancements

---

## 🛠 Tech Stack

### Languages
![TypeScript](https://img.shields.io/badge/TypeScript-%23007ACC.svg?style=for-the-badge&logo=typescript&logoColor=white)
![Python](https://img.shields.io/badge/Python-%233776AB.svg?style=for-the-badge&logo=python&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-%23F7DF1E.svg?style=for-the-badge&logo=javascript&logoColor=black)

### Frameworks & Libraries
![React](https://img.shields.io/badge/React-%2361DAFB.svg?style=for-the-badge&logo=react&logoColor=black)
![Vite](https://img.shields.io/badge/Vite-%23646CFF.svg?style=for-the-badge&logo=vite&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/TailwindCSS-%2306B6D4.svg?style=for-the-badge&logo=tailwind-css&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-%23009688.svg?style=for-the-badge&logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-%235C3EE8.svg?style=for-the-badge&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

### DevOps / CI / Tools
![ESLint](https://img.shields.io/badge/ESLint-%234B32C3.svg?style=for-the-badge&logo=eslint&logoColor=white)
![Uvicorn](https://img.shields.io/badge/Uvicorn-%23499848.svg?style=for-the-badge&logo=gunicorn&logoColor=white)
![Git](https://img.shields.io/badge/Git-%23F05032.svg?style=for-the-badge&logo=git&logoColor=white)

---

## 📦 Dependencies & Packages

### Frontend Dependencies

<details>
<summary><strong>Runtime Dependencies</strong></summary>

[![@tailwindcss/vite](https://img.shields.io/npm/v/@tailwindcss/vite?style=for-the-badge&label=%40tailwindcss%2Fvite&color=06B6D4)](https://www.npmjs.com/package/@tailwindcss/vite) - Vite plugin for TailwindCSS v4  
[![lucide-react](https://img.shields.io/npm/v/lucide-react?style=for-the-badge&label=lucide-react&color=F56565)](https://www.npmjs.com/package/lucide-react) - Beautiful & consistent icon toolkit  
[![react](https://img.shields.io/npm/v/react?style=for-the-badge&label=react&color=61DAFB)](https://www.npmjs.com/package/react) - JavaScript library for building user interfaces  
[![react-dom](https://img.shields.io/npm/v/react-dom?style=for-the-badge&label=react-dom&color=61DAFB)](https://www.npmjs.com/package/react-dom) - React package for working with the DOM  
[![tailwindcss](https://img.shields.io/npm/v/tailwindcss?style=for-the-badge&label=tailwindcss&color=06B6D4)](https://www.npmjs.com/package/tailwindcss) - Utility-first CSS framework  

</details>

<details>
<summary><strong>Dev Dependencies</strong></summary>

[![@eslint/js](https://img.shields.io/npm/v/@eslint/js?style=for-the-badge&label=%40eslint%2Fjs&color=4B32C3)](https://www.npmjs.com/package/@eslint/js) - ESLint JavaScript rules  
[![@types/node](https://img.shields.io/npm/v/@types/node?style=for-the-badge&label=%40types%2Fnode&color=339933)](https://www.npmjs.com/package/@types/node) - TypeScript definitions for Node.js  
[![@types/react](https://img.shields.io/npm/v/@types/react?style=for-the-badge&label=%40types%2Freact&color=61DAFB)](https://www.npmjs.com/package/@types/react) - TypeScript definitions for React  
[![@types/react-dom](https://img.shields.io/npm/v/@types/react-dom?style=for-the-badge&label=%40types%2Freact-dom&color=61DAFB)](https://www.npmjs.com/package/@types/react-dom) - TypeScript definitions for React DOM  
[![@vitejs/plugin-react](https://img.shields.io/npm/v/@vitejs/plugin-react?style=for-the-badge&label=%40vitejs%2Fplugin-react&color=646CFF)](https://www.npmjs.com/package/@vitejs/plugin-react) - Official React plugin for Vite  
[![eslint](https://img.shields.io/npm/v/eslint?style=for-the-badge&label=eslint&color=4B32C3)](https://www.npmjs.com/package/eslint) - Pluggable linting utility for JavaScript and TypeScript  
[![eslint-plugin-react-hooks](https://img.shields.io/npm/v/eslint-plugin-react-hooks?style=for-the-badge&label=eslint-plugin-react-hooks&color=61DAFB)](https://www.npmjs.com/package/eslint-plugin-react-hooks) - ESLint rules for React Hooks  
[![eslint-plugin-react-refresh](https://img.shields.io/npm/v/eslint-plugin-react-refresh?style=for-the-badge&label=eslint-plugin-react-refresh&color=61DAFB)](https://www.npmjs.com/package/eslint-plugin-react-refresh) - ESLint plugin for React Fast Refresh  
[![globals](https://img.shields.io/npm/v/globals?style=for-the-badge&label=globals&color=CB3837)](https://www.npmjs.com/package/globals) - Global identifiers from different JavaScript environments  
[![typescript](https://img.shields.io/npm/v/typescript?style=for-the-badge&label=typescript&color=007ACC)](https://www.npmjs.com/package/typescript) - TypeScript language and compiler  
[![typescript-eslint](https://img.shields.io/npm/v/typescript-eslint?style=for-the-badge&label=typescript-eslint&color=007ACC)](https://www.npmjs.com/package/typescript-eslint) - Tooling for TypeScript with ESLint  
[![vite](https://img.shields.io/npm/v/vite?style=for-the-badge&label=vite&color=646CFF)](https://www.npmjs.com/package/vite) - Next generation frontend build tool  

</details>

### Backend Dependencies

<details>
<summary><strong>Runtime Dependencies</strong></summary>

[![numpy](https://img.shields.io/pypi/v/numpy?style=for-the-badge&label=numpy&color=013243)](https://pypi.org/project/numpy/) - Fundamental package for scientific computing  
[![scikit-learn](https://img.shields.io/pypi/v/scikit-learn?style=for-the-badge&label=scikit-learn&color=F7931E)](https://pypi.org/project/scikit-learn/) - Machine learning library for Python  
[![joblib](https://img.shields.io/pypi/v/joblib?style=for-the-badge&label=joblib&color=3776AB)](https://pypi.org/project/joblib/) - Lightweight pipelining and caching for Python  
[![opencv-python](https://img.shields.io/pypi/v/opencv-python?style=for-the-badge&label=opencv-python&color=5C3EE8)](https://pypi.org/project/opencv-python/) - Computer vision and image processing library  
[![fastapi](https://img.shields.io/pypi/v/fastapi?style=for-the-badge&label=fastapi&color=009688)](https://pypi.org/project/fastapi/) - Modern, high-performance web framework for APIs  
[![uvicorn](https://img.shields.io/pypi/v/uvicorn?style=for-the-badge&label=uvicorn&color=499848)](https://pypi.org/project/uvicorn/) - Lightning-fast ASGI server  
[![python-multipart](https://img.shields.io/pypi/v/python-multipart?style=for-the-badge&label=python-multipart&color=3776AB)](https://pypi.org/project/python-multipart/) - Multipart form data parser  
[![loguru](https://img.shields.io/pypi/v/loguru?style=for-the-badge&label=loguru&color=3776AB)](https://pypi.org/project/loguru/) - Python logging made simple  
[![typing-extensions](https://img.shields.io/pypi/v/typing-extensions?style=for-the-badge&label=typing-extensions&color=3776AB)](https://pypi.org/project/typing-extensions/) - Backported and experimental type hints  

</details>

---

## 🚀 Installation

### Prerequisites

Before you begin, ensure you have the following installed:

- **Node.js** (v18.0.0 or higher) - [Download](https://nodejs.org/)
- **Python** (v3.8 or higher) - [Download](https://www.python.org/downloads/)
- **npm** or **yarn** - Comes with Node.js
- **Git** - [Download](https://git-scm.com/downloads)

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/H0NEYP0T-466/TumorClassifier-RAW-vs-DIP.git
   cd TumorClassifier-RAW-vs-DIP
   ```

2. **Install Frontend Dependencies**
   ```bash
   npm install
   # or
   yarn install
   ```

3. **Install Backend Dependencies**
   ```bash
   cd backend
   pip install -r requirements.txt
   # or using a virtual environment (recommended)
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. **Verify Model Files**
   
   Ensure the pre-trained models exist in the `models/` directory:
   - `raw_svm_model.pkl`
   - `dip_svm_model.pkl`
   - `raw_metrics.pkl`
   - `dip_metrics.pkl`

   If models are missing, you'll need to train them first (see Usage section).

---

## ⚡ Usage

### Running the Application

#### 1. Start the Backend Server

```bash
cd backend
python -m App.main

# Or using uvicorn directly:
uvicorn App.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: `http://localhost:8000`

**API Documentation:**
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

#### 2. Start the Frontend Development Server

In a new terminal:

```bash
npm run dev
# or
yarn dev
```

The application will be available at: `http://localhost:5173`

### API Endpoints

#### Health Check
```bash
GET /health
```

#### Predict (Single Model - RAW)
```bash
POST /api/v1/predict
Content-Type: multipart/form-data

file: <MRI_image.jpg>
```

#### Compare Predictions (RAW vs DIP)
```bash
POST /api/v1/predict/compare
Content-Type: multipart/form-data

file: <MRI_image.jpg>
```

### Building for Production

#### Frontend
```bash
npm run build
# Output will be in the 'dist' directory
```

#### Backend
```bash
# The backend can be deployed with Uvicorn
uvicorn App.main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## 🤖 Model Training & Performance

### Training Details

**Dataset:**
- Source: Brain MRI dataset (tumor/no tumor classification)
- Training split: 80% training, 20% testing
- Image format: Grayscale MRI scans
- Image size: Standardized to 128x128 pixels

**RAW Pipeline Training:**
```bash
cd backend/Model
python RawDataModel.py
```
- Preprocessing: Grayscale conversion only
- Feature extraction: Flattened pixel values
- Classifier: Linear SVM (sklearn.svm.LinearSVC)
- Hyperparameters: Default LinearSVC parameters
- Training time: ~2-5 minutes (CPU)

**DIP Pipeline Training:**
```bash
cd backend/Model
python DipDataModel.py
```
- Preprocessing: CLAHE → Gaussian → Morphology
- Feature extraction: PCA-reduced features
- Classifier: Linear SVM (sklearn.svm.LinearSVC)
- PCA components: Optimized for 95% variance retention
- Training time: ~5-10 minutes (CPU)

### Performance Metrics

Performance metrics are saved to `models/` directory:
- `raw_metrics.pkl` - RAW pipeline performance
- `dip_metrics.pkl` - DIP pipeline performance

**Typical Results:**
| Metric | RAW Pipeline | DIP Pipeline | Improvement |
|--------|-------------|--------------|-------------|
| Accuracy | ~85-90% | ~90-95% | +5-10% |
| Precision | ~82-88% | ~88-93% | +6-8% |
| Recall | ~80-87% | ~87-92% | +7-10% |
| F1-Score | ~81-87% | ~87-92% | +6-8% |

*Note: Exact metrics depend on dataset and train/test split. Run training scripts to generate current metrics.*

### Model Artifacts

Pre-trained models are stored in `models/` directory:
- `raw_svm_model.pkl` - RAW pipeline SVM (serialized with joblib)
- `dip_svm_model.pkl` - DIP pipeline SVM (serialized with joblib)
- `dip_trainingLOGS.txt` - Detailed training logs

**Model Size:**
- RAW SVM: ~5-10 MB
- DIP SVM: ~3-8 MB (smaller due to PCA reduction)

### Preprocessing Steps (DIP Pipeline)

1. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**
   - Enhances local contrast
   - Clip limit: 2.0
   - Tile grid size: 8x8

2. **Gaussian Blur**
   - Reduces noise and smooths images
   - Kernel size: 5x5
   - Sigma: 1.0

3. **Morphological Operations**
   - Opening: Removes small noise
   - Closing: Fills small gaps
   - Kernel: 3x3 ellipse

4. **PCA (Principal Component Analysis)**
   - Reduces dimensionality
   - Retains 95% variance
   - Speeds up training and inference

---

## 🚀 Deployment

### Live Demo
**Try it now**: [https://tumor-classifier-raw-vs-dip.vercel.app](https://tumor-classifier-raw-vs-dip.vercel.app)

### Frontend Deployment (Vercel)

```bash
# 1. Build the frontend
npm run build

# 2. Deploy to Vercel
npm install -g vercel
vercel

# 3. Configure environment variables in Vercel dashboard:
# VITE_API_URL = https://your-backend-api.com
```

**Vercel Configuration (vercel.json):**
```json
{
  "buildCommand": "npm run build",
  "outputDirectory": "dist",
  "framework": "vite"
}
```

### Backend Deployment (Railway/Render/Fly.io)

**Option 1: Railway**
```bash
# 1. Install Railway CLI
npm install -g @railway/cli

# 2. Login and deploy
railway login
railway init
railway up
```

**Option 2: Render**
- Connect GitHub repository
- Set build command: `pip install -r requirements.txt`
- Set start command: `cd backend && uvicorn App.main:app --host 0.0.0.0 --port $PORT`
- Add environment variables

**Option 3: Docker Deployment**
```dockerfile
# Dockerfile (backend)
FROM python:3.10-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .
COPY models/ /app/models/

EXPOSE 8000

CMD ["uvicorn", "App.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t tumor-classifier-backend .
docker run -p 8000:8000 tumor-classifier-backend
```

### Environment Variables

**Frontend (.env):**
```bash
VITE_API_URL=http://localhost:8000  # Development
# VITE_API_URL=https://your-backend.com  # Production
```

**Backend (No .env needed):**
- FastAPI runs with default settings
- CORS configured for local development
- Update CORS origins in `main.py` for production

### Performance Optimization

**Frontend:**
- Use lazy loading for images
- Implement code splitting
- Enable Vite build optimizations
- Use CDN for static assets

**Backend:**
- Use gunicorn with multiple workers
- Implement response caching
- Optimize image preprocessing
- Use async file uploads

---


## 🔬 Research & Clinical Context

### Medical Background

**Brain Tumors:**
- Abnormal growths of cells in the brain
- Can be benign (non-cancerous) or malignant (cancerous)
- Early detection critical for treatment success
- MRI is the gold standard for brain imaging

**Challenges:**
- Manual MRI analysis is time-consuming
- Requires expert radiologist interpretation
- Inter-observer variability in diagnosis
- Need for fast, automated screening tools

### AI-Powered Solution

This platform addresses these challenges by:
1. **Automating** initial tumor detection screening
2. **Standardizing** image analysis with consistent preprocessing
3. **Comparing** different preprocessing approaches scientifically
4. **Providing** instant results for preliminary assessment

### Preprocessing Impact Study

**Research Question:**
> Does sophisticated digital image processing improve brain tumor classification accuracy compared to minimal (RAW) preprocessing?

**Hypothesis:**
Advanced preprocessing (CLAHE, Gaussian blur, morphological operations) will:
- ✅ Enhance tumor boundary visibility
- ✅ Reduce noise and artifacts
- ✅ Improve feature extraction quality
- ✅ Result in higher classification accuracy

**Expected Trade-offs:**
- ⏱️ Increased computational time
- 🔧 Additional preprocessing complexity
- 💻 Higher implementation maintenance

### Limitations & Disclaimers

⚠️ **This is a research and educational tool. It is NOT intended for clinical diagnosis.**

**Current Limitations:**
- Binary classification only (tumor vs no tumor)
- No tumor type classification (glioma, meningioma, etc.)
- No tumor size/location analysis
- No multi-modal MRI support (T1, T2, FLAIR)
- No 3D volumetric analysis
- Limited to specific dataset characteristics

**For Clinical Use:**
- Requires FDA/regulatory approval
- Needs validation on diverse patient populations
- Must be integrated into clinical workflow
- Requires radiologist oversight and final decision
- Needs continuous monitoring and updates

### Future Enhancements

Potential improvements:
- 🔹 Multi-class tumor type classification
- 🔹 Tumor segmentation and localization
- 🔹 3D volumetric analysis
- 🔹 Multi-modal MRI fusion
- 🔹 Deep learning models (CNN, ViT)
- 🔹 Explainable AI visualizations
- 🔹 DICOM format support
- 🔹 Batch processing for multiple scans

---

## 📂 Folder Structure

```
TumorClassifier-RAW-vs-DIP/
│
├── backend/                      # Python FastAPI backend
│   ├── App/
│   │   ├── Utils/
│   │   │   ├── preprocessing.py  # Image preprocessing utilities
│   │   │   ├── logger.py         # Logging configuration
│   │   │   └── __init__.py
│   │   ├── main.py               # FastAPI application entry point
│   │   └── __init__.py
│   ├── Model/
│   │   ├── RawDataModel.py       # RAW pipeline model training
│   │   ├── DipDataModel.py       # DIP pipeline model training
│   │   └── __init__.py
│   ├── requirements.txt          # Python dependencies
│   └── run_commands.txt          # Backend setup commands
│
├── models/                       # Trained ML models
│   ├── raw_svm_model.pkl         # RAW pipeline SVM model
│   ├── dip_svm_model.pkl         # DIP pipeline SVM model
│   ├── raw_metrics.pkl           # RAW model performance metrics
│   ├── dip_metrics.pkl           # DIP model performance metrics
│   └── dip_trainingLOGS.txt      # Training logs
│
├── src/                          # React frontend source
│   ├── components/
│   │   └── LandingPage.tsx       # Main UI component
│   ├── assets/                   # Static assets
│   ├── App.tsx                   # Root React component
│   ├── main.tsx                  # React entry point
│   ├── App.css                   # Global styles
│   └── index.css                 # Base styles
│
├── public/                       # Static public assets
│   └── vite.svg
│
├── .github/                      # GitHub configuration
│   ├── ISSUE_TEMPLATE/           # Issue templates
│   └── pull_request_template.md  # PR template
│
├── eslint.config.js              # ESLint configuration
├── tsconfig.json                 # TypeScript configuration (base)
├── tsconfig.app.json             # TypeScript config for app
├── tsconfig.node.json            # TypeScript config for Node
├── vite.config.ts                # Vite build configuration
├── index.html                    # HTML entry point
├── package.json                  # Node.js dependencies
├── package-lock.json             # Locked versions
├── README.md                     # Project documentation
├── LICENSE                       # MIT License
├── CONTRIBUTING.md               # Contribution guidelines
├── SECURITY.md                   # Security policy
└── CODE_OF_CONDUCT.md            # Code of conduct
```

---

## 🤝 Contributing

We welcome contributions from the community! Whether you're fixing bugs, adding new features, or improving documentation, your help is appreciated.

Please read our [Contributing Guidelines](CONTRIBUTING.md) to get started.

### Quick Start for Contributors

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🛡 Security

Security is a top priority for this project. If you discover a security vulnerability, please follow our responsible disclosure guidelines.

See [SECURITY.md](SECURITY.md) for more information on reporting vulnerabilities.

---

## 📏 Code of Conduct

We are committed to providing a welcoming and inspiring community for all. Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before participating.

---

<p align="center">Made with ❤️ by H0NEYP0T-466</p>
