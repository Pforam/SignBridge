# SignBridge 🤟
### Bridging Communication Through Artificial Intelligence

SignBridge is a collaborative AI and computer vision project exploring **Indian Sign Language (ISL) recognition**. It uses hand-landmark information and machine learning/deep learning approaches to work with static alphabet gestures and sequential word-level signs.

The project brings together a frontend, backend recognition workflows, dataset-processing scripts, and model-training components.

> **Project status:** Academic development project. Recognition coverage and reliability depend on the available dataset, trained model files, and runtime configuration. SignBridge should not be treated as a complete or production-ready sign-language translator.

---

## Highlights

- **Static alphabet recognition:** Uses hand-landmark features with an XGBoost classification workflow.
- **Sequential word recognition:** Includes word-recognition workflows using temporal landmark sequences and deep-learning architectures.
- **Hand landmark extraction:** Uses MediaPipe Holistic to extract hand and relevant body landmarks from images and videos.
- **Dataset processing:** Includes scripts for static-image and video landmark extraction, dataset ingestion, and model training.
- **Frontend and backend:** Organizes the application into separate frontend and backend components.
- **Quality and project documentation:** Includes tests, architecture notes, project context, and QA-related materials.

## Technology Stack

| Technology | Role in the project |
|---|---|
| Python | Backend, data processing, and model workflows |
| MediaPipe Holistic | Hand and body landmark extraction |
| OpenCV | Image and video processing |
| NumPy | Numerical and feature operations |
| XGBoost | Static alphabet classification |
| PyTorch | Deep-learning model workflows |
| TensorFlow | Used in related deep-learning workflows |
| Scikit-learn | Machine-learning utilities |
| JavaScript | Frontend development |
| Vite | Frontend development and build tooling |

The exact packages and versions required to run the current project are defined by its dependency files and source imports.

## Recognition Approach

SignBridge handles static and dynamic signs as different recognition problems.

### Static alphabet workflow

1. Read an alphabet gesture image.
2. Extract hand landmarks.
3. Convert the landmarks into a numerical feature representation.
4. Pass the features to the alphabet classification model.
5. Return the predicted alphabet label.

The alphabet workflow uses **XGBoost** and hand landmark features.

### Sequential word workflow

1. Read a sign-language video.
2. Extract landmark information across video frames.
3. Prepare a temporal sequence for model input.
4. Pass the sequence to the configured word-recognition model.
5. Return the predicted word label when a valid prediction is produced.

The project includes **ST-GCN** and **CNN-LSTM** word-recognition workflows. The model selected at runtime depends on the backend configuration and available trained model files.

## Repository Structure

```text
SignBridge/
├── .agents/                 # Agent/project configuration
├── backend/                 # Backend services, extraction and training scripts
├── src/                     # Frontend source code
├── public/                  # Public frontend assets
├── tests/                   # Project tests
├── qa_analysis/             # QA and analysis materials
├── k8s/                     # Kubernetes configuration files
├── graphify-out/            # Generated analysis output (if retained)
├── ARCHITECTURE.md          # Architecture documentation
├── PROJECT_CONTEXT.md       # Project context
├── AGENTS.md                # Agent instructions
├── AI_MEMORY.md             # Project working notes
├── package.json             # Frontend package configuration
├── package-lock.json        # Locked frontend dependencies
├── vite.config.js           # Vite configuration
├── index.html               # Frontend entry HTML
├── .gitignore
└── README.md

Private/local data (not included in the public repository):
├── dataset/
├── Dataset_Words/
├── landmark_output/
└── other local dataset and generated-data folders
```

The exact contents of `backend/` and other working directories may change as development continues. Private dataset folders are intentionally excluded from the public repository.

## Getting Started

### Prerequisites

- Git
- Node.js and npm
- Python version compatible with the project's backend dependencies
- A webcam for workflows that use live camera input
- Locally available trained model files and datasets for the recognition workflows you want to run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/SignBridge.git
cd SignBridge
```

### 2. Install frontend dependencies

```bash
npm install
```

### 3. Start the frontend

Check the scripts declared in `package.json`:

```bash
npm run
```

Run the appropriate development script shown in that list. In a typical Vite setup, this is:

```bash
npm run dev
```

### 4. Set up the backend

Create and activate a virtual environment from the backend directory:

```bash
cd backend
python -m venv .venv
```

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**

```cmd
.venv\Scripts\activate.bat
```

Install dependencies using the backend dependency file provided in the repository. If no backend `requirements.txt` is present, create one from the actual backend imports and tested environment before attempting a fresh installation.

Follow the backend configuration and source-code entry points to start the API or recognition service. Startup commands can vary with the configured application entry point.

## Datasets and Model Files

**The training datasets and extracted landmark data are private and are not distributed in this public repository.**

The `.gitignore` is configured to exclude local dataset directories, extracted landmark outputs, generated experiments, and dataset archives. Anyone cloning the repository should not expect those files to be present.

To run training or recognition workflows that depend on these resources:

1. Obtain the datasets separately, with permission to use them.
2. Place them in the directory structure expected by the relevant scripts.
3. Obtain or train the required model files.
4. Check backend configuration for the expected model paths and input formats.

Do not commit private datasets, personal data, or model files that you do not have permission to redistribute.

## Model and Data Workflows

The backend includes workflows for:

- Static image landmark extraction
- Video landmark extraction
- Dataset ingestion
- ZIP-based dataset ingestion
- Unified model training
- Alphabet and word recognition services

Refer to the backend scripts and project configuration for the current arguments, paths, and model filenames.

## Limitations

- Recognition quality depends on dataset size, balance, diversity, and annotation quality.
- Camera angle, lighting, signing speed, hand orientation, and occlusion can affect landmark extraction.
- Missing or noisy landmarks can reduce prediction reliability.
- Word recognition requires consistent temporal sequence preparation.
- The supported vocabulary is limited to the labels represented in the available training data.
- Real-world performance requires broader testing across signers and environments.

## Future Scope

- Expand and diversify the alphabet and word datasets.
- Improve robustness to hand orientation, occlusion, and lighting changes.
- Improve temporal sequence handling and prediction stability.
- Expand vocabulary and explore continuous sign recognition.
- Evaluate the system with more signers and varied recording conditions.
- Explore additional accessibility features such as text-to-speech output.

## Contributors

Developed collaboratively by:

- **Foram Patel**
- **Stuti**
- **Chintan**
- **Parth**

## Acknowledgements

SignBridge builds on open-source tools and research in computer vision and machine learning, including MediaPipe, OpenCV, PyTorch, TensorFlow, Scikit-learn, and XGBoost.

---

**SignBridge — Bridging Communication, One Gesture at a Time.** 🤟
