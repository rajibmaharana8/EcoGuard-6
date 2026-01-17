# EcoGuard PRO: Dual-Perspective Illegal Dumping Detector

EcoGuard PRO is a state-of-the-art surveillance system designed to detect illegal dumping sites using a hybrid AI architecture. It seamlessly integrates Aerial (Satellite) and Ground (Drone/Land) perspectives to provide comprehensive environmental monitoring.

## 🚀 Key Features

*   **Dual-Perspective Mode**:
    *   **Satellite Mode**: Utilizes **ResNet50 + FPN** (Feature Pyramid Network) to detect large-scale illegal landfills from aerial imagery.
    *   **Ground Mode**: Deploys **YOLOv8** combined with a custom **Chaos Analysis** algorithm (texture/variance analysis) to identify trash piles at ground level.
*   **Hybrid Fusion Heatmaps**: Generates surgical precision heatmaps by fusing Neural Class Activation Maps (CAMs) with texture analysis signals.
*   **Geo-Tagging & Community Alerts**:
    *   Automatically logs detection coordinates.
    *   Triggers **Community Alerts** when a cluster of high-confidence reports (≥3) is detected in a specific zone, simulating notification to municipal authorities.
*   **Real-time Dashboard**: A modern, glassmorphism-inspired UI for live monitoring and instant analysis feedback.

## 📂 Project Structure

The project has been restructured for modularity:

*   **`backend/`**: Contains the FastAPI server and AI models.
    *   `main.py`: The core application server.
    *   `models/`: Stores the ResNet and YOLO model weights and architecture definitions.
    *   `utils/`: Helper scripts for image processing.
    *   `reports.db`: SQLite database for storing geo-tagged reports.
*   **`frontend/`**: Contains the client-side user interface.
    *   `index.html`: Main dashboard interface.
    *   `style.css`: Modern styling specifications.
    *   `script.js`: Frontend logic for API communication and UI updates.
*   **`venv/`**: Python virtual environment.

## 🛠️ Setup & Installation

### 1. Prerequisites
Ensure you have Python 3.8+ installed.

### 2. Environment Setup
Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

Install dependencies (create a requirements.txt if not exists or install manually):
```bash
pip install fastapi uvicorn torch torchvision opencv-python pillow ultralytics numpy
```

### 3. Model Weights
To run the system, you need the pre-trained weights.
*   **Aerial Model**: Place `checkpoint.pth` (from [Google Drive Link](https://drive.google.com/drive/folders/1xy9BDFWWFkyaw3P8npEZxpTDFxkzA3NK?usp=sharing)) into `backend/models/aerial/`.
*   **Ground Model**: Ensure `taco_yolov8.pt` is present in `backend/models/ground/`.

### 4. Running the Application
Navigate to the `backend` directory and start the server:
```bash
cd backend
python main.py
```
*   The server will start at `http://0.0.0.0:8000`.
*   The API documentation is available at `http://localhost:8000/docs`.

### 5. Using the Dashboard
Once the backend is running, open your browser and visit:
```
http://localhost:8000
```
*   **Upload**: Drag and drop an image.
*   **Select Mode**: Choose between "Satellite" or "Land/Drone".
*   **Analyze**: View the prediction score, status, and generated heatmap.

## 📊 Technical Details

### Aerial Classifier (ResNet50+FPN)
*   **Backbone**: ResNet50 pre-trained on ImageNet.
*   **Architecture**: Feature Pyramid Network (FPN) for multi-scale detection.
*   **Input Size**: Resized to 800x800.
*   **Training**: Trained on the [AerialWaste](https://aerialwaste.org/) dataset.

### Ground Classifier (YOLOv8 + Chaos)
*   Computes a "Chaos Index" using Laplacian variance to distinguish unstructured trash from structured objects.
*   Fuses object detection confidence with texture chaos for robust ground-level detection.

## 📜 License
Creative Commons CC BY licensing scheme.

## 📚 Cite Us
```bibtex
@article{torres2023aerialwaste,
  title={AerialWaste dataset for landfill discovery in aerial and satellite images},
  author={Torres, Rocio Nahime and Fraternali, Piero},
  journal={Scientific Data},
  volume={10},
  number={1},
  pages={63},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```
Visit our site for more details: https://aerialwaste.org/
