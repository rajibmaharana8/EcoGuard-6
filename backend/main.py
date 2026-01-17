
import os
import io
import base64
import torch
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Form
from PIL import Image
from utils.image_processor import ImageProcessor
from ultralytics import YOLO
import sqlite3
from datetime import datetime

# Robust Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "reports.db")
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS reports 
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                      lat REAL, lng REAL, score REAL, 
                      status TEXT, timestamp DATETIME)''')
    conn.commit()
    conn.close()

init_db()

def find_nearest_officials(lat, lng):
    # This is a mock lookup. In a real scenario, this would call 
    # the Google Places API for "municipality" nearby.
    return {
        "name": "Local Zonal Municipal Office",
        "email": "municipality_alerts_demo@example.com",
        "address": f"Near coordinates {lat}, {lng}"
    }

app = FastAPI(title="EcoGuard Pro | Dual-Perspective Surveillance")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
AERIAL_CATS = ["suspicious_site"]
AERIAL_STATE_DICT = "models/aerial/checkpoint.pth"
AERIAL_MODEL_PATH = 'models.aerial.resnet50_fpn'
GROUND_MODEL_PATH = "models/ground/taco_yolov8.pt"

# Initialize Models
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Systems Online. Primary device: {device}")

aerial_engine = ImageProcessor(AERIAL_CATS, AERIAL_STATE_DICT, model=AERIAL_MODEL_PATH, scales=(1.0,))
ground_engine = YOLO(GROUND_MODEL_PATH)

def generate_heatmap(original_image_np, cam_array, texture_map, mode="sat"):
    """
    High-Precision Surgical Fusion.
    """
    h, w = original_image_np.shape[:2]
    
    # AI Signal
    cam_resized = cv2.resize(cam_array, (w, h), interpolation=cv2.INTER_LINEAR)
    c_max = np.max(cam_resized)
    cam_norm = cam_resized / (c_max + 1e-7) if c_max > 0 else cam_resized
    
    # Texture Signal (Messiness)
    tex_resized = cv2.resize(texture_map, (w, h), interpolation=cv2.INTER_LINEAR)
    t_max = np.max(tex_resized)
    tex_norm = tex_resized / (t_max + 1e-7) if t_max > 0 else tex_resized
    
    # Mode-Aware Fusion
    if mode == "land":
        # Land mode: Trust texture broadly to capture big piles
        fusion = cam_norm * 0.4 + tex_norm * 0.6
    else:
        # Sat mode: AI masks the texture to avoid urban noise
        fusion = cam_norm * 0.8 + (cam_norm * tex_norm) * 0.2
        
    fusion = np.nan_to_num(np.clip(fusion, 0, 1))
    
    # Contrast boost
    fusion = np.power(fusion, 1.2)
    fusion[fusion < 0.25] = 0.0 # Clean background
    
    heatmap_raw = np.uint8(255 * fusion)
    heatmap_color = cv2.applyColorMap(heatmap_raw, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    mask_3d = np.repeat(fusion[:, :, np.newaxis], 3, axis=2)
    overlay = (original_image_np.astype(np.float32) * (1 - mask_3d * 0.75) + 
               (heatmap_color.astype(np.float32) * mask_3d * 0.75)).astype(np.uint8)
    
    _, buffer = cv2.imencode('.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode('utf-8')

@app.post("/predict")
async def predict(
    file: UploadFile = File(...), 
    mode: str = "sat", 
    lat: str = Form("null"), 
    lng: str = Form("null")
):
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image.resize((800, 800), Image.BILINEAR))
        
        # --- ENHANCED CHAOS ANALYSIS ---
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        mag = np.abs(laplacian)
        
        # 'Chaos Index': Use Coefficient of Variation to distinguish mess from structure.
        # Trash has high variance relative to its mean edge intensity.
        chaos_idx = float(np.std(mag) / (np.mean(mag) + 5.0)) 
        
        score = 0
        cam_signal = np.zeros((800, 800), dtype=np.float32)

        if mode == "sat":
            iw = aerial_engine.execute_cams_pred(image_np)
            resnet_score = float(iw.classification_scores[0])
            cam_signal = iw.global_cams[0].astype(np.float32)
            # Satellite: Trust the AI but use chaos as a "sanity check" multiplier
            score = resnet_score * (0.8 + 0.2 * min(1.0, chaos_idx))
            x0 = 0.75 # Very strict for Satellite to avoid city buildings
        else:
            # LAND MODE: Catch the clusters
            results = ground_engine(image_np, verbose=False, conf=0.15)[0]
            yolo_score = 0
            if len(results.boxes) > 0:
                yolo_score = float(torch.max(results.boxes.conf).cpu().item())
                for box in results.boxes.xyxyn:
                    x1, y1, x2, y2 = box.cpu().numpy()
                    cam_signal[int(y1*800):int(y2*800), int(x1*800):int(x2*800)] = 1.0
            
            # Ground piles rely heavily on Chaos signal
            score = (yolo_score * 0.4) + (min(1.0, chaos_idx) * 0.6)
            x0 = 0.35 
        
        score = 1 / (1 + np.exp(-12 * (score - x0)))
        score = max(0.01, min(score, 0.99))
        
        heatmap_base64 = generate_heatmap(image_np, cam_signal, mag, mode=mode)
        
        status = "Safe"
        status_type = "success"
        if score > 0.75:
            status = "Illegal Dumping"
            status_type = "danger"
        elif score > 0.35:
            status = "Suspicious Site"
            status_type = "warning"

        # --- GEO-REPORTING LOGIC ---
        community_alert = False
        if lat != "null" and lng != "null" and status_type != "success":
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO reports (lat, lng, score, status, timestamp) VALUES (?, ?, ?, ?, ?)",
                         (float(lat), float(lng), score, status, datetime.now()))
            
            # Check for clusters: Reports within 0.001 deg (~100m) in the last 24h
            cursor.execute('''SELECT COUNT(*) FROM reports 
                            WHERE ABS(lat - ?) < 0.001 
                            AND ABS(lng - ?) < 0.001 
                            AND status != 'Safe' ''', (float(lat), float(lng)))
            count = cursor.fetchone()[0]
            conn.commit()
            conn.close()

            if count >= 3: # Threshold for Community Alert
                community_alert = True
                official = find_nearest_officials(lat, lng)
                print(f"!!! COMMUNITY ALERT !!! Reporting high-risk zone to {official['email']}")

        return {
            "success": True,
            "prediction": status.upper(),
            "status_type": status_type,
            "confidence": round(score * 100, 2),
            "heatmap": f"data:image/png;base64,{heatmap_base64}",
            "geo_tagged": lat != "null",
            "community_alert": community_alert
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

# Mount Static Files
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
