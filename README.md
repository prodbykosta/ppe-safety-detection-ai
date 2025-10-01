# 🏗️ Workplace Safety Monitor - AI-Powered PPE Detection System

A sophisticated computer vision system that monitors Personal Protective Equipment (PPE) compliance in real-time using advanced YOLO models and intelligent tracking algorithms.

## 🎬 Demo

![PPE Detection Demo](output.gif)

*Real-time PPE detection in action - tracking workers and monitoring helmet & vest compliance with sequential ID assignment*

## 🎯 Overview

The Workplace Safety Monitor is designed to enhance workplace safety by automatically detecting people and their required safety equipment (helmets and safety vests) in video streams. The system uses state-of-the-art computer vision techniques to provide accurate, real-time monitoring with minimal false positives.

## ✨ Key Features

- **🔍 Real-time Detection**: Live person and PPE detection using YOLO models
- **🧠 Smart Association**: Intelligent PPE-to-person matching using anatomical region analysis
- **📹 Multi-Source Support**: Works with webcams, IP cameras, and video files
- **🎥 Video Recording**: Outputs annotated MP4 videos with detection results
- **📊 Temporal Smoothing**: Reduces false positives through frame-to-frame consistency
- **🎯 High Accuracy**: Advanced filtering and validation for reliable results
- **⚡ Real-time Performance**: Optimized for live video processing
- **🔧 Flexible Configuration**: Customizable thresholds and parameters

## 🚀 Use Cases

### 1. **Workplace Safety Monitoring**
Monitor construction sites, factories, and industrial facilities for PPE compliance in real-time.

### 2. **Access Control Security System** 🔐
**Deploy as a security checkpoint** to allow facility entry only for workers wearing proper safety equipment:
- Install cameras at facility entrances
- Automatically verify helmet and vest compliance
- Integrate with access control systems (gates, turnstiles)
- Log entry attempts and safety violations
- Send alerts for non-compliant personnel
- Generate compliance reports for safety managers

### 3. **Construction Site Oversight**
Continuous monitoring of construction workers to ensure safety protocol adherence.

### 4. **Industrial Facility Monitoring**
Monitor manufacturing plants, warehouses, and processing facilities for safety compliance.

### 5. **Safety Training & Education**
Use recorded footage to train workers and demonstrate proper PPE usage.

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- OpenCV 4.x
- NumPy
- (Optional) Ultralytics for advanced YOLO models

### Quick Setup
```bash
# Clone the repository
git clone <repository-url>
cd Safety_Gear_Detection_Camera

# Install dependencies
pip install opencv-python numpy
pip install ultralytics  # Optional, for advanced YOLO models

# Verify installation
python3 workplace_safety_monitor.py --help
```

## 📋 Requirements

Create a `requirements.txt` file:
```
opencv-python>=4.8.0
numpy>=1.21.0
ultralytics>=8.0.0
torch>=1.13.0
torchvision>=0.14.0
```

## 🎮 Usage

### Basic Video Processing
```bash
python3 workplace_safety_monitor.py \
    --source test_videos/construction_site.mp4 \
    --ppe-weights best.pt \
    --save-vis output/
```

### Live Camera Monitoring
```bash
# Built-in webcam
python3 workplace_safety_monitor.py \
    --source 0 \
    --ppe-weights best.pt \
    --save-vis live_output/

# External USB camera
python3 workplace_safety_monitor.py \
    --source 1 \
    --ppe-weights best.pt \
    --save-vis live_output/

# IP Camera (RTSP stream)
python3 workplace_safety_monitor.py \
    --source "rtsp://username:password@192.168.1.100:554/stream1" \
    --ppe-weights best.pt \
    --save-vis live_output/
```

### Security Access Control Setup
```bash
# For access control at facility entrance
python3 workplace_safety_monitor.py \
    --source "rtsp://entrance_camera_ip:554/stream" \
    --ppe-weights best.pt \
    --save-vis security_logs/ \
    --conf-helmet 0.80 \
    --conf-vest 0.75
```

### Model Evaluation
```bash
# Evaluate model performance on test dataset
python3 workplace_safety_monitor.py \
    --ppe-weights best.pt \
    --eval-root dataset/valid \
    --conf-helmet 0.65 \
    --conf-vest 0.70
```

## ⚙️ Configuration Options

### Input Sources
- `--source`: Video file, camera index (0,1,2...), or stream URL
- `--ppe-weights`: Ultralytics YOLO model file (.pt)
- `--ppe-onnx`: ONNX model file (alternative to .pt)
- `--person-weights`: Person detection model (default: auto-download)

### Detection Thresholds
- `--conf-helmet`: Helmet detection confidence (default: 0.65)
- `--conf-vest`: Vest detection confidence (default: 0.70)
- `--nms-iou`: Non-maximum suppression IoU threshold (default: 0.50)

### Tracking & Association
- `--track-iou`: IoU threshold for person tracking (default: 0.35)
- `--track-max-age`: Maximum frames to keep unmatched tracks (default: 20)
- `--head-iou-gate`: Helmet-to-head region overlap threshold (default: 0.10)
- `--torso-iou-gate`: Vest-to-torso region overlap threshold (default: 0.15)

### Output Options
- `--save-vis`: Directory to save annotated frames and video
- `--eval-root`: Dataset directory for evaluation mode

## 📁 Project Structure

```
Safety_Gear_Detection_Camera/
├── workplace_safety_monitor.py    # Main application
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── best.pt                       # Trained PPE detection model
├── yolo12n.pt                    # Person detection model
├── dataset/                      # Training/validation data
│   └── safety-Helmet-Reflective-Jacket/
├── test_videos/                  # Sample test videos
│   └── veo3_construction.mp4
└── output/                       # Generated outputs
    ├── frame_*.jpg               # Individual frames
    └── output.mp4                # Annotated video
```

## 🧠 How It Works

### 1. **Person Detection**
- Uses YOLO models for high-accuracy person detection
- Falls back to OpenCV HOG detector if YOLO unavailable
- Filters detections by size and aspect ratio

### 2. **PPE Detection**
- Separate YOLO model trained specifically for helmets and safety vests
- Supports both Ultralytics (.pt) and ONNX formats
- Configurable confidence thresholds per equipment type

### 3. **Intelligent Association**
- Maps PPE items to specific persons using anatomical regions
- Helmets must overlap with head region (upper 40% of person box)
- Vests must overlap with torso region (middle section)
- Ensures PPE is mostly contained within person boundaries

### 4. **Temporal Smoothing & Smart Tracking**
- Tracks persons across frames using IoU-based tracking
- Applies temporal filtering to reduce single-frame false positives
- Maintains detection history for stable results
- **Smart boundary validation**: Automatically removes tracks stuck at frame edges
- **Edge detection**: Identifies when people exit the frame and cleans up tracking boxes immediately

### 5. **Visualization & Output**
- Real-time display with color-coded bounding boxes
- Green: Compliant (has required PPE)
- Red: Non-compliant (missing PPE)
- Blue: PPE detection boxes
- Saves annotated MP4 video and individual frames

## 🔧 Advanced Configuration

### For Access Control Systems
```bash
# High-security configuration with strict thresholds
python3 workplace_safety_monitor.py \
    --source "rtsp://gate_camera:554/stream" \
    --ppe-weights best.pt \
    --conf-helmet 0.85 \
    --conf-vest 0.80 \
    --nms-iou 0.45 \
    --temporal-window 10 \
    --save-vis security_checkpoint/
```

### For Training/Validation
```bash
# Evaluate different threshold combinations
python3 workplace_safety_monitor.py \
    --ppe-weights models/helmet_vest_v2.pt \
    --eval-root dataset/validation \
    --conf-helmet 0.70 \
    --conf-vest 0.75
```

## 🚦 Integration with Access Control

To use as a security system for facility access:

1. **Camera Placement**: Install cameras at entry points
2. **Model Training**: Train on your specific PPE types and lighting conditions
3. **Threshold Tuning**: Use evaluation mode to optimize detection thresholds
4. **System Integration**: Connect to access control hardware (gates, alarms)
5. **Logging**: Implement database logging for audit trails
6. **Alerts**: Set up real-time notifications for violations

## 📊 Performance Optimization

### Hardware Recommendations
- **GPU**: NVIDIA GPU with CUDA support for faster inference
- **CPU**: Multi-core processor for parallel processing
- **RAM**: 8GB+ for processing high-resolution videos
- **Storage**: SSD for faster model loading and video I/O

### Software Optimization
```bash
# Enable GPU acceleration (if available)
export CUDA_VISIBLE_DEVICES=0

# Optimize for real-time processing
python3 workplace_safety_monitor.py \
    --source 0 \
    --ppe-weights best.pt \
    --input-size 640 \
    --save-vis output/
```

## 🛡️ Security Considerations

- Validate all input file paths to prevent directory traversal
- Implement proper authentication for IP camera access
- Secure model files to prevent tampering
- Log all detection events for audit purposes
- Consider privacy implications of video monitoring

## 🐛 Troubleshooting

### Common Issues

**"Provide --ppe-weights (.pt) or --ppe-onnx (.onnx)" Error**
```bash
# Solution: Specify PPE model file
python3 workplace_safety_monitor.py --ppe-weights best.pt --source 0
```

**Camera Not Opening**
```bash
# Try different camera indices
python3 workplace_safety_monitor.py --source 1 --ppe-weights best.pt
```

**Low Detection Accuracy**
```bash
# Adjust confidence thresholds
python3 workplace_safety_monitor.py \
    --ppe-weights best.pt \
    --conf-helmet 0.50 \
    --conf-vest 0.50 \
    --source video.mp4
```

## 📈 Future Enhancements

- [ ] Multi-camera support for comprehensive coverage
- [ ] Database integration for violation logging
- [ ] Web-based dashboard for remote monitoring
- [ ] Mobile app for notifications
- [ ] Additional PPE types (gloves, boots, masks)
- [ ] Integration with existing security systems
- [ ] Machine learning model auto-updates
- [ ] Analytics and reporting features

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Dataset**: [Construction Site Safety Image Dataset](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow) by Snehil Sanyal on Kaggle
- Ultralytics for YOLO implementation
- OpenCV community for computer vision tools
- Roboflow for dataset processing and annotation tools
- Contributors to the safety detection dataset

## 📞 Support

For questions, issues, or feature requests:
- Create an issue in the repository
- Contact contact@prodbykosta.com
- Check the troubleshooting section above

---

**⚠️ Safety Notice**: This system is designed to assist with safety monitoring but should not be the sole method of ensuring workplace safety compliance. Always follow proper safety protocols and regulations.