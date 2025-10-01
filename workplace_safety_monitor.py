#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Workplace Safety Monitor — AI-Powered PPE Detection System

A computer vision system for monitoring Personal Protective Equipment (PPE) compliance
in workplace environments. Detects people and their safety gear (helmets, vests) in
real-time video streams using YOLO models with advanced tracking and association algorithms.

Key Features:
- Real-time person and PPE detection
- Intelligent PPE-to-person association using anatomical regions
- Temporal smoothing to reduce false positives
- Support for live cameras and video files
- MP4 video output with annotated detections
- Evaluation mode for model performance testing

Use Cases:
- Workplace safety monitoring and compliance
- Access control systems (facility entry with proper PPE)
- Construction site safety oversight
- Industrial facility monitoring
- Safety training and education

Author: Safety Detection Team
License: MIT
"""

from __future__ import annotations
import os
import cv2
import time
import math
import glob
import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

# -------------------------------
# Optional Ultralytics
# -------------------------------
try:
    from ultralytics import YOLO
    _ULTRA_AVAILABLE = True
except Exception:
    _ULTRA_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ppe-monitor")

# -------------------------------
# Utility
# -------------------------------

@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    cls: int
    label: str

    def xyxy(self) -> Tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    def w(self) -> int:
        return max(0, self.x2 - self.x1)

    def h(self) -> int:
        return max(0, self.y2 - self.y1)

    def area(self) -> int:
        return self.w() * self.h()

def iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(1, (ax2 - ax1) * (ay2 - ay1))
    b_area = max(1, (bx2 - bx1) * (by2 - by1))
    union = a_area + b_area - inter
    return inter / union

def clip_box(b: Tuple[int, int, int, int], W: int, H: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = b
    return int(max(0, x1)), int(max(0, y1)), int(min(W - 1, x2)), int(min(H - 1, y2))

def is_track_valid(box: Box, W: int, H: int, edge_threshold: float = 0.5) -> bool:
    """
    Check if a track is valid (not stuck at frame boundaries).
    
    Args:
        box: Bounding box to check
        W: Frame width
        H: Frame height
        edge_threshold: Fraction of box that must be inside frame (0.5 = 50%)
    
    Returns:
        True if track is valid, False if it should be removed
    """
    x1, y1, x2, y2 = box.xyxy()
    
    # Calculate box dimensions
    box_w = x2 - x1
    box_h = y2 - y1
    box_area = box_w * box_h
    
    # If box is too small, consider it invalid
    if box_area < 100:  # Minimum 10x10 pixel box
        return False
    
    # More aggressive boundary checking - if box is mostly at edges, remove it
    margin = 10  # 10 pixel margin from edges
    
    # Check if box is stuck at frame edges
    if (x1 <= margin and x2 <= margin + 50) or \
       (x2 >= W - margin and x1 >= W - margin - 50) or \
       (y1 <= margin and y2 <= margin + 50) or \
       (y2 >= H - margin and y1 >= H - margin - 50):
        return False
    
    # Calculate area inside frame boundaries
    clipped_x1 = max(0, x1)
    clipped_y1 = max(0, y1)
    clipped_x2 = min(W, x2)
    clipped_y2 = min(H, y2)
    
    clipped_w = max(0, clipped_x2 - clipped_x1)
    clipped_h = max(0, clipped_y2 - clipped_y1)
    clipped_area = clipped_w * clipped_h
    
    # Check if enough of the box is inside the frame
    if box_area > 0:
        inside_ratio = clipped_area / box_area
        return inside_ratio >= edge_threshold
    
    return False

def box_center(b: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5

def inside_fraction(inner: Tuple[int,int,int,int], outer: Tuple[int,int,int,int]) -> float:
    # fraction of inner's area that lies inside outer
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    jx1, jy1 = max(ix1, ox1), max(iy1, oy1)
    jx2, jy2 = min(ix2, ox2), min(iy2, oy2)
    iw, ih = max(0, jx2-jx1), max(0, jy2-jy1)
    inter = iw * ih
    ia = max(1, (ix2-ix1)*(iy2-iy1))
    return inter / ia

# -------------------------------
# Person regions (head / torso)
# -------------------------------

def head_torso_regions(person: Box) -> Dict[str, Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = person.xyxy()
    w, h = x2 - x1, y2 - y1
    # conservative splits to avoid FP from surrounding clutter
    head_h = int(h * 0.26)
    torso_h = int(h * 0.52)
    head = (x1 + int(0.22*w), y1, x2 - int(0.22*w), y1 + head_h)
    torso = (x1 + int(0.10*w), y1 + head_h, x2 - int(0.10*w), min(y1 + head_h + torso_h, y2))
    return {"head": head, "torso": torso}

# -------------------------------
# Person detector
# -------------------------------

class PersonDetector:
    def __init__(self, weights: Optional[str] = None, conf: float = 0.45):
        self.conf = conf
        self.use_ultra = _ULTRA_AVAILABLE and weights is not None
        if self.use_ultra:
            try:
                self.model = YOLO(weights)
                dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                _ = self.model(dummy, conf=self.conf, verbose=False)
                logger.info("Person detector: Ultralytics")
            except Exception as e:
                logger.warning(f"Ultralytics person model failed: {e}. Falling back to HOG.")
                self.use_ultra = False
        if not self.use_ultra:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            logger.info("Person detector: OpenCV HOG fallback")

    def detect(self, frame: np.ndarray) -> List[Box]:
        H, W = frame.shape[:2]
        out: List[Box] = []
        if self.use_ultra:
            res = self.model(frame, conf=self.conf, verbose=False)
            for r in res:
                names = r.names if hasattr(r, 'names') else {}
                for b in r.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    c = float(b.conf[0])
                    cls = int(b.cls[0])
                    label = names.get(cls, str(cls))
                    if label == 'person':
                        x1, y1, x2, y2 = clip_box((x1, y1, x2, y2), W, H)
                        out.append(Box(x1, y1, x2, y2, c, cls, 'person'))
        else:
            rects, weights = self.hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
            for (x, y, w, h), c in zip(rects, weights):
                if w < 32 or h < 64:
                    continue
                ar = h / max(1e-6, w)
                if ar < 1.2:
                    continue
                x1, y1, x2, y2 = clip_box((x, y, x + w, y + h), W, H)
                out.append(Box(x1, y1, x2, y2, float(c), 0, 'person'))
        return out

# -------------------------------
# PPE detectors
# -------------------------------

class PPEUltralytics:
    def __init__(self, weights: str, conf_th: float, nms_iou: float, class_map: Dict[int, str]):
        if not _ULTRA_AVAILABLE:
            raise RuntimeError("Ultralytics not installed")
        self.model = YOLO(weights)
        self.conf_th = conf_th
        self.nms_iou = nms_iou
        self.class_map = class_map
        _ = self.model(np.zeros((640, 640, 3), dtype=np.uint8), conf=self.conf_th, iou=self.nms_iou, verbose=False)
        logger.info("PPE Ultralytics model loaded")

    def infer(self, frame: np.ndarray) -> List[Box]:
        H, W = frame.shape[:2]
        out: List[Box] = []
        res = self.model(frame, conf=self.conf_th, iou=self.nms_iou, verbose=False)
        for r in res:
            names = r.names if hasattr(r, 'names') else {}
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                c = float(b.conf[0])
                cls = int(b.cls[0])
                label = names.get(cls, self.class_map.get(cls, str(cls))).lower()
                if label == 'jacket':
                    label = 'vest'
                x1, y1, x2, y2 = clip_box((x1, y1, x2, y2), W, H)
                out.append(Box(x1, y1, x2, y2, c, cls, label))
        return out

class PPEOnnx:
    """Minimal YOLOv7 ONNX inference using OpenCV DNN. Adjust parser if your export differs."""
    def __init__(self, onnx_path: str, class_names: List[str], conf_th: float, nms_iou: float, input_size: int = 640):
        self.net = cv2.dnn.readNetFromONNX(onnx_path)
        self.class_names = [n.lower() for n in class_names]
        self.conf_th = conf_th
        self.nms_iou = nms_iou
        self.input_size = input_size
        logger.info(f"PPE ONNX model loaded: {onnx_path}")

    def infer(self, frame: np.ndarray) -> List[Box]:
        H, W = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (self.input_size, self.input_size), swapRB=True, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()
        if out.ndim == 3:
            out = np.squeeze(out, axis=0)
        if out.ndim == 2:
            preds = out
        else:
            preds = out.reshape(out.shape[1], -1)
        boxes: List[Box] = []
        for p in preds:
            cx, cy, w, h = p[0], p[1], p[2], p[3]
            obj = p[4]
            scores = p[5:]
            cls_id = int(np.argmax(scores))
            cls_score = scores[cls_id]
            conf = float(obj * cls_score)
            if conf < self.conf_th:
                continue
            label = self.class_names[cls_id]
            if label == 'jacket':
                label = 'vest'
            # scale back to image size (assumes letterbox-free; if letterboxed, adapt)
            x = (cx - w/2) * W / self.input_size
            y = (cy - h/2) * H / self.input_size
            ww = w * W / self.input_size
            hh = h * H / self.input_size
            x1, y1, x2, y2 = int(x), int(y), int(x + ww), int(y + hh)
            x1, y1, x2, y2 = clip_box((x1, y1, x2, y2), W, H)
            boxes.append(Box(x1, y1, x2, y2, conf, cls_id, label))
        # NMS
        if boxes:
            idxs = cv2.dnn.NMSBoxes(
                bboxes=[(b.x1, b.y1, b.w(), b.h()) for b in boxes],
                scores=[b.conf for b in boxes],
                score_threshold=self.conf_th,
                nms_threshold=self.nms_iou,
            )
            keep = set([i for i in (idxs.flatten().tolist() if len(idxs) else [])])
            boxes = [b for i, b in enumerate(boxes) if i in keep]
        return boxes

# -------------------------------
# Tracking (IoU, greedy)
# -------------------------------

@dataclass
class Track:
    tid: int
    box: Box
    last_seen: int
    hits: int = 1

class IoUTracker:
    def __init__(self, iou_th: float = 0.3, max_age: int = 20):
        self.iou_th = iou_th
        self.max_age = max_age
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1
        self.frame_idx = 0

    def update(self, detections: List[Box], frame_width: int = None, frame_height: int = None) -> Dict[int, Box]:
        self.frame_idx += 1
        assigned = set()
        # Try match existing tracks
        for tid, tr in list(self.tracks.items()):
            best_idx, best_i = -1, 0.0
            for i, d in enumerate(detections):
                if i in assigned:
                    continue
                ii = iou(tr.box.xyxy(), d.xyxy())
                if ii > best_i:
                    best_idx, best_i = i, ii
            if best_i >= self.iou_th and best_idx >= 0:
                tr.box = detections[best_idx]
                tr.last_seen = self.frame_idx
                tr.hits += 1
                assigned.add(best_idx)
            # Note: if not matched, we keep it until max_age
        # Create new tracks for unmatched detections (but validate them first)
        for i, d in enumerate(detections):
            if i in assigned:
                continue
            # Skip creating tracks for detections that would be immediately removed
            if frame_width is not None and frame_height is not None:
                if not is_track_valid(d, frame_width, frame_height, edge_threshold=0.3):
                    continue  # Don't waste an ID on an invalid detection
            
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = Track(tid=tid, box=d, last_seen=self.frame_idx, hits=1)
        
        # Prune old tracks and edge-stuck tracks
        for tid in list(self.tracks.keys()):
            tr = self.tracks[tid]
            should_remove = False
            frames_since_seen = self.frame_idx - tr.last_seen
            
            # Original age-based removal
            if frames_since_seen > self.max_age:
                should_remove = True
            
            # Enhanced: Remove tracks that are mostly outside frame boundaries
            elif frame_width is not None and frame_height is not None:
                if not is_track_valid(tr.box, frame_width, frame_height, edge_threshold=0.3):
                    should_remove = True
            
            if should_remove:
                del self.tracks[tid]
        
        # Return current track boxes
        return {tid: tr.box for tid, tr in self.tracks.items()}

# -------------------------------
# Association, filtering, smoothing
# -------------------------------

@dataclass
class AssocConfig:
    conf_helmet: float = 0.65
    conf_vest: float = 0.70
    nms_iou: float = 0.50
    min_box_area: int = 30 * 30
    max_aspect_ratio: float = 3.5       # filter skinny/wide boxes
    head_iou_gate: float = 0.10         # min overlap with head region
    torso_iou_gate: float = 0.15        # min overlap with torso region
    ppe_inside_person_frac: float = 0.35 # fraction of PPE area inside person box
    temporal_window: int = 7

class PPESmoother:
    def __init__(self, window: int = 7):
        self.window = window
        # history[track_id] = list of (has_helmet, has_vest), trimmed to window
        self.history: Dict[int, List[Tuple[bool, bool]]] = {}

    def update(self, track_ids: List[int], have_helmet: List[bool], have_vest: List[bool]) -> List[Tuple[bool, bool]]:
        smoothed = []
        for tid, h, v in zip(track_ids, have_helmet, have_vest):
            self.history.setdefault(tid, [])
            self.history[tid].append((h, v))
            if len(self.history[tid]) > self.window:
                self.history[tid] = self.history[tid][-self.window:]
            hh = [x[0] for x in self.history[tid]]
            vv = [x[1] for x in self.history[tid]]
            smoothed.append((sum(hh) >= (len(hh)+1)//2, sum(vv) >= (len(vv)+1)//2))
        return smoothed

# -------------------------------
# Visualization
# -------------------------------

def draw_person_ppe(frame: np.ndarray, person: Box, have_helmet: bool, have_vest: bool,
                    helmet_box: Optional[Box], vest_box: Optional[Box], track_id: int):
    x1, y1, x2, y2 = person.xyxy()
    color = (0, 200, 0) if (have_helmet and have_vest) else (0, 165, 255) if (have_helmet or have_vest) else (0, 0, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"ID {track_id} | helmet={'Y' if have_helmet else 'N'} vest={'Y' if have_vest else 'N'}"
    cv2.putText(frame, label, (x1, max(15, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    regs = head_torso_regions(person)
    hx1, hy1, hx2, hy2 = regs['head']
    tx1, ty1, tx2, ty2 = regs['torso']
    cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (128, 128, 128), 1)
    cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (128, 128, 128), 1)
    if helmet_box:
        cv2.rectangle(frame, (helmet_box.x1, helmet_box.y1), (helmet_box.x2, helmet_box.y2), (255, 255, 255), 2)
        cv2.putText(frame, "helmet", (helmet_box.x1, helmet_box.y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    if vest_box:
        cv2.rectangle(frame, (vest_box.x1, vest_box.y1), (vest_box.x2, vest_box.y2), (255, 255, 255), 2)
        cv2.putText(frame, "vest", (vest_box.x1, vest_box.y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

# -------------------------------
# Core monitor
# -------------------------------

class Monitor:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.assoc_cfg = AssocConfig(
            conf_helmet=args.conf_helmet,
            conf_vest=args.conf_vest,
            nms_iou=args.nms_iou,
            min_box_area=args.min_box_area,
            max_aspect_ratio=args.max_aspect_ratio,
            head_iou_gate=args.head_iou_gate,
            torso_iou_gate=args.torso_iou_gate,
            ppe_inside_person_frac=args.ppe_inside_person_frac,
            temporal_window=args.temporal_window,
        )
        # Person
        self.person_det = PersonDetector(weights=args.person_weights, conf=args.person_conf)
        # PPE backend
        self.ppe_backend = self._init_ppe_backend(args)
        # Tracking
        self.tracker = IoUTracker(iou_th=args.track_iou, max_age=args.track_max_age)
        # Smoother
        self.smoother = PPESmoother(window=self.assoc_cfg.temporal_window)
        # Output
        self.save_dir = Path(args.save_vis) if args.save_vis else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Monitor initialized.")

    def _init_ppe_backend(self, args) -> Any:
        if args.ppe_weights and _ULTRA_AVAILABLE:
            class_map = {0: 'helmet', 1: 'vest'}
            return PPEUltralytics(args.ppe_weights, min(args.conf_helmet, args.conf_vest), args.nms_iou, class_map)
        elif args.ppe_onnx:
            names = [n.strip() for n in (args.class_names or ['helmet', 'vest'])]
            return PPEOnnx(args.ppe_onnx, names, min(args.conf_helmet, args.conf_vest), args.nms_iou, input_size=args.input_size)
        else:
            raise RuntimeError("Provide --ppe-weights (.pt) or --ppe-onnx (.onnx)")

    def _filter_boxes(self, boxes: List[Box]) -> List[Box]:
        out = []
        for b in boxes:
            if b.area() < self.assoc_cfg.min_box_area:
                continue
            ar = b.h() / max(1e-6, b.w())
            if ar > self.assoc_cfg.max_aspect_ratio or ar < 1.0 / self.assoc_cfg.max_aspect_ratio:
                continue
            out.append(b)
        return out

    def _split_classes(self, boxes: List[Box]) -> Tuple[List[Box], List[Box]]:
        helmets, vests = [], []
        for b in boxes:
            lbl = b.label.lower()
            if lbl in ("helmet", "hardhat"):
                if b.conf >= self.assoc_cfg.conf_helmet:
                    helmets.append(b)
            elif lbl in ("vest", "jacket", "reflective_jacket", "reflective_vest"):
                if b.conf >= self.assoc_cfg.conf_vest:
                    vests.append(b)
        return helmets, vests

    def _assign_ppe_to_persons(self, persons: List[Box], helmets: List[Box], vests: List[Box]) -> Tuple[List[Optional[Box]], List[Optional[Box]]]:
        """Greedy one-to-one matching: choose PPE that best overlaps the correct region, and is sufficiently inside person box."""
        matched_h = [None] * len(persons)
        matched_v = [None] * len(persons)

        # Precompute regions
        regions = [head_torso_regions(p) for p in persons]

        # HELMETS
        used = set()
        for pi, p in enumerate(persons):
            head_reg = regions[pi]['head']
            best, best_score = None, 0.0
            for hi, h in enumerate(helmets):
                if hi in used:
                    continue
                # region IoU + containment inside person box
                reg_iou = iou(h.xyxy(), head_reg)
                inside_frac = inside_fraction(h.xyxy(), p.xyxy())
                score = reg_iou * 0.7 + inside_frac * 0.3
                if score > best_score:
                    best, best_score = (hi, h), score
            if best is not None:
                hi, h = best
                # gates
                if iou(h.xyxy(), head_reg) >= self.assoc_cfg.head_iou_gate and inside_fraction(h.xyxy(), p.xyxy()) >= self.assoc_cfg.ppe_inside_person_frac:
                    matched_h[pi] = h
                    used.add(hi)

        # VESTS
        used = set()
        for pi, p in enumerate(persons):
            torso_reg = regions[pi]['torso']
            best, best_score = None, 0.0
            for vi, v in enumerate(vests):
                if vi in used:
                    continue
                reg_iou = iou(v.xyxy(), torso_reg)
                inside_frac = inside_fraction(v.xyxy(), p.xyxy())
                score = reg_iou * 0.7 + inside_frac * 0.3
                if score > best_score:
                    best, best_score = (vi, v), score
            if best is not None:
                vi, v = best
                if iou(v.xyxy(), torso_reg) >= self.assoc_cfg.torso_iou_gate and inside_fraction(v.xyxy(), p.xyxy()) >= self.assoc_cfg.ppe_inside_person_frac:
                    matched_v[pi] = v
                    used.add(vi)

        return matched_h, matched_v

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        # Detect persons and PPE
        persons_raw = self.person_det.detect(frame)
        ppe_all = self.ppe_backend.infer(frame)
        ppe_all = self._filter_boxes(ppe_all)
        helmets, vests = self._split_classes(ppe_all)

        # Track persons (use detections as inputs)
        frame_height, frame_width = frame.shape[:2]
        tracks = self.tracker.update(persons_raw, frame_width, frame_height)
        # Freeze order for visualization/association
        track_items = sorted(tracks.items(), key=lambda kv: kv[0])  # (tid, Box)
        track_ids = [tid for tid, _ in track_items]
        persons = [box for _, box in track_items]

        # Assign PPE -> persons with geometry gates
        matched_h, matched_v = self._assign_ppe_to_persons(persons, helmets, vests)
        have_h = [mh is not None for mh in matched_h]
        have_v = [mv is not None for mv in matched_v]

        # Temporal smoothing over stable track IDs
        smoothed = self.smoother.update(track_ids, have_h, have_v)

        # Draw
        vis = frame.copy()
        for i, p in enumerate(persons):
            sh, sv = smoothed[i]
            draw_person_ppe(vis, p, sh, sv, matched_h[i], matched_v[i], track_ids[i])
        return vis

    # ---------------------------
    # Streaming run
    # ---------------------------
    def run_stream(self):
        source = self.args.source
        cap = cv2.VideoCapture(int(source)) if str(source).isdigit() else cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")

        # Video writer initialization
        out_writer = None
        if self.save_dir:
            output_path = str(self.save_dir / "output.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 25  # Fallback if FPS not available
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            logger.info(f"Saving annotated video to {output_path}")

        idx = 0
        t0 = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out = self.process_frame(frame)
            cv2.imshow("PPE Monitor", out)

            if self.save_dir is not None:
                # Save individual frames
                cv2.imwrite(str(self.save_dir / f"frame_{idx:06d}.jpg"), out)
                # Write frame to video
                if out_writer:
                    out_writer.write(out)

            idx += 1
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break

        cap.release()
        if out_writer:
            out_writer.release()
        cv2.destroyAllWindows()
        fps = idx / max(1e-6, time.time() - t0)
        logger.info(f"Processed {idx} frames at {fps:.2f} FPS")

    # ---------------------------
    # Validation on dataset
    # ---------------------------
    def run_eval(self, root: str):
        """Evaluates PPE detector (NOT person association) on a YOLO-format set.
        Computes per-class P/R and recommends thresholds. Use it to tune --conf-helmet / --conf-vest.
        """
        img_dir = Path(root) / "images"
        lbl_dir = Path(root) / "labels"
        if not img_dir.exists() or not lbl_dir.exists():
            raise RuntimeError(f"Expected {root}/images and {root}/labels")

        imgs = sorted(glob.glob(str(img_dir / "*.*")))
        if not imgs:
            raise RuntimeError(f"No images found in {img_dir}")

        gt_total = { "helmet": 0, "vest": 0 }
        tp = { "helmet": 0, "vest": 0 }
        fp = { "helmet": 0, "vest": 0 }

        # Evaluate with current thresholds
        for img_path in imgs:
            im = cv2.imread(img_path)
            if im is None:
                continue
            H, W = im.shape[:2]
            # Read GT
            label_path = str(lbl_dir / (Path(img_path).stem + ".txt"))
            gt_boxes: Dict[str, List[Tuple[int,int,int,int]]] = { "helmet": [], "vest": [] }
            if os.path.isfile(label_path):
                with open(label_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        cid, cx, cy, ww, hh = int(parts[0]), *map(float, parts[1:])
                        lbl = "helmet" if cid == 0 else "vest"
                        x = (cx - ww/2) * W
                        y = (cy - hh/2) * H
                        x1, y1, x2, y2 = int(x), int(y), int(x + ww*W), int(y + hh*H)
                        x1, y1, x2, y2 = clip_box((x1, y1, x2, y2), W, H)
                        gt_boxes[lbl].append((x1, y1, x2, y2))

            gt_total["helmet"] += len(gt_boxes["helmet"])
            gt_total["vest"] += len(gt_boxes["vest"])

            # Predictions (PPE only)
            preds = self._filter_boxes(self.ppe_backend.infer(im))
            helmets, vests = self._split_classes(preds)
            pred_map = {
                "helmet": [h.xyxy() for h in helmets],
                "vest": [v.xyxy() for v in vests],
            }
            # Match greedily by IoU >= 0.5
            for cls in ("helmet", "vest"):
                gts = gt_boxes[cls][:]
                prs = pred_map[cls][:]
                used = set()
                for g in gts:
                    best_i, best_j = 0.0, -1
                    for j, p in enumerate(prs):
                        if j in used:
                            continue
                        ii = iou(g, p)
                        if ii > best_i:
                            best_i, best_j = ii, j
                    if best_i >= 0.5 and best_j >= 0:
                        tp[cls] += 1
                        used.add(best_j)
                fp[cls] += max(0, len(prs) - len(used))

        def pr(cls):
            T = tp[cls]
            F = fp[cls]
            G = gt_total[cls]
            prec = T / max(1, T+F)
            rec = T / max(1, G)
            return prec, rec, T, F, G

        for cls in ("helmet", "vest"):
            prec, rec, T, F, G = pr(cls)
            logger.info(f"[{cls.upper()}] P={prec:.3f} R={rec:.3f}  TP={T} FP={F} GT={G}")

        # crude suggestion: if precision < 0.9, suggest increasing conf by +0.05
        suggest_h = self.assoc_cfg.conf_helmet + (0.05 if (tp["helmet"] / max(1, tp["helmet"]+fp["helmet"])) < 0.9 else 0.0)
        suggest_v = self.assoc_cfg.conf_vest + (0.05 if (tp["vest"] / max(1, tp["vest"]+fp["vest"])) < 0.9 else 0.0)
        logger.info(f"SUGGESTED conf_helmet≈{suggest_h:.2f}, conf_vest≈{suggest_v:.2f} (raise if still many FP)")
        logger.info("Note: eval ignores person association; it isolates PPE detector quality.")

# -------------------------------
# CLI
# -------------------------------

def build_argparser() -> argparse.Namespace:
    p = argparse.ArgumentParser("Improved YOLO PPE Monitor")
    # Inputs
    p.add_argument('--source', type=str, default='0', help='Video source path or camera index (string or int)')
    # Person model
    p.add_argument('--person-weights', type=str, default='yolo12n.pt', help='Ultralytics model for person')
    p.add_argument('--person-conf', type=float, default=0.45, help='Confidence for person detector')
    # PPE model (choose one)
    p.add_argument('--ppe-weights', type=str, default='', help='Ultralytics .pt weights for PPE (helmet/vest)')
    p.add_argument('--ppe-onnx', type=str, default='', help='YOLOv7 ONNX model for PPE')
    p.add_argument('--class-names', nargs='*', default=['helmet', 'vest'], help='Class names for ONNX model')
    p.add_argument('--input-size', type=int, default=640, help='ONNX input size (square)')
    # Thresholds / NMS
    p.add_argument('--conf-helmet', type=float, default=0.65)
    p.add_argument('--conf-vest', type=float, default=0.70)
    p.add_argument('--nms-iou', type=float, default=0.50)
    p.add_argument('--min-box-area', type=int, default=30*30)
    p.add_argument('--max-aspect-ratio', type=float, default=3.5)
    p.add_argument('--head-iou-gate', type=float, default=0.10)
    p.add_argument('--torso-iou-gate', type=float, default=0.15)
    p.add_argument('--ppe-inside-person-frac', type=float, default=0.35)
    p.add_argument('--temporal-window', type=int, default=7)
    # Tracking
    p.add_argument('--track-iou', type=float, default=0.35, help='IoU match threshold for tracker')
    p.add_argument('--track-max-age', type=int, default=20, help='Max frames to keep unmatched track (default: 20, original setting)')
    p.add_argument('--track-edge-cleanup', action='store_true', help='Enable aggressive edge cleanup for tracking boxes')
    # Output
    p.add_argument('--save-vis', type=str, default='', help='Directory to save annotated frames')
    # Eval
    p.add_argument('--eval-root', type=str, default='', help='If set (e.g., dataset/valid), runs evaluation and exits')
    args = p.parse_args()
    # normalize empties
    if args.ppe_weights == '':
        args.ppe_weights = None
    if args.ppe_onnx == '':
        args.ppe_onnx = None
    return args


def main():
    args = build_argparser()
    mon = Monitor(args)
    if args.eval_root:
        mon.run_eval(args.eval_root)
        return
    mon.run_stream()

if __name__ == '__main__':
    main()
