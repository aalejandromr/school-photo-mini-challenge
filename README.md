# 🎨 Realistic Shadow Generator

A full-stack web application that generates realistic shadows by compositing foreground subjects onto background images with advanced shadow effects including contact shadows, soft falloff, and directional light control.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [API Documentation](#api-documentation)
- [Usage Guide](#usage-guide)

## 🎯 Overview

This application takes a foreground image (with a subject) and a background image, then generates a realistic composite where the subject casts a shadow on the background. The shadow includes:

- **Directional light control** - Adjustable light angle (0-360°) and elevation (0-90°)
- **Contact shadow** - Sharp, dark shadow near contact points that fades with distance
- **Soft shadow falloff** - Progressive blur and opacity decrease with distance
- **Subject silhouette matching** - Shadow accurately follows the subject's shape

## ✨ Features

### Core Features
- ✅ Automatic subject extraction using AI (rembg)
- ✅ Realistic shadow generation with contact and soft components
- ✅ Directional light control (angle and elevation)
- ✅ Real-time preview with debounced generation
- ✅ Download individual images or complete ZIP package
- ✅ Debug images included (shadow_only.png, mask_debug.png)

### Output Files
The API returns a ZIP file containing:
- **composite.png** - Final result (background + shadow + foreground)
- **shadow_only.png** - Shadow visualization on transparent background
- **mask_debug.png** - Extracted subject mask for debugging

## 🏗️ Architecture

```
┌─────────────────┐         HTTP/REST         ┌─────────────────┐
│                 │ ◄──────────────────────► │                 │
│   React 19      │    POST /api/generate-    │   FastAPI       │
│   Frontend      │         shadow           │   Backend       │
│   (Port 3000)   │                          │   (Port 8000)   │
│                 │                          │                 │
│  - Image Upload │                          │  - Subject      │
│  - Light Controls│                         │    Extraction   │
│  - Preview      │                          │  - Shadow       │
│  - Download     │                          │    Generation   │
│                 │                          │  - Compositing  │
└─────────────────┘                          └─────────────────┘
```

### Data Flow

1. **User uploads images** → Frontend sends to API via FormData
2. **API processes images** → Extracts subject, generates shadow, composites
3. **API returns ZIP** → Contains composite + debug images
4. **Frontend extracts ZIP** → Displays composite, enables downloads

## 🛠️ Technology Stack

### Backend
- **FastAPI** - Modern Python web framework
- **Uvicorn** - ASGI server
- **OpenCV** - Image processing and transformations
- **Pillow (PIL)** - Image manipulation
- **rembg** - AI-powered subject extraction
- **NumPy** - Array operations
- **Python 3.9+**

### Frontend
- **React 19** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **Axios** - HTTP client
- **JSZip** - ZIP file handling

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- Node.js 18+ and npm
- Virtual environment (for Python)

## Preview

You can preview the results with the provided `composite.png`, `mask_debug.png` and `shadow_only.png`