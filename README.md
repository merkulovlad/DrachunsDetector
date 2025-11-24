# Violence Detection System

## Objective  
This project focuses on building a system that can **detect violent activity in video**, preferably in **real-time**.  

## Why It Matters  
By automatically identifying violent behavior, the system aims to:  
- Increase **public safety**  
- Enable **faster response times** for law enforcement


## Intended Users  
- Ministry of Internal Affairs  
- Police departments  
- Emergency services  


## Structure of repository

- `main_code`: SCT-GCN model code and supporting utilities.
- `notebooks`: experiment notebooks and exploratory runs.
- `stream_platform`: MoViNet streaming pipeline components with MediaMTX.
- `r2p1d18`: R2+1D model code and assets.
- `app`: FastAPI service for exposing the models.

## Prerequisites

- **Docker** (latest stable version)
- **Docker Compose** (if not included in your Docker installation)

 Check versions:

```bash
docker --version
docker compose version
```

## How to setup
1. Clone repository
```bash
git clone https://github.com/merkulovlad/DrachunsDetector
cd DrachunsDetector
```
2. Build and start with Docker
```bash
docker compose up --build
```
