"""Main FastAPI application for task orchestration.
This rewrite separates concerns clearly:
* Postgres (SQLAlchemy) stores task metadata and results.
* tusd handles file uploads; we create four placeholders for each task
  so the client can PATCH the data asynchronously.
* A light in-process queue executes heavy 3-D processing in the background.
* MinIO/Boto3 generates presigned URLs for results on demand.

Environment variables expected (see docker-compose):

    DATABASE_URL        postgresql://tus:tuspass@db:5432/tasks
    TUS_ENDPOINT        http://tusd:1080/files
    MINIO_ENDPOINT      http://minio:9000
    MINIO_ACCESS_KEY    minioadmin
    MINIO_SECRET_KEY    minioadmin123
    AWS_S3_ADDRESSING_STYLE path
"""
from __future__ import annotations

import base64
import io
import os
import queue
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path
from enum import Enum as PyEnum

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, status, APIRouter, Request
from pydantic import BaseModel
from sqlalchemy import Enum as SQLEnum
from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    String,
    create_engine,
    text,
    Boolean,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker
from starlette.responses import JSONResponse

import laspy
import numpy as np
import torch
import geopandas as gpd
import py4dgeo
import open3d as o3d
import volcal_baseline.pipeline as pipeline
import requests
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ────────────────────────────── settings ────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL")
TUS_ENDPOINT = os.getenv("TUS_ENDPOINT")

s3_client = boto3.client(
    "s3",
    endpoint_url=os.getenv("MINIO_ENDPOINT"),
    aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("MINIO_SECRET_KEY"),
    config=Config(
        signature_version="s3v4",
        s3={"addressing_style": os.getenv("AWS_S3_ADDRESSING_STYLE", "path")},
    ),
)

DEVICE = torch.device("cuda:{}".format(0))
PADDING = "same"
VOXEL_SIZE = 2.0
PV = 5
NV = -1
PPV = -1
NUM_WORKERS = 24
ROTATION_CHOICE = "gen"
QUANTILE_THR = 0.2
ICP_VERSION = "generalized"
MAX_ITER = 2048
REF_RATIO = 0.01

CHUNK_SIZE   = 10 * 1024 * 1024

# ────────────────────────────── database ────────────────────────────────
engine = create_engine(DATABASE_URL, pool_size=10, max_overflow=20, echo=False)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
Base = declarative_base()


class TaskStatus(PyEnum):
    pending = "pending"  # files uploaded, waiting for worker
    running = "running"  # worker processing
    done    = "done"     # finished successfully
    error   = "error"    # failed
    waiting = "waiting"  # waiting for file uploads
    killed  = "killed"   # worker killed


class Task(Base):
    __tablename__ = "tasks"

    id: uuid.UUID | str = Column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=text("uuid_generate_v4()")
    )
    status : TaskStatus = Column(
        SQLEnum(TaskStatus, name="taskstatus"), 
        default=TaskStatus.waiting, 
        nullable=False
    )

    pre_key: Optional[str] = Column(String)
    post_key: Optional[str] = Column(String)
    shp_key: Optional[str] = Column(String)
    shx_key: Optional[str] = Column(String)
    
    pre_done: bool = Column(Boolean, default=False, nullable=False)
    post_done: bool = Column(Boolean, default=False, nullable=False)
    shp_done: bool = Column(Boolean, default=False, nullable=False)
    shx_done: bool = Column(Boolean, default=False, nullable=False)

    result: Optional[Dict[str, Any]] = Column(JSON)
    error: Optional[str] = Column(String)

    created_at: datetime = Column(DateTime, default=datetime.utcnow)
    started_at: Optional[datetime] = Column(DateTime)
    finished_at: Optional[datetime] = Column(DateTime)


Base.metadata.create_all(engine)


# ────────────────────────────── utils ───────────────────────────────────

def get_db():
    """FastAPI dependency providing a SQLAlchemy session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _b64(text_: str) -> str:
    return base64.b64encode(text_.encode()).decode()


def create_tus_placeholder(task_id: str, label: str) -> tuple[str, str]:
    """Create an *empty* upload at tusd and return (location, object_key)."""
    filename_meta = _b64(f"{task_id}_{label}")
    metadata_hdr = f"filename {filename_meta},task_id {_b64(task_id)},label {_b64(label)}"

    headers = {
        "Tus-Resumable": "1.0.0",
        "Upload-Defer-Length": "1",
        "Upload-Metadata": metadata_hdr,
        "Content-Length": "0",
    }

    resp = requests.post(TUS_ENDPOINT, headers=headers, allow_redirects=False, timeout=10)
    if resp.status_code != 201:
        raise RuntimeError(f"tusd returned {resp.status_code}: {resp.text}")
    location = resp.headers["Location"]
    key = location.rsplit("/", 1)[-1].split("+")[0]
    return location, key


def all_file_keys(task: Task) -> Dict[str, Optional[str]]:
    return {
        "pre": task.pre_key,
        "post": task.post_key,
        "shp": task.shp_key,
        "shx": task.shx_key,
    }


def missing_files(task: Task) -> List[str]:
    missing = []
    for label in ("pre", "post", "shp", "shx"):
        if not getattr(task, f"{label}_done"):
            missing.append(label)
    return missing


def presign_s3(key: str, bucket: str = "uploads", expires: int = 900) -> str:
    return s3_client.generate_presigned_url(
        "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expires
    )


def get_file(bucket="uploads", key=None, local_path=None):
    if not key or not local_path:
        raise ValueError("Both key and local_path must be provided.")
    
    Path(os.path.dirname(local_path)).mkdir(parents=True, exist_ok=True)
    try:
        s3_client.download_file(Bucket=bucket, Key=key, Filename=local_path)
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            print(f"Object not found: {bucket}/{key}")
            raise SystemExit(1)
        else:
            raise


def tus_upload(buf: bytes, filename: str, endpoint: str = TUS_ENDPOINT) -> str:
    """上传内存中的 bytes 到 tusd, 返回 object key"""
    meta = base64.b64encode(filename.encode()).decode()
    headers = {
        "Tus-Resumable": "1.0.0",
        "Upload-Length": str(len(buf)),
        "Upload-Metadata": f"filename {meta}",
    }
    # 1) Create
    r = requests.post(endpoint, headers=headers, allow_redirects=False)
    r.raise_for_status()
    location = r.headers["Location"]
    key = location.split("/")[-1].split("+")[0]

    # 2) PATCH chunks
    offset = 0
    while offset < len(buf):
        chunk = buf[offset : offset + CHUNK_SIZE]
        headers = {
            "Tus-Resumable": "1.0.0",
            "Content-Type": "application/offset+octet-stream",
            "Upload-Offset": str(offset),
        }
        r = requests.patch(location, data=chunk, headers=headers)
        r.raise_for_status()
        offset = int(r.headers["Upload-Offset"])
    return key


# ────────────────────────────── worker thread ───────────────────────────
TASK_QUEUE: "queue.Queue[str]" = queue.Queue()
shutdown_flag = threading.Event()


def heavy_processing(task: Task, db: Session):
    """Placeholder for heavy 3-D processing pipeline.
    Adapt it to call your existing pipeline functions and populate `task.result`.
    """
    try:
        task_dir = Path("/tmp") / str(task.id)
        task_dir.mkdir(parents=True, exist_ok=True)
        
        pre_path  = task_dir / "pre.las"
        post_path = task_dir / "post.las"
        shp_path  = task_dir / "shape.shp"
        shx_path  = task_dir / "shape.shx"
        
        get_file(key=task.pre_key,  local_path=str(pre_path))
        get_file(key=task.post_key, local_path=str(post_path))
        get_file(key=task.shp_key,  local_path=str(shp_path))
        get_file(key=task.shx_key, local_path=str(shx_path))
        
        pre_las = laspy.read(pre_path)
        x1, y1, z1 = pre_las.x, pre_las.y, pre_las.z
        
        post_las = laspy.read(post_path)
        x4_trans, y4_trans, z4_trans = post_las.x, post_las.y, post_las.z
        
        stable_shp = gpd.read_file(shp_path)
        
        source = np.vstack((x4_trans, y4_trans, z4_trans)).T
        target = np.vstack((x1, y1, z1)).T
        
        EGS_T = pipeline.EGS(
            source=source,
            target=target,
            voxel_size=VOXEL_SIZE,
            padding=PADDING,
            ppv=PPV,
            pv=PV,
            nv=NV,
            num_workers=NUM_WORKERS,
            rotation_choice=ROTATION_CHOICE,
            rotation_root_path="volcal_baseline/exhaustive-grid-search/data/rotations",
        )
        
        GICP_T = pipeline.auto_GICP(
            source=source, target=target, T_init=EGS_T, thr=QUANTILE_THR, max_iter=MAX_ITER
        )
        
        x4_refined, y4_refined, z4_refined = pipeline.apply_transformation(x4_trans, y4_trans, z4_trans, GICP_T)
        cloud4_refined = np.vstack((x4_refined, y4_refined, z4_refined)).T

        stable_before, mask_before = pipeline.isolate_stable(target, stable_shp)
        stable_after,  mask_after  = pipeline.isolate_stable(cloud4_refined, stable_shp)

        epoch_stable_before = py4dgeo.Epoch(stable_before)
        epoch_stable_after  = py4dgeo.Epoch(stable_after)
        
        avg_spacing = pipeline.estimate_avg_spacing(target)
        init_voxel_size = avg_spacing / REF_RATIO
        adapted_voxel_size = pipeline.adaptive_voxel_size(target, REF_RATIO, init_voxel_size, 25, 15, 1)
        down_source = pipeline.voxel_downsample(target, adapted_voxel_size)
        
        m3c2 = py4dgeo.M3C2(
            epochs=(epoch_stable_before, epoch_stable_after),
            corepoints=epoch_stable_before.cloud[::],
            normal_radii=(adapted_voxel_size * 2.0,),
            cyl_radius=(adapted_voxel_size),
            max_distance=(15.0),
            registration_error=(0.0),
        )
        m3c2_distances_stableparts, uncertainties_stableparts = m3c2.run()
        reg_target_source = np.nanstd(m3c2_distances_stableparts)
        
        epoch_before = py4dgeo.Epoch(target)
        epoch_after = py4dgeo.Epoch(cloud4_refined)

        corepoints_pcd        = o3d.geometry.PointCloud()
        corepoints_pcd.points = o3d.utility.Vector3dVector(epoch_before.cloud)
        corepoints_pcd        = corepoints_pcd.voxel_down_sample(voxel_size=adapted_voxel_size)#0.1)
        corepoints            = np.asarray(corepoints_pcd.points)

        m3c2 = py4dgeo.M3C2(
            epochs=(epoch_before, epoch_after),
            corepoints=corepoints,
            normal_radii=(adapted_voxel_size * 2.0,),
            cyl_radius=(adapted_voxel_size),
            max_distance=(15.0),
            registration_error=(reg_target_source),
        )
        m3c2_distances, uncertainties = m3c2.run()
        change_sign = np.where(abs(m3c2_distances) > uncertainties["lodetection"], True, False)
        
        results = []
        hulls = pipeline.segment_changes(corepoints, change_sign)
        for idx, hull in enumerate(hulls):
            hull_id = f"{task.id}_{idx}"
            
            # ---------- Volume calculation ----------
            inside_raw     = pipeline.is_inside_selected_hulls_vectorized([hull], target[:, :2])
            inside_refined = pipeline.is_inside_selected_hulls_vectorized([hull], cloud4_refined[:, :2])

            if inside_raw.sum() == 0 or inside_refined.sum() == 0:
                print(f"Skip {hull_id}: no points inside")
                continue

            dem_before, dem_after, _, _, grid_res = pipeline.reletive_DEM(
                target[inside_raw], cloud4_refined[inside_refined],
                grid_res=None, method="linear", mask_hulls=[hull]
            )
            net_vol, cut_vol, fill_vol, _ = pipeline.calculate_volume(
                dem_before, dem_after, grid_res=grid_res, threshold=reg_target_source
            )
            
            # ---------- Image generation ----------
            inside_nonroi_raw = pipeline.is_inside_selected_hulls_vectorized(
                [h for h in hulls if h is not hull], target[:, :2]
            )
            mask_combined = np.logical_and(~inside_nonroi_raw, ~inside_raw)

            filtered_raw_roi      = target[inside_raw]
            unfiltered_raw_other  = target[mask_combined][::100]
            filtered_raw_nonroi   = target[inside_nonroi_raw]

            fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(7, 5))
            ax.scatter(filtered_raw_roi[:, 0], filtered_raw_roi[:, 1], filtered_raw_roi[:, 2],
                    c=filtered_raw_roi[:, 2], cmap="viridis", s=1)
            ax.scatter(unfiltered_raw_other[:, 0], unfiltered_raw_other[:, 1], unfiltered_raw_other[:, 2],
                    c="red", s=1, alpha=0.1)
            ax.scatter(filtered_raw_nonroi[:, 0], filtered_raw_nonroi[:, 1], filtered_raw_nonroi[:, 2],
                    c="blue", s=1)
            ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
            ax.view_init(elev=40, azim=240)
            plt.colorbar(ax.collections[0], ax=ax, pad=0.1, label="Elevation [m]")
            plt.tight_layout()
            
            # Save the figure to a buffer and upload to TUS
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
            plt.close(fig)
            img_key = tus_upload(buf.getvalue(), f"{hull_id}.png")
            
            # ---------- Save results ------
            results.append(
                dict(
                    id=hull_id,
                    area=float(hull.area),
                    cut_volume=float(cut_vol),
                    fill_volume=float(fill_vol),
                    net_volume=float(net_vol),
                    image_key=img_key,
                )
            )
        task.result = {"hulls": results}
        task.status = TaskStatus.done
        task.finished_at = datetime.utcnow()
    except Exception as exc:
        task.status = TaskStatus.error
        task.error = str(exc)
        task.finished_at = datetime.utcnow()
    finally:
        db.add(task)
        db.commit()


def worker_loop():
    while not shutdown_flag.is_set():
        task_id: str = TASK_QUEUE.get()
        with SessionLocal() as db:
            task: Task = db.get(Task, uuid.UUID(task_id))
            if not task:
                TASK_QUEUE.task_done()
                continue
            task.status = TaskStatus.running
            task.started_at = datetime.utcnow()
            db.add(task)
            db.commit()
            heavy_processing(task, db)
        TASK_QUEUE.task_done()


worker_thread = threading.Thread(target=worker_loop, daemon=True, name="worker")


# ────────────────────────────── API schemas ─────────────────────────────


class CreateTaskResponse(BaseModel):
    task_id: str
    upload_urls: Dict[str, str]


class StartTaskResponse(BaseModel):
    started: bool


class QueryTaskResponse(BaseModel):
    status: str
    missing: Optional[List[str]] = None
    results: Optional[Any] = None


# ────────────────────────────── API routes ──────────────────────────────

router = APIRouter()


@router.post("/tusd_hook", status_code=status.HTTP_200_OK)
async def tusd_hook(req: Request, db: Session = Depends(get_db)):
    payload = await req.json()
    
    if payload.get("Type") != "post-finish":
        return {}

    meta = payload.get("Event", {}).get("Upload", {}).get("MetaData", {})
    task_id = meta.get("task_id")
    label   = meta.get("label")
    if not (task_id and label):
        return {}

    try:
        task_uuid = uuid.UUID(task_id)
    except ValueError:
        return {}

    task = db.get(Task, task_uuid)
    if not task:
        return {}
    
    setattr(task, f"{label}_done", True)
    
    if all(getattr(task, f"{lbl}_done") for lbl in ("pre", "post", "shp", "shx")):
        task.status = TaskStatus.pending
        db.add(task)
        db.commit()
        
        TASK_QUEUE.put(str(task.id))
    
    db.add(task)
    db.commit()
    
    return {}


# ────────────────────────────── FastAPI app ─────────────────────────────

app = FastAPI(title="Task Orchestrator", version="1.0.0")
app.include_router(router)


@app.on_event("startup")
def on_startup():
    worker_thread.start()
    
    with SessionLocal() as db:
        stale_tasks = db.query(Task).filter(
            Task.status.in_([TaskStatus.running, TaskStatus.pending])
        ).all()
        for task in stale_tasks:
            task.status = TaskStatus.killed
            task.finished_at = datetime.utcnow()
            db.add(task)
        db.commit()


@app.on_event("shutdown")
def on_shutdown():
    shutdown_flag.set()
    worker_thread.join(timeout=5)
    
    with SessionLocal() as db:
        db.query(Task) \
          .filter(Task.status == TaskStatus.running) \
          .update({Task.status: TaskStatus.killed})
        db.commit()


@app.post("/create_task", response_model=CreateTaskResponse, status_code=status.HTTP_201_CREATED)
def create_task(db: Session = Depends(get_db)):
    """Create a task and return 4 TUS upload endpoints (pre/post/shp/shx)."""
    new_task = Task()
    db.add(new_task)
    db.commit()
    db.refresh(new_task)

    urls: Dict[str, str] = {}
    for label in ("pre", "post", "shp", "shx"):
        loc, key = create_tus_placeholder(str(new_task.id), label)
        urls[label] = loc
        setattr(new_task, f"{label}_key", key)  # store expected object id
    new_task.status = TaskStatus.waiting
    db.add(new_task)
    db.commit()

    return CreateTaskResponse(task_id=str(new_task.id), upload_urls=urls)


@app.post("/start_task/{task_id}", response_model=StartTaskResponse)
def start_task(task_id: str, db: Session = Depends(get_db)):
    task = db.get(Task, uuid.UUID(task_id))
    if not task:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Task not found")

    missing = missing_files(task)
    if missing:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"missing": missing},
        )

    if task.status in (TaskStatus.pending, TaskStatus.running):
        return {"started": True}

    task.status = TaskStatus.pending
    db.add(task)
    db.commit()

    TASK_QUEUE.put(task_id)
    return StartTaskResponse(started=True)


@app.get("/query_task/{task_id}", response_model=QueryTaskResponse)
def query_task(task_id: str, db: Session = Depends(get_db)):
    task = db.get(Task, uuid.UUID(task_id))
    if not task:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Task not found")

    if task.status in (TaskStatus.running, TaskStatus.pending, TaskStatus.waiting):
        return QueryTaskResponse(status=task.status, missing=missing_files(task))

    if task.status == TaskStatus.error:
        return QueryTaskResponse(status="error", results=task.error)

    enriched = []
    if task.result and "hulls" in task.result:
        for hull in task.result["hulls"]:
            enriched.append({
                **hull,
                "image_url": presign_s3(hull["image_key"]),
            })

    return QueryTaskResponse(
        status="done",
        results={"hulls": enriched},
    )
