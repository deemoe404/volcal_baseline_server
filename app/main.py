# Application
import os, pathlib
import laspy
import numpy as np
import torch
import geopandas as gpd
import py4dgeo
import open3d as o3d
import volcal_baseline.pipeline as pipeline

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

import io, os, base64, uuid, csv
import requests, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

TUS_ENDPOINT = os.getenv("TUS_ENDPOINT")
CHUNK_SIZE   = 10 * 1024 * 1024

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


# File handler
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

s3 = boto3.client(
    "s3",
    endpoint_url=os.getenv("MINIO_ENDPOINT"),
    aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("MINIO_SECRET_KEY"),
    config=Config(
        signature_version="s3v4",
        s3={"addressing_style": os.getenv("AWS_S3_ADDRESSING_STYLE", "path")},
    )
)

def get_file(bucket="uploads", key=None, local_path=None):
    if not key or not local_path:
        raise ValueError("Both key and local_path must be provided.")
    
    pathlib.Path(os.path.dirname(local_path)).mkdir(parents=True, exist_ok=True)
    try:
        s3.download_file(Bucket=bucket, Key=key, Filename=local_path)
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            print(f"Object not found: {bucket}/{key}")
            raise SystemExit(1)
        else:
            raise

# Application code

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import threading, queue, uuid, time, atexit

app = FastAPI(title="Heavy Task Queue")

# ── 全局任务队列 & 状态存储 ────────────────────────────────────
TASK_QUEUE: queue.Queue = queue.Queue()
TASKS     : dict[str, dict] = {}   # task_id -> {status, …}

# ── 工作线程：始终只有 1 个，串行消费队列 ────────────────────
def worker():
    while True:
        task_id, params = TASK_QUEUE.get()     # 阻塞等待
        TASKS[task_id]["status"] = "running"
        try:
            pre_key, post_key, shp_key, shx_key = params
            # ---------- 你的重活开始 ----------
            get_file(key=shp_key, local_path=f"/tmp/shape.shp")
            get_file(key=shx_key, local_path=f"/tmp/shape.shx")
            stable_shp = gpd.read_file(f"/tmp/shape.shp")
            
            get_file(key=pre_key, local_path=f"/tmp/{pre_key}.las")
            pre_las = laspy.read(f"/tmp/{pre_key}.las")
            x1, y1, z1 = pre_las.x, pre_las.y, pre_las.z
            
            get_file(key=post_key, local_path=f"/tmp/{post_key}.las")
            post_las = laspy.read(f"/tmp/{post_key}.las")
            x4_trans, y4_trans, z4_trans = post_las.x, post_las.y, post_las.z
            
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
                hull_id = f"h{idx:03d}"  # 若需全局唯一可改为 uuid.uuid4().hex[:8]

                # ---------- 1) 体积 ----------
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

                # ---------- 2) 绘图 ----------
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

                # 保存到 BytesIO
                buf = io.BytesIO()
                plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
                plt.close(fig)
                img_key = tus_upload(buf.getvalue(), f"{hull_id}.png")

                # ---------- 3) 记录 ----------
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
            
            csv_buf = io.StringIO()
            writer = csv.DictWriter(csv_buf, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

            csv_key = tus_upload(csv_buf.getvalue().encode(), "hull_volumes.csv")
            # ---------- 你的重活结束 ----------
            TASKS[task_id] |= {"status":"done", "result":csv_key}
        except Exception as e:
            TASKS[task_id] |= {"status":"error", "error":str(e)}
        finally:
            TASK_QUEUE.task_done()

worker_thread = threading.Thread(target=worker, daemon=True)
worker_thread.start()
atexit.register(worker_thread.join, timeout=1)

# ── 请求 & 模型 ────────────────────────────────────────────
class TaskParams(BaseModel):
    pre_key : str
    post_key: str
    shp_key : str
    shx_key : str

@app.post("/process", status_code=202)
def process(params: TaskParams):
    task_id = str(uuid.uuid4())
    TASKS[task_id] = {"status":"pending", "params":params.dict()}
    TASK_QUEUE.put((task_id, (params.pre_key, params.post_key, params.shp_key, params.shx_key)))
    # Location 头可帮助客户端直接跳到查询接口
    return {"task_id": task_id, "status_url": f"/status/{task_id}"}

@app.get("/status/{task_id}")
def status(task_id: str):
    info = TASKS.get(task_id)
    if not info:
        raise HTTPException(404, "任务不存在")
    return info

@app.get("/result/{task_id}")
def result(task_id: str):
    info = TASKS.get(task_id)
    if not info:
        raise HTTPException(404, "任务不存在")
    if info["status"] == "done":
        return {"result": info["result"]}
    elif info["status"] == "error":
        raise HTTPException(500, f"任务失败: {info['error']}")
    else:
        raise HTTPException(202, "任务尚未完成")

# ── 健康检查（可选）─────────────────────────────────────────
@app.get("/healthz")
def health():
    return {"queue_length": TASK_QUEUE.qsize(),
            "running": any(t["status"]=="running" for t in TASKS.values())}

@app.get("/presign/{key}")
def presign(key: str, bucket: str = "uploads", expires: int = 900):
    try:
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expires
        )
        return {"url": url}
    except Exception as e:
        raise HTTPException(404, f"key {key} not found") from e
