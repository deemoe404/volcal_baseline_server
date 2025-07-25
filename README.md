
# Task Orchestrator API Guide

This guide describes how to use the Task Orchestrator FastAPI application for [volcal_baseline](https://github.com/deemoe404/volcal_baseline).

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [API Endpoints](#api-endpoints)
  - [Create Task](#create-task)
  - [Start Task](#start-task)
  - [Query Task](#query-task)
  - [TUSD Hook](#tusd-hook)
- [Task Lifecycle](#task-lifecycle)
- [File Upload Workflow](#file-upload-workflow)
- [Presigned Result URLs](#presigned-result-urls)
- [Error Handling](#error-handling)
- [License](#license)

---

## Overview

The Task Orchestrator is a FastAPI application designed to manage [volcal_baseline](https://github.com/deemoe404/volcal_baseline) tasks. It separates concerns as follows:

- **PostgreSQL (SQLAlchemy):** Stores task metadata and processing results.
- **tusd (TUS protocol):** Handles large file uploads asynchronously.
- **Background Worker:** Executes [volcal_baseline](https://github.com/deemoe404/volcal_baseline) worker in a separate thread queue.
- **MinIO/Boto3:** Generates presigned URLs for processed result assets on demand.

Clients interact with the service via RESTful endpoints to create tasks, upload data, trigger processing, and fetch results.

## Requirements

- Python 3.11+
- PostgreSQL 12+
- Docker & Docker Compose (recommended, for deployment)

Python dependencies could be installed via `pip install -r requirements.txt` if you are not using Docker.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-org/task-orchestrator.git
   cd task-orchestrator
   ```
2. **Configure environment** by editing the `docker-compose.yml` file.
3. **Start services** (with Docker Compose):
   ```bash
    docker compose up
    ```
4. **Run migrations** (if any) or allow SQLAlchemy to create tables automatically.

## API Endpoints

### Create Task

```http
POST /create_task
```

**Description:** Creates a new task and returns four TUS upload URLs for the required files:
- `pre`: Pre-change point cloud (LAS)
- `post`: Post-change point cloud (LAS)
- `shp`: Shapefile (.shp)
- `shx`: Shapefile index (.shx)

**Response (201 Created):**

```json
{
  "task_id": "<uuid>",
  "upload_urls": {
    "pre": "<tus_upload_endpoint>",
    "post": "<tus_upload_endpoint>",
    "shp": "<tus_upload_endpoint>",
    "shx": "<tus_upload_endpoint>"
  }
}
```

**Example:**

```bash
curl -X POST http://localhost:8000/create_task
```

### Start Task

```http
POST /start_task/{task_id}
```

- **Description:** Validates that all files have been uploaded. If so, enqueues the task for processing.
- **Path Parameter:** `task_id` – UUID of the task.

**Responses:**

- **200 OK:** `{ "started": true }`
- **400 Bad Request:** `{ "missing": ["pre", "post"] }` if files are missing.
- **404 Not Found:** if `task_id` is invalid.

**Example:**

```bash
curl -X POST http://localhost:8000/start_task/123e4567-e89b-12d3-a456-426614174000
```

### Query Task

```http
GET /query_task/{task_id}
```

- **Description:** Retrieves the current status, missing files (if any), or results if complete.
- **Path Parameter:** `task_id` – UUID of the task.

**Response Examples:**

- **In Progress (200 OK):**
  ```json
  {
    "status": "running",
    "missing": null,
    "results": null
  }
  ```
- **Pending Uploads (200 OK):**
  ```json
  {
    "status": "waiting",
    "missing": ["shp", "shx"]
  }
  ```
- **Error (200 OK):**
  ```json
  {
    "status": "error",
    "results": "<error message>"
  }
  ```
- **Done (200 OK):**
  ```json
  {
    "status": "done",
    "results": {
      "hulls": [
        {
          "id": "<hull_id>",
          "area": 12.34,
          "cut_volume": 5.67,
          "fill_volume": 2.34,
          "net_volume": 3.33,
          "image_url": "<presigned_url>"
        }
      ]
    }
  }
  ```

### TUSD Hook

```http
POST /tusd_hook
```

- **Description:** Internal webhook endpoint called by tusd when an upload finishes. Marks individual file parts as done and enqueues the task when all are ready.
- **Authentication:** Handled internally; clients do not call this endpoint directly.

## Task Lifecycle

Tasks progress through the following statuses:

1. **waiting:** Task record created; waiting for uploads.
2. **pending:** All files uploaded; queued for processing.
3. **running:** Worker thread is processing the task.
4. **done:** Processing completed successfully.
5. **error:** Processing failed; see `error` field in results.
6. **killed:** Task was terminated on startup/shutdown or after a crash.

## File Upload Workflow

1. **Create Task:** Obtain `upload_urls` for each label.
2. **Client Uploads:** Use TUS client to `PATCH` each URL with the file bytes.
3. **tusd Webhook:** Marks each part as complete when upload finishes.
4. **Start Task:** Either manually via `/start_task` or automatically when all parts uploaded.

## Presigned Result URLs

Image assets generated during processing are stored in MinIO. Clients receive presigned URLs in the `results.hulls` array under `image_url`, valid for a limited time.

## Error Handling

- **404 Not Found:** Invalid `task_id`.
- **400 Bad Request:** Missing file labels on start.
- **500 Internal Server Error:** Unhandled exceptions.
- The `/query_task` endpoint surfaces processing errors under `results` when `status` is `error`.

## License

This project is released under the MIT License. Feel free to fork and modify.




Example of `/create_task` endpoint response:

``` shell
❯ curl -i -X POST http://192.168.1.26:8000/create_task
HTTP/1.1 201 Created
date: Fri, 25 Jul 2025 13:55:35 GMT
server: uvicorn
content-length: 855
content-type: application/json

{"task_id":"7a61d5b3-a254-42cd-8b5b-01b7463ddf0c","upload_urls":{"pre":"http://192.168.1.26:1080/files/cf38d9097fdc518e1eed1d3e1b4885d9+NmMzZWQ5ZTAtNWVkZS00ZmIyLWI4NGQtOWNkMjc0MDg0MjM4LjRkZGNlMDUyLWFiMDQtNGI2ZC05MWY1LTE4MjcyZjQyZGFiY3gxNzUzNDUxNzM2MTA4NDg5OTkz","post":"http://192.168.1.26:1080/files/a9b9dfacda8db22de66af51931f4efbf+NmMzZWQ5ZTAtNWVkZS00ZmIyLWI4NGQtOWNkMjc0MDg0MjM4LjA1ZGY5MGE0LWRjNWEtNDY4Mi1iYWI0LWMxYzQ1ODhmYjg2YXgxNzUzNDUxNzM2MTE3MDQ2NjE3","shp":"http://192.168.1.26:1080/files/d375958a45efae208f64cca80c281ab8+NmMzZWQ5ZTAtNWVkZS00ZmIyLWI4NGQtOWNkMjc0MDg0MjM4LjQ1ZjI5ZDg2LTJjOWMtNGNjZC1iMDViLTE1YjYyMzhhNDNmY3gxNzUzNDUxNzM2MTIzODIxNzYw","shx":"http://192.168.1.26:1080/files/a6bd157d0e474999230ef48390fb1841+NmMzZWQ5ZTAtNWVkZS00ZmIyLWI4NGQtOWNkMjc0MDg0MjM4LmRkYWNmMTQ3LWM1ZjYtNDcwNy05ZDg0LTNkNjA5NzI0OTI4YngxNzUzNDUxNzM2MTMwNTY2NTkx"}}

❯ curl -i -X GET http://192.168.1.26:8000/query_task/7a61d5b3-a254-42cd-8b5b-01b7463ddf0c
HTTP/1.1 200 OK
date: Fri, 25 Jul 2025 13:55:51 GMT
server: uvicorn
content-length: 72
content-type: application/json

{"status":"waiting","missing":["pre","post","shp","shx"],"results":null}

❯ ./upload.sh
→ Uploading post (41150349 bytes) …
✔ post uploaded.
→ Uploading shp (332 bytes) …
✔ shp uploaded.
→ Uploading shx (108 bytes) …
✔ shx uploaded.
→ Uploading pre (24101265 bytes) …
✔ pre uploaded.

❯ curl -i -X GET http://192.168.1.26:8000/query_task/7a61d5b3-a254-42cd-8b5b-01b7463ddf0c
HTTP/1.1 200 OK
date: Fri, 25 Jul 2025 13:56:42 GMT
server: uvicorn
content-length: 48
content-type: application/json

{"status":"running","missing":[],"results":null}

❯ curl -i -X GET http://192.168.1.26:8000/query_task/7a61d5b3-a254-42cd-8b5b-01b7463ddf0c
HTTP/1.1 200 OK
date: Fri, 25 Jul 2025 14:00:57 GMT
server: uvicorn
content-length: 7239
content-type: application/json

{"status":"done","missing":null,"results":{"hulls":[{"id":"7a61d5b3-a254-42cd-8b5b-01b7463ddf0c_0","area":0.30064636883459195,"cut_volume":0.0038709541447043147,"fill_volume":0.0,"net_volume":-0.0038709541447043147,"image_key":"9fa9ef786097f1749b19d0ae3c77de70","image_url":"http://192.168.1.26:9000/uploads/9fa9ef786097f1749b19d0ae3c77de70?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=minioadmin%2F20250725%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250725T140058Z&X-Amz-Expires=900&X-Amz-SignedHeaders=host&X-Amz-Signature=9c5e20fb68efca5761d50b64730ce560375676ff6ba999dbbc3de1ae8922cba9"},{"id":"7a61d5b3-a254-42cd-8b5b-01b7463ddf0c_1","area":2.701144103821653,"cut_volume":0.19396142750065984,"fill_volume":0.3364764775808974,"net_volume":0.14251505008023754,"image_key":"4bf8de5291b31c4dbab2b244fb29f4b9","image_url":"http://192.168.1.26:9000/uploads/4bf8de5291b31c4dbab2b244fb29f4b9?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=minioadmin%2F20250725%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250725T140058Z&X-Amz-Expires=900&X-Amz-SignedHeaders=host&X-Amz-Signature=8e706ea771f970f873177a2f72ba1913c0a9ec51604f8a15ffc9f00de5af1000"},{"id":"7a61d5b3-a254-42cd-8b5b-01b7463ddf0c_2","area":0.09188220844474629,"cut_volume":0.043950532968600575,"fill_volume":0.027817577444576316,"net_volume":-0.01613295552402426,"image_key":"393b761ca4cd05b296f8ffbc908bb4ba","image_url":"http://192.168.1.26:9000/uploads/393b761ca4cd05b296f8ffbc908bb4ba?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=minioadmin%2F20250725%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250725T140058Z&X-Amz-Expires=900&X-Amz-SignedHeaders=host&X-Amz-Signature=ab93e5bd26157476082b7812c5f2e78719ae122950bdfcc2b1b553781f5bf075"},{"id":"7a61d5b3-a254-42cd-8b5b-01b7463ddf0c_3","area":0.6705980350873966,"cut_volume":0.2936417176440822,"fill_volume":0.22532493363850412,"net_volume":-0.06831678400557809,"image_key":"15746be53b4eecdf5565a8f80e530ecf","image_url":"http://192.168.1.26:9000/uploads/15746be53b4eecdf5565a8f80e530ecf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=minioadmin%2F20250725%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250725T140058Z&X-Amz-Expires=900&X-Amz-SignedHeaders=host&X-Amz-Signature=063fb3f12a6642d7f5e5b7b93b865d4e9788715d2eafb2dfac29f40ab712cd36"},{"id":"7a61d5b3-a254-42cd-8b5b-01b7463ddf0c_4","area":3.204014940918807,"cut_volume":2.4346130811329907,"fill_volume":3.212584397251544,"net_volume":0.7779713161185535,"image_key":"04f607978e13205c1da11ce532234812","image_url":"http://192.168.1.26:9000/uploads/04f607978e13205c1da11ce532234812?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=minioadmin%2F20250725%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250725T140058Z&X-Amz-Expires=900&X-Amz-SignedHeaders=host&X-Amz-Signature=91dbc8c42971203a1ded6ce78d03c67aae47b9bda84fc70940b646b2c0cbc2ab"},{"id":"7a61d5b3-a254-42cd-8b5b-01b7463ddf0c_5","area":0.2274761956749956,"cut_volume":0.00539187903658063,"fill_volume":0.0003743429565674603,"net_volume":-0.005017536080013169,"image_key":"df8849d9fe13762376c15f1f8d68272d","image_url":"http://192.168.1.26:9000/uploads/df8849d9fe13762376c15f1f8d68272d?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=minioadmin%2F20250725%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250725T140058Z&X-Amz-Expires=900&X-Amz-SignedHeaders=host&X-Amz-Signature=b4fd94ef14927a5c7813f32e67f7f36a7ddaaac21250aa259eaf8453335f63f7"},{"id":"7a61d5b3-a254-42cd-8b5b-01b7463ddf0c_6","area":0.09683621945220303,"cut_volume":0.0,"fill_volume":0.0,"net_volume":0.0,"image_key":"a973d5703a4e6d29763acb3088ac5c31","image_url":"http://192.168.1.26:9000/uploads/a973d5703a4e6d29763acb3088ac5c31?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=minioadmin%2F20250725%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250725T140058Z&X-Amz-Expires=900&X-Amz-SignedHeaders=host&X-Amz-Signature=af62da06d89f280edc73484197487289a46a8162d5d944620f84708586bcf080"},{"id":"7a61d5b3-a254-42cd-8b5b-01b7463ddf0c_7","area":0.17872697920702124,"cut_volume":0.006333242071276544,"fill_volume":0.0,"net_volume":-0.006333242071276544,"image_key":"518235fb3327161931043112b6f22ece","image_url":"http://192.168.1.26:9000/uploads/518235fb3327161931043112b6f22ece?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=minioadmin%2F20250725%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250725T140058Z&X-Amz-Expires=900&X-Amz-SignedHeaders=host&X-Amz-Signature=b83a11df03f3dd1da8be1f207cedab27717c51e030fe093ed796c528823bf589"},{"id":"7a61d5b3-a254-42cd-8b5b-01b7463ddf0c_9","area":0.23638903711108028,"cut_volume":0.07892455944673199,"fill_volume":0.24732417673106644,"net_volume":0.16839961728433445,"image_key":"ff15c5fe291e73f9f0b1eb2e78b6701d","image_url":"http://192.168.1.26:9000/uploads/ff15c5fe291e73f9f0b1eb2e78b6701d?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=minioadmin%2F20250725%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250725T140058Z&X-Amz-Expires=900&X-Amz-SignedHeaders=host&X-Amz-Signature=8ffe2154267ed878d365e136ada3fa7ba50b4f964e8d49ecd45ec63a61f4b127"},{"id":"7a61d5b3-a254-42cd-8b5b-01b7463ddf0c_10","area":0.3275448217991217,"cut_volume":0.014261002741385878,"fill_volume":0.006689550812503664,"net_volume":-0.007571451928882214,"image_key":"67a65e3b0a3313174091357a46dee039","image_url":"http://192.168.1.26:9000/uploads/67a65e3b0a3313174091357a46dee039?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=minioadmin%2F20250725%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250725T140058Z&X-Amz-Expires=900&X-Amz-SignedHeaders=host&X-Amz-Signature=e9cbd19209599855b042f886ed1558ae2ae563cae987fc7811e274ab7d216931"},{"id":"7a61d5b3-a254-42cd-8b5b-01b7463ddf0c_11","area":0.29761344599655415,"cut_volume":0.06938538598593004,"fill_volume":0.03309378197258002,"net_volume":-0.03629160401335002,"image_key":"dab9f3353da7ddf76f8bdabe84044513","image_url":"http://192.168.1.26:9000/uploads/dab9f3353da7ddf76f8bdabe84044513?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=minioadmin%2F20250725%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250725T140058Z&X-Amz-Expires=900&X-Amz-SignedHeaders=host&X-Amz-Signature=5554118094ca46e2edb0e21be2a6f2f48e4da965a7abbed2c8a13c649c01e8c0"},{"id":"7a61d5b3-a254-42cd-8b5b-01b7463ddf0c_12","area":0.0984716689060718,"cut_volume":0.022537509079291123,"fill_volume":0.09326440814727327,"net_volume":0.07072689906798214,"image_key":"5a4ef0dd49e4dc5a0ee2181dbceb1b48","image_url":"http://192.168.1.26:9000/uploads/5a4ef0dd49e4dc5a0ee2181dbceb1b48?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=minioadmin%2F20250725%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250725T140058Z&X-Amz-Expires=900&X-Amz-SignedHeaders=host&X-Amz-Signature=749c8b4d302fc6cc3762ea66477a68709941aff79add5ae14e4aa72d4a9b3212"},{"id":"7a61d5b3-a254-42cd-8b5b-01b7463ddf0c_13","area":0.5735939096791851,"cut_volume":0.03405167406629948,"fill_volume":0.11415696254841465,"net_volume":0.08010528848211518,"image_key":"8b1abc93276f0783fbb37e1f207fe26d","image_url":"http://192.168.1.26:9000/uploads/8b1abc93276f0783fbb37e1f207fe26d?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=minioadmin%2F20250725%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250725T140058Z&X-Amz-Expires=900&X-Amz-SignedHeaders=host&X-Amz-Signature=e2d19b1c423e5957334b6c3f539c941714dcbaf88f1baa848335335a60f42204"}]}}
```

