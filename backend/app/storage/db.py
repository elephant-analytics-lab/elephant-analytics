import sqlite3
from datetime import datetime, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "data" / "elephant_analytics.db"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                video_path TEXT NOT NULL,
                status TEXT NOT NULL,
                message TEXT,
                model_key TEXT,
                pipeline_mode TEXT,
                gpx_path TEXT,
                processing_started_at TEXT,
                processing_completed_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS gps_points (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                lat REAL NOT NULL,
                lon REAL NOT NULL,
                ts TEXT,
                FOREIGN KEY(job_id) REFERENCES jobs(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                frame_index INTEGER NOT NULL,
                track_id INTEGER,
                cls_id INTEGER NOT NULL,
                cls_label TEXT,
                conf REAL NOT NULL,
                x1 REAL NOT NULL,
                y1 REAL NOT NULL,
                x2 REAL NOT NULL,
                y2 REAL NOT NULL,
                FOREIGN KEY(job_id) REFERENCES jobs(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS habitat_frames (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                frame_index INTEGER NOT NULL,
                habitat_label TEXT NOT NULL,
                conf REAL NOT NULL,
                FOREIGN KEY(job_id) REFERENCES jobs(id)
            )
            """
        )
        job_cols = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)")}
        if "processing_started_at" not in job_cols:
            conn.execute("ALTER TABLE jobs ADD COLUMN processing_started_at TEXT")
        if "processing_completed_at" not in job_cols:
            conn.execute("ALTER TABLE jobs ADD COLUMN processing_completed_at TEXT")
        if "model_key" not in job_cols:
            conn.execute("ALTER TABLE jobs ADD COLUMN model_key TEXT")
        if "pipeline_mode" not in job_cols:
            conn.execute("ALTER TABLE jobs ADD COLUMN pipeline_mode TEXT")
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(detections)")}
        if "track_id" not in cols:
            conn.execute("ALTER TABLE detections ADD COLUMN track_id INTEGER")
            conn.execute("UPDATE detections SET track_id = 0 WHERE track_id IS NULL")


def create_job(job_id: str, filename: str, video_path: str) -> None:
    now = _utc_now()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO jobs (id, filename, video_path, status, message, gpx_path, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (job_id, filename, video_path, "uploaded", None, None, now, now),
        )


def update_job(
    job_id: str,
    status: str,
    message: str | None = None,
    gpx_path: str | None = None,
    model_key: str | None = None,
    pipeline_mode: str | None = None,
) -> None:
    now = _utc_now()
    with _connect() as conn:
        conn.execute(
            """
            UPDATE jobs
            SET status = ?,
                message = ?,
                model_key = COALESCE(?, model_key),
                pipeline_mode = COALESCE(?, pipeline_mode),
                gpx_path = COALESCE(?, gpx_path),
                processing_started_at = CASE
                    WHEN ? = 'processing' THEN ?
                    ELSE processing_started_at
                END,
                processing_completed_at = CASE
                    WHEN ? IN ('complete','failed','cancelled') THEN ?
                    ELSE processing_completed_at
                END,
                updated_at = ?
            WHERE id = ?
            """,
            (
                status,
                message,
                model_key,
                pipeline_mode,
                gpx_path,
                status,
                now,
                status,
                now,
                now,
                job_id,
            ),
        )


def clear_job_data(job_id: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM gps_points WHERE job_id = ?", (job_id,))
        conn.execute("DELETE FROM detections WHERE job_id = ?", (job_id,))
        conn.execute("DELETE FROM habitat_frames WHERE job_id = ?", (job_id,))


def insert_gps_points(job_id: str, points: list[tuple[float, float, str | None]]) -> None:
    if not points:
        return
    with _connect() as conn:
        conn.executemany(
            "INSERT INTO gps_points (job_id, lat, lon, ts) VALUES (?, ?, ?, ?)",
            [(job_id, lat, lon, ts) for lat, lon, ts in points],
        )


def insert_detections(job_id: str, detections: list[dict]) -> None:
    if not detections:
        return
    rows = [
        (
            job_id,
            det["frame_index"],
            det.get("track_id"),
            det["cls_id"],
            det.get("cls_label"),
            det["conf"],
            det["x1"],
            det["y1"],
            det["x2"],
            det["y2"],
        )
        for det in detections
    ]
    with _connect() as conn:
        conn.executemany(
            """
            INSERT INTO detections
            (job_id, frame_index, track_id, cls_id, cls_label, conf, x1, y1, x2, y2)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def insert_habitat_rows(job_id: str, rows: list[dict]) -> None:
    if not rows:
        return
    with _connect() as conn:
        conn.execute("DELETE FROM habitat_frames WHERE job_id = ?", (job_id,))
        conn.executemany(
            """
            INSERT INTO habitat_frames (job_id, frame_index, habitat_label, conf)
            VALUES (?, ?, ?, ?)
            """,
            [
                (job_id, int(r["frame_index"]), str(r["habitat_label"]), float(r["conf"]))
                for r in rows
            ],
        )


def get_jobs() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, filename, status, message,
                   model_key, pipeline_mode,
                   processing_started_at, processing_completed_at,
                   created_at, updated_at
            FROM jobs
            ORDER BY created_at DESC
            """
        ).fetchall()
    jobs = []
    for row in rows:
        item = dict(row)
        duration_seconds = None
        try:
            start = item.get("processing_started_at")
            end = item.get("processing_completed_at")
            if start and end:
                duration_seconds = (
                    datetime.fromisoformat(end) - datetime.fromisoformat(start)
                ).total_seconds()
        except Exception:
            duration_seconds = None
        item["duration_seconds"] = duration_seconds
        jobs.append(item)
    return jobs


def get_job(job_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, filename, video_path, status, message, model_key, pipeline_mode, gpx_path, created_at, updated_at FROM jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
    return dict(row) if row else None


def get_gps_points(job_id: str, limit: int | None = None) -> list[dict]:
    sql = "SELECT lat, lon, ts FROM gps_points WHERE job_id = ? ORDER BY id"
    params: list[object] = [job_id]
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    with _connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def get_detection_count(job_id: str) -> int:
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM detections WHERE job_id = ?",
            (job_id,),
        ).fetchone()
    return int(row["cnt"]) if row else 0


def get_unique_track_count(job_id: str) -> int:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT COUNT(DISTINCT track_id) AS cnt
            FROM detections
            WHERE job_id = ? AND track_id IS NOT NULL AND track_id != 0
            """,
            (job_id,),
        ).fetchone()
    return int(row["cnt"]) if row else 0


def get_stable_track_count(
    job_id: str,
    min_samples: int = 3,
    min_avg_conf: float = 0.30,
) -> int:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS cnt
            FROM (
                SELECT track_id
                FROM detections
                WHERE job_id = ?
                  AND track_id IS NOT NULL
                  AND track_id != 0
                GROUP BY track_id
                HAVING COUNT(*) >= ?
                   AND AVG(conf) >= ?
            ) t
            """,
            (job_id, min_samples, min_avg_conf),
        ).fetchone()
    return int(row["cnt"]) if row else 0


def get_detection_series(job_id: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT frame_index, COUNT(*) AS herd_size
            FROM detections
            WHERE job_id = ?
            GROUP BY frame_index
            ORDER BY frame_index
            """,
            (job_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_class_distribution(job_id: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT COALESCE(cls_label, 'Unknown') AS cls_label, COUNT(*) AS cnt
            FROM detections
            WHERE job_id = ?
            GROUP BY COALESCE(cls_label, 'Unknown')
            ORDER BY cnt DESC
            """,
            (job_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_detections(job_id: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT frame_index, track_id, cls_id, cls_label, conf, x1, y1, x2, y2
            FROM detections
            WHERE job_id = ?
            ORDER BY frame_index, track_id
            """,
            (job_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_timeline(job_id: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT
                COALESCE(track_id, 0) AS elephant_id,
                MIN(frame_index) AS first_seen,
                MAX(frame_index) AS last_seen,
                (MAX(frame_index) - MIN(frame_index)) AS duration_frames
            FROM detections
            WHERE job_id = ?
            GROUP BY COALESCE(track_id, 0)
            ORDER BY COALESCE(track_id, 0)
            """,
            (job_id,),
        ).fetchall()
        class_rows = conn.execute(
            """
            SELECT
                COALESCE(track_id, 0) AS elephant_id,
                COALESCE(cls_label, 'Unknown') AS age_class,
                COUNT(*) AS cnt
            FROM detections
            WHERE job_id = ?
            GROUP BY COALESCE(track_id, 0), COALESCE(cls_label, 'Unknown')
            """,
            (job_id,),
        ).fetchall()
    best_class = {}
    for row in class_rows:
        key = row["elephant_id"]
        if key not in best_class or row["cnt"] > best_class[key]["cnt"]:
            best_class[key] = {"age_class": row["age_class"], "cnt": row["cnt"]}
    enriched = []
    for row in rows:
        item = dict(row)
        item["age_class"] = best_class.get(item["elephant_id"], {}).get("age_class", "Unknown")
        enriched.append(item)
    return enriched


def clear_history() -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM habitat_frames")
        conn.execute("DELETE FROM gps_points")
        conn.execute("DELETE FROM detections")
        conn.execute("DELETE FROM jobs")


def get_jobs_minimal(job_ids: list[str]) -> list[dict]:
    if not job_ids:
        return []
    placeholders = ",".join("?" for _ in job_ids)
    with _connect() as conn:
        rows = conn.execute(
            f"""
            SELECT id, video_path, gpx_path
            FROM jobs
            WHERE id IN ({placeholders})
            """,
            job_ids,
        ).fetchall()
    return [dict(row) for row in rows]


def get_habitat_distribution(job_id: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT habitat_label, COUNT(*) AS cnt, AVG(conf) AS avg_conf
            FROM habitat_frames
            WHERE job_id = ?
            GROUP BY habitat_label
            ORDER BY cnt DESC
            """,
            (job_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_habitat_rows(job_id: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT frame_index, habitat_label, conf
            FROM habitat_frames
            WHERE job_id = ?
            ORDER BY frame_index
            """,
            (job_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_habitat_timeline(job_id: str) -> list[dict]:
    rows = get_habitat_rows(job_id)
    if not rows:
        return []

    segments: list[dict] = []
    start = int(rows[0]["frame_index"])
    end = int(rows[0]["frame_index"])
    label = str(rows[0]["habitat_label"])
    conf_sum = float(rows[0]["conf"])
    count = 1

    for row in rows[1:]:
        frame = int(row["frame_index"])
        habitat = str(row["habitat_label"])
        conf = float(row["conf"])
        if habitat == label:
            end = frame
            conf_sum += conf
            count += 1
            continue
        segments.append(
            {
                "start": start,
                "end": end,
                "habitat": label,
                "confidence": conf_sum / max(count, 1),
                "samples": count,
            }
        )
        start = frame
        end = frame
        label = habitat
        conf_sum = conf
        count = 1

    segments.append(
        {
            "start": start,
            "end": end,
            "habitat": label,
            "confidence": conf_sum / max(count, 1),
            "samples": count,
        }
    )
    return segments


def delete_jobs(job_ids: list[str]) -> int:
    if not job_ids:
        return 0
    placeholders = ",".join("?" for _ in job_ids)
    with _connect() as conn:
        conn.execute(
            f"DELETE FROM habitat_frames WHERE job_id IN ({placeholders})",
            job_ids,
        )
        conn.execute(
            f"DELETE FROM gps_points WHERE job_id IN ({placeholders})",
            job_ids,
        )
        conn.execute(
            f"DELETE FROM detections WHERE job_id IN ({placeholders})",
            job_ids,
        )
        cur = conn.execute(
            f"DELETE FROM jobs WHERE id IN ({placeholders})",
            job_ids,
        )
    return int(cur.rowcount or 0)
