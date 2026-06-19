# Importing the Ignition Perspective Dashboard

The dashboard is shipped as **exported artifacts** so anyone with an Ignition gateway can
stand it up — no Docker, no build step. Everything below is optional: if you just want to
*see* the dashboard, the [screenshots in the main README](../README.md) show all seven views.

## What's in `exports/`

| File | What it is | Use it when |
| --- | --- | --- |
| `surface_defect_project.zip` | The Perspective project (views + named queries) | You already have a gateway and a MySQL connection |
| `surface_defect_gateway.gwbk` | A full gateway backup (project **+** the `SurfaceDefect` DB connection) | You want the whole thing restored in one shot |
| `surfaceDefect.sql` | `mysqldump` of the database — schema **and** data (854 inspections, predictions, the synthetic ops layer) | Always — this is the data the views read |

## Prerequisites

- **Ignition 8.1** gateway (the free download runs in 2-hour trial mode, which is fine — Perspective
  just pauses every ~2h; log in to resume).
- **MySQL 8** (or MariaDB) reachable from the gateway.
- A MySQL JDBC driver on the gateway. The project was built against the bundled **MariaDB** driver.

## 1. Load the data

```bash
mysql -u root -p < exports/surfaceDefect.sql
```

This creates the `surfaceDefect` database with the full star schema and all rows. Verify:

```sql
SELECT modelName, metricValue FROM surfaceDefect.metricOverall WHERE metricName='accuracy';
-- resnet18 -> 0.978923,  vit_b_16 -> 0.765808
```

## 2. Get the project onto the gateway

### Option A — Import the project (recommended)

1. In the gateway web UI: **Config → Projects → Import Project**.
2. Choose `exports/surface_defect_project.zip`, name it `surface_defect`.
3. Create the database connection the named queries expect:
   **Config → Databases → Connections → Create new** →
   - **Name:** `SurfaceDefect` (exact — the named queries reference it by name)
   - **Driver:** MariaDB
   - **Connect URL:** `jdbc:mariadb://<your-mysql-host>:3306/surfaceDefect`
   - **Username / Password:** your MySQL credentials

### Option B — Restore the full backup

1. **Config → System → Backup/Restore → Restore**, choose `surface_defect_gateway.gwbk`.
2. The `SurfaceDefect` connection comes with it, but its URL points at the Docker hostname
   `mysql`. Edit it under **Config → Databases → Connections** to your MySQL host (see Option A).

## 3. Open the dashboard

```
http://<gateway-host>:8088/data/perspective/client/surface_defect
```

(In trial mode you'll see Ignition's trial screen first — **Launch Gateway → log in → reopen**.)

The KPI tiles should read **97.9% / 0.978 / 0.979 / 854**. If they show errors, the
`SurfaceDefect` connection name or URL is off — re-check step 2.

## 4. (Optional) Chip images for the Chip Viewer

The **Chip Viewer** view references the cropped defect PNGs by URL (`http://localhost:8090/chips/...`).
Serve the chips from the repo's `results/chips/` with any static server:

```bash
cd ../results        # repo's results/ dir (contains chips/)
python3 -m http.server 8090
```

Without this the other six views are unaffected — only the five featured chip thumbnails go blank.

## 5. (Optional) Live Inspection feed

The **Live Inspection** view polls a `liveInspection` table every 2s. The dump already includes a
populated snapshot, so the view shows data out of the box. To *stream* fresh rows, run the replay:

```bash
cd etl
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
MYSQL_HOST=127.0.0.1 MYSQL_USER=root MYSQL_PASSWORD=<pw> .venv/bin/python replay.py
```

---

## How the data was built (source, not required to import)

- `mysql/01_schema.sql` — the star schema (readable form of what's in the dump).
- `etl/load_mysql.py` — loads `results/<model>/*.csv` (from `train.py`) into the schema.
- `etl/enrich.py` — generates the deterministic synthetic factory-ops layer
  (machine / operator / shift / timestamp / strip position / criticality / cost).
- `etl/replay.py` — samples inspections into `liveInspection` for the live feed.
- `ignition/projects/surface_defect/` — the version-controlled project source
  (Perspective `view.json` files + named queries) that `surface_defect_project.zip` is built from.
- `ignition/named_queries.sql` — all named queries collected in one readable file.
