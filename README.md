# Backend

FastAPI service and reusable Python package for Cumulus live here.

## Run locally

From the workspace root:

```powershell
python -m pip install -e .\backend[dev]
powershell -File .\backend\scripts\start-backend-local.ps1
```

The backend reads runtime config from `backend/configs` and discovers ML data and model artifacts from `ml/data` by default.
