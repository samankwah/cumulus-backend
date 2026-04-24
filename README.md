# Backend

FastAPI service and reusable Python package for Cumulus live here.

## Run locally

From the workspace root:

```powershell
python -m pip install -e .\backend[dev]
powershell -File .\backend\scripts\start-backend-local.ps1
```

The backend reads runtime config from `backend/configs` and discovers training data and model artifacts from `training/data` by default.
