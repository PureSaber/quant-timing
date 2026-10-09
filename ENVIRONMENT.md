# Reproducible Windows research environment

The checked-in lock targets CPython 3.12 on Windows. Other Python/platform combinations need their own resolved and tested lock. The Git dependency, where present, is pinned to an immutable commit. The project itself must also be checked out at the desired recorded commit.

```powershell
powershell -File tools/Install-Locked.ps1 -Python C:/path/to/python.exe -Environment D:/new-environment
D:/new-environment/Scripts/python.exe -m pytest -q
D:/new-environment/Scripts/python.exe -m ruff check src tests
```

Install into a new directory; do not overlay an existing research environment. Update this lock only in a dedicated environment, rebuild from the proposed lock, run pip check and the complete tests, and commit the lock with its validation record. Do not copy editable local paths into the dependency lock.
