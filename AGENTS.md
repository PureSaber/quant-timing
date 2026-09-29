# quant-timing

仓位择时和风格择时研究层。

```powershell
python -m pip install -e ".[dev]"
python -m quant_timing.synthetic
python -m quant_timing run --config configs/combined.yaml --out outputs/demo
python -m pytest -q
python -m ruff check src tests
```

下游只读取发布出的 `position_scale`。不要把研究订单当成已发送委托。
