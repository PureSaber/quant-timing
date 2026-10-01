# quant-timing

指数与风格组合的研究层。它做两件事：

- 仓位择时：用已经走完的波动和窗口收益决定股票预算，其余放在现金。
- 风格择时：在大小盘、价值成长这两对风格之间做多头倾斜。

交易对象是指数和风格袖套，不是个股。规则写在配置里，验证段不搜索参数。样本外对照没有通过时，不会写出可供下游读取的 `position_scale.json`。

这不是价格预测，也不会下单。订单文件里的状态固定为 `target`。

## 时点

决策日收盘可以使用当天及更早的价格。这个权重赚的是下一根 K 线的收益，调仓成本也记在下一根 K 线上。信号不足时仓位是现金，不会外推成满仓。

完整约定和一笔三天的数字例子见 [研究契约](docs/RESEARCH_CONTRACT.md)。

## 运行

```powershell
python -m pip install -e ".[dev]"
python -m quant_timing.synthetic
python -m quant_timing run --config configs/combined.yaml --out outputs/demo
python -m pytest -q
python -m ruff check src tests
```

`configs/position_only.yaml` 只做仓位择时，股票预算全部留在市场袖套。

## 输出

| 文件 | 用途 |
| --- | --- |
| `position_scale.json` | 最新总仓位。只有审计通过、并且至少有一折打出了收益时才写出。`quant-paper-sim` 现有读取逻辑只使用其中的 `position_scale`。 |
| `portfolio_overlay.yaml` | 把同一个系数抄到已有的 `quant-portfolio` 策略条目上。 |
| `style_weights.csv` | 每期的风格权重和现金。 |
| `position_history.csv` | 模型仓位、封顶后的仓位和特征。 |
| `decision.json` | 含 `use`、`hold_previous` 或 `blocked` 的完整决定。 |
| `validation/` | 走步折、泄漏审计和摘要。全样本收益只作描述。 |
| `standard/` | 与研究运行契约一致的收益、持仓、目标订单、成本和暴露。 |

持仓文件记录收盘调仓前的实际数量、金额和权重，`return_weight`用于上一笔决策的收益归因。期货和保证金使用各自的估值定义。所有产物验证成功后整体发布；导出失败不会提前发布仓位文件，非空输出目录拒绝覆盖。

```mermaid
flowchart LR
  prices[指数与风格价格] --> study[quant-timing]
  regime[quant-regime 的 position_scale] --> study
  macro[宏观上下文] --> study
  study --> scale[position_scale.json]
  study --> weights[风格权重]
  scale --> paper[quant-paper-sim 现有读取]
  scale --> portfolio[quant-portfolio 配置项]
```

下游接入时不改它们的接口。模拟配置继续指向一个含 `position_scale` 的 JSON。组合配置继续在策略条目上填写 `position_scale`。

仓位模型可以是原来的三档规则，也可以是波动率目标、时间序列动量或均线趋势。风格可以按动量、估值差、盈利修正或拥挤度倾斜，并限制相对等权的偏离。现金可以用收益率或债券价格的涨跌，股票预算也可以用股指期货调到目标 beta。单日调仓幅度和回撤线在决策前生效。冲击成本和固定费用一起扣在下一根 K 线上。

真实指数对照：

```powershell
python tools/fetch_public_indices.py --futures-only --offline
python -m quant_timing compare --config configs/market_comparison.yaml --out studies/market_comparison
```

账本修复后的十二组公开对照见[2026-10-01回归重放](studies/market_comparison/2026-10-01-regression/README.md)，原表保留为历史结果。所有模型统一从2019-03-08开始，使用同一份截至2026-09-28的既有快照；这次重放不是新增样本外证据。期货序列可由已提交的单合约收盘价和独立交易日历离线重建，上市前及数据覆盖前不会生成价格。公开成交量只用于行业相对权重，不推算人民币容量；容量及参与率保持空值。完整数据口径见`data/public/SOURCE.md`。这些固定参数下的研究结果不构成收益承诺。

外部 regime 有两种读法，不能同时用：

- `regime.snapshot`：只封顶最新一次发布，不回放成历史。
- `regime.history`：按决策日之前最近一次发布的系数封顶整条回测。

宏观序列不完整时必须声明 `baseline` 或 `hold_previous`。上下文完整但没有显式规则时，不改变仓位。

## 边界

固定规则的对照结果不是收益承诺。仓库不修改 `quant-regime`、`quant-portfolio` 或 `quant-paper-sim`。
