# 账本修复后的历史回归重放

使用已合并的会计修复提交`36ec2309f0beed5773a78a64f1fd1990b6fa3f08`，对既有配置和公开数据快照重新运行全部12组对照。输入、配置和结果的SHA-256见[receipt.json](receipt.json)，完整结果见[model_comparison.csv](model_comparison.csv)。旧表保留，没有覆盖原始行情或修改参数选取赢家。

```powershell
python -m quant_timing compare --config configs/market_comparison.yaml --out NEW_OUTPUT
```

12组均完成10个滚动测试折，历史截断泄漏检查通过。11组的平均超额收益为负；唯一为正的沪深300仓位TSMOM平均超额为0.5944%，但其描述性全样本净收益为12.5961%，低于基准18.6785%，最大回撤为22.4995%。这些数值不支持把软件修复解释为策略有效性证明。

本次使用相同的2019-03-08至2026-09-28历史窗口，是修复后的兼容性与会计回归，不能增加未触碰的样本外天数，也不是前向观察。信号、行业权重、上市前IM替代以及成交量口径的限制仍以[来源说明](../../../data/public/SOURCE.md)为准；真实容量和参与率仍不可用。
