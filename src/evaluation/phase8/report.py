"""Readable interpretation of the frozen evaluation, without strategy selection."""
import pandas as pd
from .common import PRIMARY,STRATEGIES,FEES


def _table(rows: list[list],headers: list[str]) -> str:
    """Render Markdown directly; no optional tabulate dependency."""
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
                     ['| '+' | '.join(str(x) for x in row)+' |' for row in rows])


def render(t: dict[str,pd.DataFrame]) -> str:
    """Interpret only saved account-derived tables and user-supplied prior context."""
    p=t['performance_metrics'];main=p[p.fee_rate==PRIMARY].set_index('strategy')
    annual=t['annual_metrics'];a=annual[annual.fee_rate==PRIMARY]
    costs=t['cost_attribution'];costs=costs[costs.fee_rate==PRIMARY].set_index('strategy')
    ex=t['exposure_analysis'];ex=ex[ex.fee_rate==PRIMARY].set_index('strategy')
    pct=lambda x:f'{x:.2%}'
    money=lambda x:f'{x:,.2f}'
    parts=['# Phase 8：冻结策略绩效评价与解释',
    '本报告只评价 Phase 7 首次发布的12个冻结账户。没有重算策略路径、训练模型、修改阈值、资产池、费用或执行逻辑。2026仍封存；不进入后续阶段。',
    '结论等级：**Level 1——存在局部/弱历史经济信号，但缺乏稳定性。** A在主成本下同时改善收益与若干风险指标，但2025年明显落后，较高费用下全期优势反转；B的较低风险没有转换成主成本下的超额总收益。此等级是描述性证据判断，不是模型排名或下一步选优规则。',
    '## 口径与冻结边界',
    '评价区间为2022-01-04至2025-12-31，共969个境内会话，每账户初始资金100000元。A=单资产逻辑回归分类，B=汇总随机森林回归，C=可投资ETF等权基准。沿用原冻结规则：A概率严格>0.55；B预测5日收益严格>0.001；全局每5个会话调仓，锚点2022-01-04，跨年不重置。主成本为单边fee=0.0003，所有情景slippage=0.0002。',
    '所有日收益来自冻结NAV（Net Asset Value，资产净值）：r[t]=NAV[t]/previous_NAV[t]-1。首日现金会话的零收益也计入N。CAGR（Compound Annual Growth Rate，复合年增长率）=(期末/期初)^(252/N)-1；年化波动率=日收益样本标准差(ddof=1)×√252；Sharpe Ratio（夏普比率）=日均收益/日收益样本标准差×√252，risk-free rate（无风险利率）=0。常数/不足样本的未定义指标留空并附原因，不填造分数。',
    '回撤曲线=NAV/截至当日最高NAV-1，为非正数；Maximum Drawdown（最大回撤）表列正的损失幅度，因此A-C回撤差为负代表A回撤较浅。包含初始资本作为虚拟前一观察；相同高点取下跌前最后一个，相同谷底取第一个；首次回到峰值为恢复。持续期=峰值至恢复/窗口末的会话间隔，同时保存日历天数及水下会话数。未恢复标记unrecovered，绝不猜未来恢复日期。年度回撤单独重置到年初余额。',
    '年度首个收益使用上一年最后NAV作分母，2022只代表1月4日起可用策略区间。年度收益差不能相加作为全期复合优势；年度人民币财富差增量可以相加，属于算术分解，不是因果alpha（超额能力）归因。',
    '## 主成本账户：收益与风险一起看',
    _table([[s,money(main.loc[s,'terminal_nav']),pct(main.loc[s,'total_return']),pct(main.loc[s,'cagr']),
             pct(main.loc[s,'annualized_volatility']),f"{main.loc[s,'sharpe_ratio']:.3f}",pct(main.loc[s,'maximum_drawdown'])] for s in STRATEGIES],
           ['策略','期末NAV/元','总收益','CAGR','年化波动','Sharpe','最大回撤损失']),
    '这同时回答A的几个问题：主成本下A的CAGR、Sharpe均高于C，最大回撤更小。但A持有更多现金，不能将这些差异全部归因于模型选择能力。B的波动、回撤更低，但总收益和CAGR低于C。',
    '## 年度稳定性',
    _table([[r.year,r.strategy,pct(r.total_return),pct(r.annualized_volatility),f'{r.sharpe_ratio:.3f}',pct(r.maximum_drawdown),pct(r.average_gross_exposure)]
            for r in a.sort_values(['year','strategy']).itertuples()],['年度','策略','年度收益','年化波动','Sharpe','最大回撤','平均暴露'])]
    y=t['yearly_strategy_comparison'];y=y[y.fee_rate==PRIMARY]
    parts += [_table([[r.year,r.strategy+'-C',f'{100*r.total_return_difference:+.2f}个百分点',money(r.wealth_gap_change)]
                      for r in y.sort_values(['year','strategy']).itertuples()],['年度','比较','年度收益差','财富差年度增量/元']),
    'A在2022、2023、2024三个年度高于C，2025明显低于C，优势不是每年都有。2024是最大正向人民币财富差贡献年，2025回吐的差额比2024新增优势还大。年度财富差贡献可能超过最终差额的100%，因为其他年份抵消；不能把2024简单解释为最终优势的独立因果来源。B在2022–2023领先、2024–2025落后，方向同样不稳定。完整四费用年度比较见yearly_strategy_comparison.csv。',
    '## 四个冻结成本情景',
    _table([[r.strategy,pct(float(r.fee_rate)),money(r.terminal_nav),pct(r.total_return),pct(r.cagr),f'{r.sharpe_ratio:.3f}',pct(r.maximum_drawdown),money(r.total_direct_trading_friction)]
            for r in p.itertuples()],['策略','单边fee','期末NAV','总收益','CAGR','Sharpe','最大回撤','直接摩擦/元'])]
    b=t['benchmark_comparison']
    parts += [_table([[r.strategy+'-C',pct(float(r.fee_rate)),f'{100*r.total_return_difference:+.2f}个百分点',
                       f'{100*r.cagr_difference:+.2f}个百分点',f'{r.sharpe_ratio_difference:+.3f}',
                       f'{100*r.maximum_drawdown_difference:+.2f}个百分点'] for r in b.itertuples()],
                     ['比较','单边fee','总收益差','CAGR差','Sharpe差','回撤损失差']),
    'A在0、0.0003、0.0005费用下期末高于C，在0.001下低于C。B在fee=0（仍有滑点）时只比C多约26元，这个差额很普通，不能视为强优势；主成本及更高费用下落后C。没有求解盈亏平衡/最优费率，没有按结果更改B的0.001准入门槛。',
    '## 交易摩擦与账户路径',
    _table([[s,money(costs.loc[s,'explicit_fees']),money(costs.loc[s,'slippage_cost']),f"{costs.loc[s,'cumulative_turnover']:.3f}",
             pct(costs.loc[s,'friction_over_initial_capital']),money(costs.loc[s,'same_fill_gross_terminal_value']),pct(costs.loc[s,'friction_over_same_fill_gross_value'])] for s in STRATEGIES],
            ['策略','手续费/元','滑点/元','累计换手/倍NAV','摩擦/初始资金','同成交gross/元','摩擦/同成交gross']),
    'A换手最高，其次B，C最低。换手是每日实际成交金额/pre-trade NAV之和；买卖各计实际成交金额，不是年化换手。direct friction（直接摩擦）=explicit fees（显式费用）+slippage cost（滑点成本）。',
    'same-fill gross account value（同一成交路径的费用加回账面值）=net NAV+累计手续费+累计滑点，仅是会计桥梁，**不是另一套无费用策略的反事实收益**。不同费用会改变现金、可负担的整数份额和后续成交路径，因此不同账户终值差不能全部解释为直接手续费。高换手让A/B相对低换手C更脆弱。',
    '## 市场暴露、现金与权重漂移',
    _table([[s,pct(ex.loc[s,'average_gross_exposure']),pct(ex.loc[s,'average_cash_ratio']),pct(ex.loc[s,'max_gross_exposure']),
             pct(ex.loc[s,'max_actual_asset_weight']),pct(ex.loc[s,'max_absolute_weight_drift']),f"{ex.loc[s,'average_holding_count']:.3f}",int(ex.loc[s,'sessions_invested'])] for s in STRATEGIES],
            ['策略','平均暴露','平均现金','最大暴露','最大实际单资产权重','最大绝对权重漂移','平均持仓数','有持仓会话']),
    '现金比率使用账本cash，包含未解锁卖出款，不能全部视为当日可买入力；应收分红是另一资产类别，所以平均现金与持仓暴露之和可能略低于100%。目标单资产25%上限与市场变动后的实际权重不同。漂移参考最近一次已到下一执行会话的冻结目标，未成交、余量和价格变动均可能造成偏离，不自动触发再平衡。',
    _table([[r.strategy,r.held_assets,r.sessions,pct(r.fraction)] for r in t['holding_count_distribution'][t['holding_count_distribution'].fee_rate==PRIMARY].itertuples()],
            ['策略','实际持有ETF数','会话数','会话占比']),
    'A平均约72%暴露、28%现金；B约85%暴露、15%现金；C接近满仓。较低暴露可以同时降低收益与风险。未运行额外风险匹配策略或回归归因，不能声称A/B有经过验证的风险调整alpha。',
    '## 预测覆盖是主结论的限制',
    'A/B各有1552个“资产×调仓日”，覆盖结构相同：各有1355条预测，覆盖87.31%。各197条缺预测中，196被记为missing_prediction零目标；另1条同时不可交易，主要原因记为not_tradable，不能重复计数。A因阈值产生654个零目标，B为370。C不依赖预测覆盖，不能用A/B缺预测删去C的资产。']
    cov=t['coverage_effect_summary']
    for grouping,title,key in [('year','年度预测覆盖','year'),('asset','每资产预测覆盖','asset_code')]:
        rows=cov[(cov.strategy=='A')&(cov.grouping==grouping)]
        parts += ['### '+title+'（A/B覆盖相同）',_table([[getattr(r,key),int(r.predictions_available),int(r.observations),pct(r.prediction_coverage)] for r in rows.itertuples()],
             ['分组','有预测','应评估调仓点','覆盖率'])]
    parts += ['缺预测的互斥主原因：特征不完整130条、588000/2022训练历史不足49条、事件/间隔标签无效10条、开发期末标签跨界7条，另不可交易1条。这些继承了历史监督学习样本资格，不等价于真实部署时模型绝对无法预测。',
    '2025-12-26最后调仓日，A/B的8只ETF全部缺预测，且8条均存在开发期末未来5日标签越界；其中511010也特征不完整，所以互斥主原因中只有7条归到标签越界。冻结规则产生全零目标，12月29日清掉剩余持仓，最后3个会话持有现金。这不是模型成功预见未来或主动避险，也不是期末强制平仓。未补预测、未插值、未使用2026。',
    '**本实验是基于冻结历史OOS（Out-of-Sample，样本外）prediction coverage（预测覆盖）的开发期策略研究，不是完整的Point-in-Time live deployment simulation（历史时点实时部署模拟）。** 未做新的补预测策略，因此这里只能解释覆盖限制，不能量化假想完整覆盖会多赚/少赚多少。',
    '## 最大回撤与主要水下区间',
    _table([[s,pct(main.loc[s,'maximum_drawdown']),main.loc[s,'drawdown_peak_date'],main.loc[s,'drawdown_trough_date'],
             main.loc[s,'recovery_date'] or '窗口内未恢复',int(main.loc[s,'drawdown_duration_sessions']),int(main.loc[s,'longest_drawdown_duration_sessions'])] for s in STRATEGIES],
            ['策略','最大损失','峰值日','谷底日','恢复日','该回撤持续会话','最长持续会话'])]
    ep=t['drawdown_episodes'];ep=ep[ep.fee_rate==PRIMARY]
    major=pd.concat([ep[ep.strategy==s].sort_values(['maximum_drawdown','peak_date'],ascending=[False,True]).head(3) for s in STRATEGIES])
    parts += [_table([[r.strategy,r.peak_date,r.trough_date,r.recovery_date or '窗口内未恢复',pct(r.maximum_drawdown),r.duration_sessions,r.recovery_status] for r in major.itertuples()],
           ['策略','峰值','谷底','恢复','损失幅度','持续会话','状态']),
    '上表按每策略损失最深的三个区间做风险展示，不用来排名或选择策略；全部回撤区间与未恢复状态见drawdown_episodes.csv。日期持续期同时有会话与日历口径，最长回撤包括窗口末尚未恢复的区间。',
    '## 四层证据解释',
    '### A. Predictive evidence（预测证据）',
    'Phase5.5已经看过2022–2025开发期预测质量；据已验收上下文，B的预测误差相对稳定。本阶段不重新读取预测、标签或模型，不重算预测指标，也不从当前投资收益倒推预测能力。A/B模型族是开发阶段选择，因此当前策略收益不是完全未见的最终检验。',
    '### B. Economic evidence（历史经济证据）',
    '冻结的预测→准入→等权→实际成交链条下，A主成本账户相对C有正的历史经济差额，并且CAGR、Sharpe与回撤指标也较好；这值得继续严谨研究，但不是全面成功。B主成本期末落后C，较低风险不能替代超额收益证据。',
    'MAE/RMSE（Mean Absolute Error / Root Mean Squared Error，平均绝对误差/均方根误差）改善衡量全部预测的平均误差，未必意味着超过0.001门槛的子集有更强可交易排序。B采用二元准入后等权，超过门槛的预测大小不再连续影响仓位，可能丢失误差改善所含的信息。预测未来收盘到收盘收益与下一开盘实际成交也不是同一收益定义；费用、现金、取消订单、权益事件和预测覆盖都会影响转换。以上是机制解释，不是已证明的单一亏损原因，不据此修改规则。',
    '### C. Robustness（稳定性）',
    'A优势跨若干年份和较低成本存在，但2025明显反向且高费用下消失；B年度方向反复，fee=0时也几乎没有总收益优势。高换手是A/B的重要脆弱性，主成本A风险指标改善还可能来自较多现金。Level1比“稳定跨年、合理成本普遍领先”的Level2更符合当前证据；不讨论Level3。',
    '### D. Remaining uncertainty（剩余不确定性）',
    '只有一个历史路径，ETF彼此相关、日收益存在时间相关，969行日收益不是969次独立实验。不做普通独立样本t-test、显著性保证、置信区间或稳定alpha声明。资产池已冻结，但仍有Survivorship Bias（幸存者偏差）和历史可得数据范围限制；研究级成交不等于真实券商撮合，最小单位、最低佣金、税费等真实性尚未全部核验。',
    '每次解释都须同时检查Look-ahead Bias（前视偏差）、Survivorship Bias（幸存者偏差）和Transaction Costs（交易成本）。本阶段用冻结账本、哈希和只读评价避免改路径，不会消除上游研究的全部限制。2026保持封存：本次只授权评价开发账户；未来需单独冻结最终验证协议并明确授权后才可打开，不能因为当前结果好就随时查看。',
    '## 图表',
    '图中A/B/C含义同上；比例轴以百分比显示，收益差也是百分点；图表仅来自本报告CSV，无新增策略。']
    for filename,title in [('01_cumulative_nav','主成本累计净值'),('02_drawdown','主成本回撤'),('03_annual_returns','年度收益'),
        ('04_sharpe_by_fee','Sharpe与费用'),('05_max_drawdown_by_fee','最大回撤与费用'),('06_cost_sensitivity','六项成本敏感性'),
        ('07_turnover_friction','换手与直接摩擦'),('08_exposure_cash','暴露与现金'),('09_prediction_coverage','预测覆盖'),
        ('10_yearly_return_difference','年度相对收益'),('11_annual_cumulative_nav','各年累计净值'),('12_cost_relative_advantage','成本下的相对优势')]:
        parts.append(f'![{title}](charts/{filename}.png)')
    parts += ['## 文件、验证与门槛',
    '主要交付：performance_metrics.csv、annual_metrics.csv、benchmark_comparison.csv、cost_sensitivity.csv、drawdown_analysis.csv、exposure_analysis.csv、coverage_effect_summary.csv、yearly_strategy_comparison.csv。补充daily_return_drawdown、annual_nav_curves、drawdown_episodes、holding_count_distribution、asset_weight_drift、cost_attribution、coverage_cashification；均为CSV。input_audit.json列明实际读取文件；manifest.json绑定所有输出哈希、Phase7 build_key/spec/rules/12个account hash、评价源码和运行环境。',
    '发布只有在冻结输入逐字节核验、账户恒等式、费用/成交/持仓汇总、年度复利、年度财富差合计全部通过后发生。verify重新计算的是评价表及图表，不执行策略或训练模型。失败不会形成complete目录。完整pytest/pip check及Git记录见仓库results/reports/phase8-validation.json。',
    'performance_evaluation_completed=true；phase8_reconciled=true；strategy_rules_frozen=true；strategy_optimization_performed=false；holdout_files_opened=false；live_execution_ready=false；strategy_replayed=false；model_retrained=false。',
    '## 初学者版回答',
    'A有没有比C强：在主成本这段历史上，收益与所列风险指标都较好，但不是每年、更不是所有费用下都更好。B怎么样：风险较低，主成本收益却没超过C，预测误差小不自动等于投资收益高。最令人兴奋的是A的主成本Sharpe与较浅回撤同时改善；最令人担心的是高换手、2025落后以及覆盖缺失造成的现金化。',
    f"主成本直接摩擦分别吃掉A {money(costs.loc['A','direct_friction'])}元、B {money(costs.loc['B','direct_friction'])}元、C {money(costs.loc['C','direct_friction'])}元；这不是全部路径影响。A相对C的终值优势从fee=0时约8002元缩到主成本约4732元，再到fee=0.001时落后约2759元。不能把这段差额的变化简单全部记成手续费。",
    '现在不能说模型未来会赚钱，也不能用历史结果承诺真实投资收益。最诚实的一句话：**冻结开发期存在局部历史经济信号，风险指标有所改善，但年度、成本和预测覆盖稳定性不足，尚不能证明可部署的持续超额收益。** 完成本阶段后停止，不开始Forward Test（前向测试）或实盘。']
    return '\n\n'.join(parts)+'\n'
