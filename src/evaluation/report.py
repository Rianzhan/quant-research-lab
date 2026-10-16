"""Deterministic audit report; interpret scores without choosing or changing models."""
import math
import pandas as pd
from src.evaluation.cohort import EVAL_KEYS


def markdown_table(frame: pd.DataFrame) -> str:
    """Render without adding a tabulate dependency; preserve undefined values as NA."""
    def cell(value):
        if isinstance(value,float): return f'{value:.6f}' if math.isfinite(value) else 'NA'
        return str(value).replace('|','/').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |',
                      *['| '+' | '.join(cell(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None)]])


def scope_stability(pairs: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for (model,task,metric,level),group in pairs.loc[pairs.level.isin(['fold','asset','fold_asset'])].groupby(
            ['model','task','metric','level'],sort=True):
        valid=group.loc[group.status.eq('paired')&group.improvement.notna()];delta=valid.improvement
        rows.append({'model':model,'task':task,'metric':metric,'level':level,'valid_groups':len(valid),
                     'pooled_better':int(delta.gt(1e-12).sum()),'tie':int(delta.abs().le(1e-12).sum()),
                     'per_asset_better':int(delta.lt(-1e-12).sum()),'macro_improvement':delta.mean()})
    return pd.DataFrame(rows)


def concentration(parts: pd.DataFrame) -> pd.DataFrame:
    """Additive loss attribution only; never attribute AUC or RMSE by summing scores."""
    rows=[]
    selected=parts.loc[parts.level.isin(['fold','asset'])&((parts.baseline=='training_prior')|(parts.baseline=='zero_return'))]
    for identity,group in selected.groupby(['model','scope','task','baseline','metric','level'],sort=True):
        valid=group.loc[group.n>0];total=valid.loss_improvement_sum.sum();n=int(valid.n.sum())
        if not len(valid): continue
        top=valid.loc[valid.loss_improvement_sum.idxmax()];remaining=n-int(top.n)
        rows.append({**dict(zip(['model','scope','task','baseline','metric','level'],identity)),
            'n':n,'total_loss_improvement_sum':total,'positive_groups':int(valid.loss_improvement_sum.gt(1e-12).sum()),
            'valid_groups':len(valid),'largest_contributor':top.fold_id if identity[-1]=='fold' else top.asset_code,
            'largest_contribution':top.loss_improvement_sum,
            'share_of_net_positive_advantage':top.loss_improvement_sum/total if total>1e-12 else float('nan'),
            'mean_improvement_excluding_largest':(total-top.loss_improvement_sum)/remaining if remaining else float('nan')})
    return pd.DataFrame(rows)


def build_report(tables: dict[str,pd.DataFrame],joined: pd.DataFrame,policy: dict) -> bytes:
    """All output conclusions are descriptive and retain every registered model and asset."""
    cl=tables['classification_metrics.csv'];reg=tables['regression_metrics.csv'];base=tables['baseline_comparison.csv']
    unique=joined[EVAL_KEYS].drop_duplicates();market=unique.drop(columns='task').drop_duplicates()
    unavailable=tables['availability.csv'].loc[tables['availability.csv'].status!='available']
    text=['# Phase 5.5 冻结样本外预测质量报告','',
        '仅评价已冻结的开发实验预测。下面都是历史样本外实验结果，不是投资收益保证。本阶段没有训练、重新预测、模型选择、概率校准、阈值搜索或组合回测。',
        '开发数据边界为2018-01-01至2025-12-31；外层样本外年度为2022、2023、2024、2025。2026最终留出候选集保持封存。',
        f'预测文件共有 {len(joined):,} 行；去掉模型与scope重复后有 {len(market):,} 个fold×日期×资产×事件片段市场样本键、{len(unique):,} 个含task评价键。不能将重复预测行当作独立市场实验。',
        '', '## 输入、基准和比较口径','',
        '源文件、历史代码提交、17列共同白名单、训练/测试成员与全部已保存模型文件逐项核对。模型文件只读取字节计算SHA-256，不反序列化。训练标签结束日必须早于对应测试开始日。',
        'per_asset（单资产模型）仅从该ETF已purge的训练成员计算上涨比例和平均收益；pooled（汇总模型）从所有合格训练成员计算。训练样本和测试样本不混合，基准不会使用当年测试上涨比例。',
        '多数类：训练上涨比例>=0.5时固定预测1，否则0（相等固定选1）；永远上涨固定预测1；training_prior输出训练上涨比例。前两种把硬判断视为0/1概率，因此Brier较差可能来自过度确定，不能把击败硬基准误读为概率模型有效。',
        '回归基准：zero_return恒为0，training_mean为对应训练集真实未来5日收益均值。各基准严格在参考模型完全相同的键上评分。',
        '配对键为fold_id、date、exchange、asset_code、price_segment、task；比较同一算法的per_asset与pooled，键集不一致就不可比较，不取交集。588000/2022仍为unavailable/insufficient。',
        '主表包括all、fold、asset、fold_asset四层。all为按预测样本加权的合并分数；stability中的macro为每年/每ETF同等权重的分数均值。AUC、相关系数和RMSE等非线性分数，两者不可互换。',
        'training_prior在单一资产单一fold内是常数；跨资产/年份合并后可能变化，因此其合并ROC-AUC并不一定恰为0.5。跨资产合并相关也可能受资产收益尺度影响，必须同时看ETF内结果。',
        '', '## 指标怎么读','',
        '- ROC-AUC = Receiver Operating Characteristic Area Under the Curve（受试者工作特征曲线下面积）：比较正样本的预测概率排在负样本之前的程度；随机排序参考值0.5，单一实际类别时未定义。',
        '- Brier Score（布里尔分数）= mean((p-y)^2)，越低越好。它同时反映概率误差的多种成分，不能单凭它宣布校准好。',
        '- Calibration（概率校准）：固定10个宽度0.1的左闭右开箱，最后一箱含1；每箱列出平均预测概率、实际正类比例和样本量。空箱保留NA；箱中样本少时曲线会不稳定。本阶段不重新校准。',
        '- 固定描述性分类规则p>=0.5。Accuracy=(TP+TN)/N；Balanced Accuracy=(TP/(TP+FN)+TN/(TN+FP))/2。Precision=TP/(TP+FP)；Recall=TP/(TP+FN)；F1=2TP/(2TP+FP+FN)。TP/FP/FN/TN分别为真阳/假阳/假阴/真阴；分母为0时该指标保留NA，F1可以在分母非0而TP=0时等于0。',
        '- MAE = Mean Absolute Error（平均绝对误差）=mean(abs(pred-y))；RMSE = Root Mean Squared Error（均方根误差）=sqrt(mean((pred-y)^2))。单位是收益率小数，例如0.01表示1个百分点，越低越好。',
        '- Pearson Correlation（皮尔逊相关系数）=协方差除以两个标准差；Spearman Rank Correlation（斯皮尔曼秩相关系数）=平均秩之间的Pearson相关。预测或真实值为常数、或少于2行时未定义，不能填0。',
        '- baseline_comparison的improvement：误差指标为基准减模型，其余为模型减基准；paired_scope_comparison中正值均表示pooled更好。1e-12只处理浮点数相等，不是显著性标准。',
        '', '## 合并描述性结果','']
    text += ['分类模型：', '',markdown_table(cl.loc[cl.level=='all',['model','scope','n','roc_auc','brier','accuracy','balanced_accuracy','precision','recall','f1','tn','fp','fn','tp']]),'',
             '回归模型：','',markdown_table(reg.loc[reg.level=='all',['model','scope','n','mae','rmse','pearson','spearman']]),'',
             '合并基准比较（完整基准和全部指标在baseline_metrics.csv和baseline_comparison.csv）：','',
             markdown_table(base.loc[(base.level=='all')&(((base.baseline=='training_prior')&base.metric.isin(['roc_auc','brier','accuracy','balanced_accuracy']))|
                                                       (base.baseline.isin(['zero_return','training_mean'])&base.metric.isin(['mae','rmse'])))
                                   ,['model','scope','baseline','metric','model_value','baseline_value','improvement']]),'']
    for name,data,metrics in [('年度分类',cl,['roc_auc','brier','accuracy','balanced_accuracy']),('年度回归',reg,['mae','rmse','pearson','spearman'])]:
        text += ['## '+name,'',markdown_table(data.loc[data.level=='fold',['fold_id','model','scope','n',*metrics]]),'']
    stability=tables['stability.csv'];chosen=stability.loc[stability.level.isin(['fold','asset'])&
        (((stability.baseline=='training_prior')&stability.metric.isin(['roc_auc','brier','accuracy','balanced_accuracy']))|
         (stability.baseline.isin(['zero_return','training_mean'])&stability.metric.isin(['mae','rmse'])))]
    text += ['## 相对朴素基准的稳定性','',
        'wins/ties/losses统计有效组；fold有4年，asset有8只，fold_asset有31个可用单元。年度某项wins=4仅表示该项每年较好，不表示所有指标都好或已获显著性证明。',
        '',markdown_table(chosen[['model','scope','baseline','metric','level','groups_valid','wins','ties','losses','macro_improvement']]),'',
        '## 单资产与汇总配对稳定性','',markdown_table(tables['scope_stability.csv'].loc[
            tables['scope_stability.csv'].level.isin(['fold','asset'])&tables['scope_stability.csv'].metric.isin(['roc_auc','brier','mae','rmse','pearson','spearman'])]),'']
    for name,data,metrics in [('ETF分类',cl,['roc_auc','brier','accuracy','balanced_accuracy']),('ETF回归',reg,['mae','rmse','pearson','spearman'])]:
        text += ['## '+name,'',markdown_table(data.loc[data.level=='asset',['asset_code','model','scope','n',*metrics]]),'']
    text += ['## 优势是否集中','',
        '下表只分解可加的Brier、MAE和平方误差。最大贡献占净优势可超过100%，表示其他组抵消部分优势；总优势非正时占比为NA。去掉最大贡献组仍为正，比只看最大亮点更有说服力，但仍是描述性诊断，不用于删除资产或选择模型。',
        '',markdown_table(tables['concentration.csv']),'', '## 不可用成员','',
        markdown_table(unavailable) if len(unavailable) else '无不可用成员。','',
        '## 图表','',*['![{}](figures/{})'.format(label,name) for label,name in [
            ('年度ROC-AUC','roc_auc_by_fold.png'),('分类与朴素基准','classification_vs_baselines.png'),
            ('Brier比较','brier_vs_baselines.png'),('概率校准','calibration_curves.png'),
            ('回归误差','regression_errors.png'),('回归相关','regression_correlations.png'),
            ('年度scope配对','paired_scope_by_year.png'),('ETF年度分类','classification_etf_heatmap.png'),
            ('ETF年度回归','regression_etf_heatmap.png')]],'',
        '## 限制、风险与复核方法','',
        '未来5交易日标签在相邻日期重叠，同一天不同ETF也受共同市场因素影响。没有采用独立样本t检验、p值或置信区间；任何微小差异不能声称显著。',
        '请同时检查：Look-ahead Bias（前视偏差）、Survivorship Bias（幸存者偏差）、Transaction Costs（交易成本）。这里只复用冻结开发预测，没有用未来数据调规则；8只当前预注册ETF不代表历史完整市场，幸存者偏差仍在；尚未估计交易成本后的表现。',
        '研究价格仍是权益事件片段内不复权价格，跨事件标签无效，存在选择性的缺失样本和上市时间不同。原始数据修订不能悄悄混入本评价，文件变化将拒绝继续。',
        'backtest_total_return_ready=false。5日收盘到收盘标签、每日重叠评价样本、未来次日开盘模拟执行之间并不等价；未纳入真实换手、手续费、滑点、现金分红和仓位路径，不能由预测分数推断能否赚钱或买哪只ETF。',
        '2026未参与特征、模型、阈值、参数选择或本评价。额外最终留出候选集不是已经完成的Forward Test（前向测试）；其封存状态不因本报告改变。',
        '复核：先检查audit.json与manifest的输入/输出SHA，再核对每年样本量、availability、baseline_parameters训练日期及purge边界；从calibration_results抽箱验证加权数量，从paired表确认两侧key_sha256相同。tests覆盖手算指标和故意破坏输入的拒绝路径。',
        '所有报告数字直接由冻结预测和标签计算，未修改数据以改善外观。初学者结论和实际验证日志另见docs/phase55-results.md。',
        '', '参考： [scikit-learn模型评价](https://scikit-learn.org/stable/modules/model_evaluation.html)、[概率校准](https://scikit-learn.org/stable/modules/calibration.html)。']
    return ('\n\n'.join(text)+'\n').encode('utf-8')
