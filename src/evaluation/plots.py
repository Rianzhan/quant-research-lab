"""Deterministic descriptive figures of frozen out-of-sample predictions only."""
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS={'logistic':'#2864a0','random_forest':'#cb6d22','xgboost':'#218571'}


def _png(fig) -> bytes:
    buffer=BytesIO()
    fig.savefig(buffer,format='png',dpi=135,bbox_inches='tight',metadata={'Software':'quant-lab Phase 5.5'})
    plt.close(fig)
    return buffer.getvalue()


def _heat(ax,table: pd.DataFrame,title: str,*,center: bool=False) -> None:
    values=table.to_numpy(dtype=float);finite=values[np.isfinite(values)]
    limit=float(np.max(np.abs(finite))) if len(finite) else 1.
    cmap=plt.get_cmap('RdBu' if center else 'viridis').with_extremes(bad='#e2e6eb')
    kwargs={'vmin':-limit if limit else -1,'vmax':limit if limit else 1} if center else {}
    img=ax.imshow(np.ma.masked_invalid(values),aspect='auto',cmap=cmap,**kwargs)
    ax.set_xticks(range(len(table.columns)),table.columns,rotation=35,ha='right')
    ax.set_yticks(range(len(table.index)),table.index)
    for i in range(len(table)):
        for j in range(len(table.columns)):
            value=values[i,j]
            ax.text(j,i,'NA' if not np.isfinite(value) else f'{value:.3f}',ha='center',va='center',fontsize=8,
                    color='black',bbox={'facecolor':'white','alpha':.72,'edgecolor':'none','pad':.6})
    ax.set_title(title,fontsize=11);ax.figure.colorbar(img,ax=ax,fraction=.024,pad=.02)


def _annual_lines(table: pd.DataFrame,metrics: list[str],title: str,baselines: list[str]) -> bytes:
    fig,axes=plt.subplots(len(metrics),2,figsize=(14,4*len(metrics)),squeeze=False,layout='constrained')
    years=sorted(table.fold_id.unique())
    for j,scope in enumerate(['per_asset','pooled']):
        rows=table.loc[table.scope==scope]
        for i,metric in enumerate(metrics):
            ax=axes[i,j]
            for model in sorted(rows.model.unique()):
                for method,style in [('model','-'),*[(b,s) for b,s in zip(baselines,['--',':','-.'])]]:
                    part=rows.loc[(rows.model==model)&(rows.method==method)].set_index('fold_id').reindex(years)
                    ax.plot(years,part[metric],linestyle=style,marker='o',markersize=3,color=COLORS[model],
                            alpha=1 if method=='model' else .65,label=model+' / '+method)
            ax.set_title(scope+' / '+metric);ax.grid(alpha=.2);ax.set_xlabel('Frozen test fold')
            ax.legend(fontsize=7,ncol=2)
    fig.suptitle(title+' | descriptive, no threshold tuning',fontsize=13)
    return _png(fig)


def build_figures(tables: dict[str,pd.DataFrame]) -> dict[str,bytes]:
    """Return labelled PNGs; no model ranking, fit, signal or financial-equity chart."""
    output={};fold=tables['fold_metrics.csv'];cl=fold.loc[fold.task=='classification'];reg=fold.loc[fold.task=='regression']
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False}):
        auc=cl.loc[cl.method=='model'].copy();auc['row']=auc.model+' / '+auc.scope
        fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
        _heat(ax,auc.pivot(index='row',columns='fold_id',values='roc_auc'),'ROC-AUC by year | chance reference 0.5')
        output['roc_auc_by_fold.png']=_png(fig)
        output['classification_vs_baselines.png']=_annual_lines(cl,['accuracy','balanced_accuracy'],
            'Classification and training-only naive baselines',['majority_class','always_up','training_prior'])
        output['brier_vs_baselines.png']=_annual_lines(cl,['brier'],'Brier score (lower is better)',
            ['training_prior','majority_class','always_up'])
        output['regression_errors.png']=_annual_lines(reg,['mae','rmse'],'Regression errors (return units; lower is better)',
            ['zero_return','training_mean'])
        output['regression_correlations.png']=_annual_lines(reg,['pearson','spearman'],
            'Regression correlation (undefined constant predictions remain NA)',['zero_return','training_mean'])
        calibration=tables['calibration_results.csv'];fig,axes=plt.subplots(3,2,figsize=(12,12),layout='constrained')
        for i,model in enumerate(['logistic','random_forest','xgboost']):
            for j,scope in enumerate(['per_asset','pooled']):
                ax=axes[i,j];ax.plot([0,1],[0,1],color='grey',linestyle='--',label='perfect calibration')
                rows=calibration.loc[(calibration.level=='fold')&(calibration.method=='model')&
                                    (calibration.model==model)&(calibration.scope==scope)]
                for year,part in rows.groupby('fold_id',sort=True):
                    valid=part.loc[part['count']>0]
                    line=ax.plot(valid.mean_probability,valid.observed_rate,marker='o',markersize=3,label=year)[0]
                    ax.scatter(valid.mean_probability,valid.observed_rate,
                               s=8+100*valid['count']/max(1,valid['count'].sum()),color=line.get_color(),alpha=.5)
                ax.set(xlim=(0,1),ylim=(0,1),title=model+' / '+scope,xlabel='Mean predicted probability',ylabel='Observed positive fraction')
                ax.grid(alpha=.2);ax.legend(fontsize=7)
        fig.suptitle('Uncalibrated frozen probabilities | fixed 10 bins; marker area reflects bin count')
        output['calibration_curves.png']=_png(fig)
        pairs=tables['paired_scope_comparison.csv'];fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
        for ax,metric in zip(axes.flat,['roc_auc','brier','rmse','spearman']):
            rows=pairs.loc[(pairs.level=='fold')&(pairs.metric==metric)]
            for model,part in rows.groupby('model',sort=True):
                part=part.sort_values('fold_id');ax.plot(part.fold_id,part.improvement,marker='o',color=COLORS[model],label=model)
            ax.axhline(0,color='grey',linestyle='--');ax.set_title(metric+' | positive = pooled better');ax.grid(alpha=.2);ax.legend()
        fig.suptitle('Exact-key per-asset vs pooled annual comparison | not significance tests')
        output['paired_scope_by_year.png']=_png(fig)
        base=tables['baseline_comparison.csv']
        for task,metric,baseline in [('classification','brier','training_prior'),('regression','mae','zero_return')]:
            rows=base.loc[(base.level=='fold_asset')&(base.task==task)&(base.metric==metric)&(base.baseline==baseline)]
            models=sorted(rows.model.unique());fig,axes=plt.subplots(len(models),2,figsize=(17,3.4*len(models)),squeeze=False,layout='constrained')
            for i,model in enumerate(models):
                for j,scope in enumerate(['per_asset','pooled']):
                    part=rows.loc[(rows.model==model)&(rows.scope==scope)]
                    _heat(axes[i,j],part.pivot(index='fold_id',columns='asset_code',values='improvement'),
                          model+' / '+scope+' | '+metric+' improvement',center=True)
            fig.suptitle(task+' vs '+baseline+' | positive = model better; grey = unavailable')
            output[task+'_etf_heatmap.png']=_png(fig)
    return {'figures/'+name:payload for name,payload in output.items()}
