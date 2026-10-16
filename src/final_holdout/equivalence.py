"""Development-only byte equivalence evidence; no new performance evaluation."""
from dataclasses import asdict,replace
from hashlib import sha256
from pathlib import Path
import pandas as pd
from src.backtest.contracts import require,stable_bytes
from src.backtest.phase7.engine import run_account
from src.backtest.phase7.audit import account_tables
from src.backtest.phase7.publication import account_id
from src.backtest.phase7.runner import EXTRA_NAMES
from src.backtest.pipeline import TABLE_FILES,SCHEMAS,csv_bytes as ledger_bytes
from src.features.builder import build_features
from src.features.scale_free import build_scale_free
from src.final_holdout.context import DateContext
from src.final_holdout.execution import run_account_context
from src.final_holdout.features import build_context_features
from src.final_holdout.training import csv_bytes,cohort
from src.final_holdout.inputs import P7
from src.portfolio.rules import Rules,schedule
from src.utils.research_rows import KEYS


def prove(data: dict, progress=None) -> dict:
    """Compare complete tables, not only terminal NAV; discard replay ledgers."""
    d=data;context=DateContext(tuple(d['calendar']))
    require(tuple(schedule(d['calendar'],Rules()))==context.rebalance_dates,'Historical rebalance changed')
    require(len(context.active_sessions)==969,'Historical session count changed')
    raw,new=build_context_features(d['panel'],d['config'],d['calendar'],d['validation'],d['registration'])
    oldraw=build_features(d['panel'],d['config'],d['calendar'])
    source=oldraw.merge(d['panel'][KEYS+['research_close']],on=KEYS,validate='one_to_one')
    old=build_scale_free(source,d['calendar'],d['config'],d['validation'],d['registration'])
    for a,b,c in [(raw,oldraw,d['raw_features']),(new,old,d['features'])]:
        require(csv_bytes(a)==csv_bytes(b)==csv_bytes(c),'Development feature bytes differ')
    selected=cohort(d['features'],d['labels'])[KEYS]
    eligible=d['eligibility'].loc[d['eligibility'].supervised_eligible,KEYS].sort_values(KEYS).reset_index(drop=True)
    require(selected.equals(eligible),'Final cohort does not match frozen supervised eligibility')
    evidence={'raw_features_sha256':sha256(csv_bytes(raw)).hexdigest(),
        'scale_free_features_sha256':sha256(csv_bytes(new)).hexdigest(),
        'development_sessions':len(d['calendar']),'historical_sessions':969,
        'rebalance_dates_sha256':sha256(stable_bytes(context.rebalance_dates)).hexdigest(),
        'first_future_index':969,'first_future_rebalance_index':970,'accounts':{}}
    arguments={k:d[k] for k in ['assets','quotes','exchanges','segments','actions']}
    arguments['sessions']=list(context.active_sessions)
    for strategy in 'ABC':
        for fee in Rules().fee_scenarios:
            identity=account_id(strategy,fee);policy=replace(d['policy'],fee_rate=fee)
            oldrun=run_account(strategy,policy,d['predictions'],**arguments)
            newrun=run_account_context(strategy,policy,d['predictions'],context=context,**arguments)
            expected=d['account_hashes'][identity]['account_hash']
            require(oldrun.fingerprint()==newrun.fingerprint()==expected,'Account hash differs: '+identity)
            require(stable_bytes(asdict(oldrun))==stable_bytes(asdict(newrun)),'Account bytes differ')
            tables,summary=account_tables(newrun);files={}
            for table,values in tables.items():
                name={**TABLE_FILES,**EXTRA_NAMES}[table]
                base={'cancelled_unfilled':'orders','corporate_actions':'journal'}.get(table,table)
                columns=list(values[0]) if values else list(SCHEMAS.get(base,{}))
                files[name]=ledger_bytes(values,columns)
            files.update({'signals.csv':ledger_bytes(newrun.signals),'decisions.csv':ledger_bytes(newrun.decisions),
                'order_provenance.json':stable_bytes(newrun.order_provenance),'summary.json':stable_bytes(summary)})
            for name,payload in files.items():
                require(payload==d['reader'].read(P7+'accounts/'+identity+'/'+name),'Frozen published bytes differ: '+identity+'/'+name)
            evidence['accounts'][identity]={'account_hash':expected,'files_sha256':{n:sha256(b).hexdigest() for n,b in sorted(files.items())},
                'all_15_published_files_equal':True,'old_new_account_bytes_equal':True}
            if progress:progress(identity)
    return evidence
