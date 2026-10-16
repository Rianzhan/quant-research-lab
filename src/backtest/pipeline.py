"""Immutable accounting demonstrations and audit publication, never a strategy run."""
from dataclasses import asdict, replace
from decimal import Decimal
from hashlib import sha256
import csv
from io import StringIO
import json
import os
from pathlib import Path
import re
from uuid import uuid4
from src.backtest.accounting import ZERO, reconcile_day
from src.backtest.actions import applicability_issues
from src.backtest.contracts import Action, Asset, OrderIntent, Policy, Quote, Result, canonical, number, require, stable_bytes
from src.backtest.engine import run

SCHEMAS={
    'orders':{
        'order_id':'string; unique frozen intent id','decision_date':'ISO date; t session close-after',
        'asset_code':'string; registered asset','side':'buy|sell','information_cutoff':'China timestamp; t 23:59:59',
        'source':'manual_test|frozen_manual_fixture','requested_shares':'decimal shares|null',
        'target_weight':'decimal [0,1]|null; never both quantity and weight','execution_date':'next calendar session|null',
        'status':'terminal reason; no auto carry','frozen_shares':'decimal; t-close-sized, lot rounded',
        'decision_nav':'CNY; t-close balance','decision_cash_available':'CNY; unrestricted t-close cash',
        'decision_shares':'shares; t-close position','decision_close':'CNY/share; same-basis legal raw close|null'},
    'fills':{
        'fill_id':'string; one fill per intent','order_id':'foreign key orders','decision_date':'ISO t date',
        'date':'ISO execution date > decision_date','asset_code':'string','side':'buy|sell','shares':'positive decimal',
        'raw_open':'CNY/share; unadjusted source','effective_price':'CNY/share; raw open with signed slippage',
        'executed_notional':'CNY; shares*effective_price','fee':'CNY; notional*single-side fee_rate',
        'slippage_cost':'CNY; shares*abs(effective_price-raw_open); already in cash',
        'cash_after':'CNY','cash_available_after':'CNY','shares_after':'nonnegative shares'},
    'daily':{
        'date':'ISO domestic session','previous_nav':'CNY; previous end NAV or initial cash',
        'cash':'CNY including restricted sale proceeds','cash_available':'CNY; cash-restricted_sale_proceeds',
        'restricted_sale_proceeds':'CNY; unavailable until next session','dividend_receivable':'CNY; accrued unpaid rights',
        'gross_asset_value':'CNY; sum of asset market values','nav':'CNY; cash+receivable+gross_asset_value',
        'pre_trade_nav':'CNY; open mark after pre-open events, before fills','executed_notional':'CNY; both sides summed',
        'turnover':'ratio; effective notional/pre_trade_nav, not halved','fees_paid':'CNY; this session',
        'slippage_cost':'CNY; this session','market_revaluation':'CNY; open and close marks',
        'dividend_accrual':'CNY; ex-date addition','share_rounding_value':'CNY; verified fractional-share rounding',
        'unexplained_pnl':'CNY; must be within epsilon','daily_reconciliation':'passed or fail before output'},
    'holdings':{
        'date':'ISO session','asset_code':'string','shares':'nonnegative decimal','valuation_price':'CNY/share|null if never observed; valuation only',
        'market_value':'CNY; shares*valuation_price','valuation_stale':'Boolean; raw close absent today',
        'last_raw_close_date':'ISO date|null','price_basis_adjustment_event':'event id or empty; cache normalization only',
        'valuation_only':'true; cannot feed execution/features'},
    'receivables':{
        'date':'ISO session','event_id':'foreign key corporate_actions','asset_code':'string',
        'kind':'entitlement|accrual|payment','entitled_shares':'record-close secondary-market shares',
        'amount':'CNY; entitlement amount not booked until accrual','receivable_after':'CNY; event balance after posting'},
    'journal':{
        'date':'ISO session','sequence':'positive integer; deterministic global event order','kind':'event/mark/trade type',
        'reference':'asset code|order id|event id|cash release','cash_delta':'CNY signed',
        'receivable_delta':'CNY signed','market_value_delta':'CNY signed','nav_before':'CNY',
        'nav_after':'CNY','expected_nav_change':'CNY; independent economic explanation','unexplained_pnl':'CNY; <= epsilon'}
}
TABLE_FILES={'orders':'orders.csv','fills':'fills.csv','daily':'daily_ledger.csv','holdings':'holdings_ledger.csv',
             'receivables':'dividend_receivable_ledger.csv','journal':'event_journal.csv'}


def schemas() -> dict:
    """Document exact output columns, including units and timing."""
    return SCHEMAS


def _intent(code: str, date: str, identifier: str, quantity: str, side: str='buy') -> OrderIntent:
    return OrderIntent(identifier,date,code,side,date+'T23:59:59+08:00','frozen_manual_fixture',requested_shares=quantity)


def synthetic_cases(policy: Policy) -> dict[str,Result]:
    """Fixed literal orders: cost round trip, dividend cash, split/halt accounting."""
    dates=['2024-01-02','2024-01-03','2024-01-04','2024-01-05','2024-01-08','2024-01-09']
    asset=[Asset('SYNTHETIC',dates[0],lot_size='1')]
    def quotes(prices):
        return [Quote(d,'SYNTHETIC',p,p) for d,p in zip(dates,prices)]
    roundtrip=run(policy,asset,dates,quotes(['10']*6),
                  [_intent('SYNTHETIC',dates[0],'buy','100'),_intent('SYNTHETIC',dates[1],'sell','100','sell')])
    no_cost=replace(policy,fee_rate='0',slippage='0')
    div=Action('synthetic_dividend','SYNTHETIC','cash_dividend','2024-01-01T23:59:59+08:00',dates[2],
               record_date=dates[1],payment_date=dates[4],cash_per_share='1',evidence_sha256='a'*64)
    dividend=run(no_cost,asset,dates,quotes(['10','10','9','9','9','9']),
                 [_intent('SYNTHETIC',dates[0],'buy','100')],[div])
    split=Action('synthetic_split','SYNTHETIC','split','2024-01-01T23:59:59+08:00',dates[2],phase='after_close',
                 record_date=dates[1],share_ratio='2',rounding='exact',evidence_sha256='b'*64,suspended_dates=(dates[2],))
    splitting=run(no_cost,asset,dates,quotes(['10','10',None,'5','5','5']),
                  [_intent('SYNTHETIC',dates[0],'buy','100'),_intent('SYNTHETIC',dates[1],'cancel_on_split','10')],[split])
    require(all(d['nav']==number(policy.initial_cash) for d in dividend.tables['daily']), 'Synthetic dividend changed total assets')
    require(all(d['nav']==number(policy.initial_cash) for d in splitting.tables['daily']), 'Synthetic pure split changed assets')
    expected_loss=sum((f['fee']+f['slippage_cost'] for f in roundtrip.tables['fills']),ZERO)
    require(abs(roundtrip.tables['daily'][-1]['nav']-number(policy.initial_cash)+expected_loss)<=number(policy.epsilon),
            'Synthetic round trip cost mismatch')
    return {'synthetic_roundtrip':roundtrip,'synthetic_dividend':dividend,'synthetic_split_halt':splitting}


def historical_cases(loaded: dict, policy: Policy) -> tuple[dict[str,Result],list[dict]]:
    """Predefined 100-share demonstrations around every complete applicable event."""
    calendar=loaded['calendar'];cases={};availability=[]
    for action in sorted(loaded['actions'],key=lambda a:a.event_id):
        require(not applicability_issues(action,loaded['assets']),'Invalid event applicability: '+action.event_id)
        if not action.applicable or action.issues():
            availability.append({'event_id':action.event_id,'status':'not_applicable_prelisting' if not action.applicable else 'blocked',
                                 'reason':action.limitation,'case_id':''})
            continue
        record_index=calendar.index(action.record_date)
        require(record_index>=2,'Insufficient pre-record history for fixed demonstration')
        end=action.payment_date if action.kind=='cash_dividend' else action.effective_date
        end_index=min(calendar.index(end)+1,len(calendar)-1)
        dates=calendar[record_index-2:end_index+1]
        asset=[a for a in loaded['assets'] if a.asset_code==action.asset_code]
        quotes=[q for q in loaded['quotes'] if q.asset_code==action.asset_code and q.date in dates]
        orders=[_intent(action.asset_code,dates[0],'manual_buy_100','100')]
        if action.kind=='cash_dividend':
            orders.append(_intent(action.asset_code,action.payment_date,'manual_sell_after_payment','100','sell'))
        else:
            orders.append(_intent(action.asset_code,action.record_date,'manual_cancel_at_split','10'))
        # Supply all this asset's events, so adjacent events are never silently omitted.
        events=[a for a in loaded['actions'] if a.asset_code==action.asset_code]
        result=run(policy,asset,dates,quotes,orders,events)
        require(result.tables['fills'] and result.tables['fills'][0]['shares']==100, 'Manual event fixture failed to establish position')
        if action.kind=='cash_dividend':
            accrual=[x for x in result.tables['receivables'] if x['event_id']==action.event_id and x['kind']=='accrual']
            require(len(accrual)==1 and accrual[0]['amount']==100*number(action.cash_per_share), 'Real dividend entitlement mismatch')
            require(result.tables['daily'][-1]['dividend_receivable']==0, 'Real payment not settled')
        else:
            holding=[x for x in result.tables['holdings'] if x['date']==action.effective_date][0]
            require(holding['shares']==100*number(action.share_ratio) and holding['valuation_stale'], 'Real split/halt mismatch')
        case_id='real_'+action.event_id
        cases[case_id]=result
        availability.append({'event_id':action.event_id,'status':'reconciled','reason':'fixed manual 100 shares; infrastructure audit only','case_id':case_id})
    return cases,availability


def validate_result(result: Result, epsilon: Decimal=Decimal('0.00000001'), initial_cash: Decimal | None=None) -> None:
    """Recheck serialized ledger stocks, cash restrictions and event flows."""
    require(set(result.tables)==set(SCHEMAS),'Incomplete ledger table set')
    for name,rows in result.tables.items():
        require(all(set(r)==set(SCHEMAS[name]) for r in rows),'Unexpected ledger schema: '+name)
    daily=result.tables['daily'];journal=result.tables['journal']
    require(bool(daily) and bool(journal),'Empty accounting history')
    dates=[r['date'] for r in daily]
    require(dates==sorted(set(dates)),'Daily date continuity')
    initial=daily[0]['previous_nav']
    require(initial_cash is None or abs(initial-initial_cash)<=epsilon,'Initial cash differs from frozen policy')
    previous=initial
    cash=initial;receivable=ZERO;market_value=ZERO
    cursor=0
    for row in daily:
        require(abs(row['previous_nav']-previous)<=epsilon,'Daily NAV continuity broken')
        start=cursor
        while cursor<len(journal) and journal[cursor]['date']==row['date']:
            entry=journal[cursor]
            require(entry['sequence']==cursor+1 and abs(entry['nav_before']-cash-receivable-market_value)<=epsilon,
                    'Journal continuity broken')
            cash+=entry['cash_delta'];receivable+=entry['receivable_delta'];market_value+=entry['market_value_delta']
            require(abs(entry['nav_after']-cash-receivable-market_value)<=epsilon,'Journal continuity broken')
            cursor+=1
        require(cursor>start and abs(row['cash']-cash)<=epsilon and abs(row['dividend_receivable']-receivable)<=epsilon
                and abs(row['gross_asset_value']-market_value)<=epsilon and abs(row['nav']-journal[cursor-1]['nav_after'])<=epsilon,
                'Journal/day balance continuity broken')
        entries=journal[start:cursor]
        for column,kinds in [('market_revaluation',{'market_open','market_close'}),
                             ('dividend_accrual',{'dividend_accrual'}),('share_rounding_value',{'share_adjustment'})]:
            expected=sum((j['expected_nav_change'] for j in entries if j['kind'] in kinds),ZERO)
            require(abs(expected-row[column])<=epsilon,'Journal/day explanation mismatch')
        previous=row['nav']
    require(cursor==len(journal),'Journal has dates outside daily ledger')
    for row in result.tables['daily']:
        reconcile_day(row,epsilon)
        require(row['cash_available']==row['cash']-row['restricted_sale_proceeds'],'Cash availability identity')
        holdings=[h for h in result.tables['holdings'] if h['date']==row['date']]
        require(all(h['shares']>=0 and ((h['valuation_price'] is None and h['shares']==h['market_value']==0)
                    or (h['valuation_price'] is not None and abs(h['market_value']-h['shares']*h['valuation_price'])<=epsilon))
                    for h in holdings),'Holdings identity')
        require(abs(sum((h['market_value'] for h in holdings),ZERO)-row['gross_asset_value'])<=epsilon,'Holdings/day identity')
        fills=[f for f in result.tables['fills'] if f['date']==row['date']]
        for daily,fill in [('fees_paid','fee'),('slippage_cost','slippage_cost'),('executed_notional','executed_notional')]:
            require(abs(sum((f[fill] for f in fills),ZERO)-row[daily])<=epsilon,'Fill/day cost identity')
    for row in result.tables['journal']:
        actual=row['cash_delta']+row['receivable_delta']+row['market_value_delta']
        require(abs(actual-row['expected_nav_change'])<=epsilon and abs(row['nav_after']-row['nav_before']-actual)<=epsilon,
                'Journal identity')


def csv_bytes(rows: list[dict], columns: list[str] | None=None) -> bytes:
    """Stable text decimals and JSON structured cells; never binary-float roundoff."""
    columns=columns or (list(rows[0]) if rows else [])
    stream=StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=columns,lineterminator='\n');writer.writeheader()
    for row in rows:
        values=canonical(row)
        writer.writerow({k:json.dumps(v,ensure_ascii=False,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in values.items()})
    return stream.getvalue().encode('utf-8')


def report_text(bundle: dict) -> str:
    """Explain accounting readiness without performance metrics or model selection."""
    lines=['# Phase 6A 历史执行与总回报记账基础设施','',
        '本报告仅为固定人工订单和权益事件对账，不是模型策略收益报告。停止于 Phase 6A，不进入 Phase 6B。',
        'holdout_files_opened=false；未读取2026行情，未解析模型预测，未训练/调用模型。',
        '','## Ready gates','', '```json',json.dumps(bundle['gates'],ensure_ascii=False,indent=2),'```','',
        '## 每日顺序与资金','',
        '释放上一会话受限卖款 → 开盘前权益事件/应收 → 合法开盘重估 → 昨日订单先卖后买 → 收盘估值 → 登记权利 → 收盘后份额调整 → 公告到账日应收转现金 → 对账 → 冻结新订单。',
        '订单只尝试下一境内会话一次。目标份额在t日收盘后固定，次日现金不足只向下缩减。跨权益事件取消；无报价/停牌不成交。收盘后拆分导致当日旧尺度close不可直接用于目标份额换算，该类目标订单取消。',
        '默认 next_session_only；卖款进入cash但在下一会话前从cash_available扣除。另一种same_day_reuse只作显式接口测试。lot=1是研究单位，未核验券商规则。',
        '','## 费用、估值与对账','',
        '买价=open×(1+slippage)，卖价=open×(1-slippage)；单边fee=实际成交份额×effective_price×fee_rate。滑点已进入现金，单列但不重复扣。默认fee=0.0003、slippage=0.0002；零摩擦合成案例用来核对纯事件恒等式。',
        'turnover=当日买卖实际成交金额绝对值之和/开盘重估后、首笔成交前NAV；单位为比率，不除以2。',
        'NAV=cash+dividend_receivable+Σshares×valuation_price。日间变化=市场重估+除息应收+核验份额取整价值−fee−slippage。支付应收不改变NAV；每笔journal与每日独立检查，超容差即失败。',
        '停牌/无close只能用本资产最近合法close作陈旧估值；拆分后仅在估值缓存除以ratio以匹配新份额。此价格不写回行情、特征或执行报价。完全没有历史合法close则拒绝估值。',
        '','## 权益事件证据','',
        '完整字段见corporate_actions.csv/json：公告known_at、登记日、生效/除息日、发放日、现金/比例、证据SHA与URL。日期型公告保守视为当日23:59:59可得，绝不假定开盘前已知。',
        '12次现金分红按登记日收盘持仓记录权利；除息日开盘前转应收，公告发放日收盘后才转现金。假定已办理指定交易的普通二级市场账户，不涉及ETF一级市场申购赎回。',
        '513100/513500提前公布整数拆分比例；拆分日停牌，日频模型在当天收盘后登记新份额并转换陈旧估值尺度。510500最终比例晚于生效日，不回写历史决策，跨事件窗口拒绝。588000上市前折算不适用于现金起步的二级市场账户。',
        '','## 案例目录','', '| 案例 | 会话数 | 成交笔数 | 对账 |','|---|---:|---:|---|']
    for name,result in sorted(bundle['cases'].items()):
        lines.append(f"| {name} | {len(result.tables['daily'])} | {len(result.tables['fills'])} | passed |")
    lines+=['','每个cases子目录含orders、fills、daily_ledger、holdings_ledger、dividend_receivable_ledger、event_journal；schema.json解释所有列和单位。人工固定100份，不按模型或收益挑选案例。',
        '','## 初学者逐笔理解','',
        '合成往返：100份，原始开盘10元。买入有效价10.002、金额1000.2、费0.30006；卖出有效价9.998、金额999.8、费0.29994。总显式费0.6、滑点0.4，所以现金净减少1元。',
        '合成分红：100份×10元在除息后变成100份×9元，同时增加100元应收，资产总额相同。到账只是把100元从应收移入现金。',
        '合成拆分：100份×10元变为200份×5元。停牌当天使用经过份额尺度转换的陈旧估值，不能把旧10元价格直接乘新200份，更不能按这个估值制造成交。',
        '','## Warning / 限制','',
        '510500的历史时点比例缺口使全开发窗口corporate_action_accounting_ready、historical_execution_ready和backtest_total_return_ready保持false。可运行的完整事件案例不等于全资产全窗口已获准。',
        '未核验8只ETF/目标券商最小交易单位、卖款复用、最低佣金、其它税费、涨跌停排队、盘口容量、实际到账日内时刻；broker/live gates保持false。历史开盘价加固定滑点不能证明真实可成交。',
        '封存/混合行情只沿用冻结清单摘要，没有重开文件计算hash，不能声称检测了其当前实体改动。旧模型/预测文件仅作不解析的完整性hash。历史供应商可能修订，快照不能证明当时价格数据实时可得。',
        '始终检查：Look-ahead Bias（前视偏差）、Survivorship Bias（幸存者偏差）、Transaction Costs（交易成本）。保留原8只资产；没有根据收益换资产或调整规则。',
        '本阶段检验钱和份额有没有算错，没有使用模型订单、交易阈值或组合策略，因此仍不能判断模型能否赚钱。']
    return '\n'.join(lines)+'\n'


def write_file(path: Path, payload: bytes) -> None:
    """Exclusive durable write; incomplete publications remain visibly pending."""
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:
        stream.write(payload);stream.flush();os.fsync(stream.fileno())


def artifact_bytes(bundle: dict) -> dict[str,bytes]:
    """Check all ledgers before preparing a publication."""
    files={'schema.json':stable_bytes(schemas()),'gates.json':stable_bytes(bundle['gates']),
           'execution_policy.json':stable_bytes(bundle['policy']),
           'corporate_actions.json':stable_bytes(bundle['actions']),
           'corporate_actions.csv':csv_bytes(bundle['actions']),
           'audit.json':stable_bytes(bundle.get('audit',{})),
           'case_availability.csv':csv_bytes(bundle.get('availability',[])),
           'report.md':report_text(bundle).encode('utf-8')}
    for name,result in sorted(bundle['cases'].items()):
        require(re.fullmatch(r'[A-Za-z0-9_-]+',name) is not None,'Unsafe case id')
        validate_result(result,initial_cash=number(bundle['policy'].get('initial_cash','100000')))
        for table,rows in result.tables.items():
            files[f'cases/{name}/{TABLE_FILES[table]}']=csv_bytes(rows,list(SCHEMAS[table]))
    return files


def verify_bundle(manifest: Path) -> dict:
    """Check complete content-addressed output; never follow arbitrary paths."""
    manifest=Path(manifest)
    require(manifest.name=='manifest.json' and not any(x.startswith('.pending-') for x in manifest.parts),'Incomplete publication')
    meta=json.loads(manifest.read_bytes());require(meta['status']=='complete','Incomplete publication')
    require(meta['holdout_files_opened'] is False,'Invalid holdout gate')
    for name,expected in meta['files'].items():
        path=(manifest.parent/name).resolve()
        require(path.is_relative_to(manifest.parent.resolve()) and not Path(name).is_absolute(),'Unsafe output path')
        require(sha256(path.read_bytes()).hexdigest()==expected,'Output hash changed: '+name)
    key=sha256(stable_bytes({'version':'phase6a-v1','bindings':meta['bindings'],'files':meta['files']})).hexdigest()
    require(meta['build_key']==key and manifest.parent.name=='phase6a-v1-'+key[:16],'Output build hash mismatch')
    return meta


def publish_bundle(parent: Path, bundle: dict) -> Path:
    """Atomic directory rename only after every ledger and output has succeeded."""
    verify_sources=bundle.get('_verify_sources',lambda:None)
    require('source_lock_sha256' not in bundle['bindings'] or '_verify_sources' in bundle,'Missing publication source verifier')
    verify_sources()
    files=artifact_bytes(bundle);hashes={n:sha256(b).hexdigest() for n,b in files.items()}
    key=sha256(stable_bytes({'version':'phase6a-v1','bindings':bundle['bindings'],'files':hashes})).hexdigest()
    parent=Path(parent);parent.mkdir(parents=True,exist_ok=True);target=parent/('phase6a-v1-'+key[:16])
    if target.exists():
        verify_bundle(target/'manifest.json');return target/'manifest.json'
    pending=parent/('.pending-'+uuid4().hex);pending.mkdir()
    for name,payload in sorted(files.items()):write_file(pending/name,payload)
    for name,expected in hashes.items():
        require(sha256((pending/name).read_bytes()).hexdigest()==expected,'Pending output hash mismatch: '+name)
    meta={'kind':'execution_accounting_infrastructure','status':'complete','version':'phase6a-v1','build_key':key,
          'bindings':bundle['bindings'],'files':hashes,'holdout_files_opened':False,
          'model_signals_used':False,'strategy_backtest_performed':False}
    write_file(pending/'manifest.json',stable_bytes(meta))
    require((pending/'manifest.json').read_bytes()==stable_bytes(meta),'Pending manifest hash mismatch')
    verify_sources()
    pending.rename(target)
    return target/'manifest.json'
