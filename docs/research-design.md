# Research design

## Question and controlled comparison

Can past daily prices and volume provide stable out-of-sample evidence about five-session ETF returns? Per-asset models and pooled models share the same feature order, global test windows and sample rules. The comparison changes the training scope rather than quietly changing the evaluation set.

Development uses 2018–2025, with expanding test years 2022, 2023, 2024 and 2025. Models include logistic classification, random-forest classification/regression and XGBoost classification/regression. The later frozen strategy protocol uses only logistic classification, random-forest regression and the non-ML equal-weight benchmark.

## Causal feature and label contracts

Feature construction groups by asset, corporate-action segment and uninterrupted observation run. Rolling and recursive states restart after boundaries. Invalid volume remains missing, including dependent volume windows. Unadjusted observations are never retrospectively altered to conceal corporate actions.

The original 19 columns are retained in the research implementation. The shared 17-column model whitelist removes exact return/momentum duplicates and substitutes eight scale-free price ratios. Features use information through date t only.

The target is `close[t+5] / close[t] - 1`, aligned to the domestic session calendar. A missing exact target quote is not replaced with a later observation. Labels crossing corporate-action boundaries are invalid, and missing classification labels are not converted to zero.

Training rows are purged whenever `label_end_date >= test_start_date`. Removing the last five rows of a pooled table would not enforce this condition. No random train/test shuffle is used.

## Decision, execution and valuation

These are separate states. Close-time intent determines quantity using then-known cash, holdings and prices. The next legal domestic session provides the execution reference; its open cannot retroactively size the intent. Missing open, suspension or a blocking action cancels execution without chasing a later quote.

Buy effective price is `open * (1 + slippage)`; sell effective price is `open * (1 - slippage)`. Fees apply on each side's actual effective notional. Frozen fee scenarios are 0, 0.0003, 0.0005 and 0.001, with slippage 0.0002 per side. The main fee is 0.0003. The regression hurdle stays fixed across cost scenarios.

Dividend entitlement, receivable accrual and payment are distinct. Payment transfers value between receivable and cash. Share adjustments conserve economic value except explicitly explained rounding/compensation. Stale prices can support marked valuation only, never execution or feature backfill.

Every session checks `NAV = cash + dividend_receivable + sum(shares * valuation_price)` and an explained change bridge. Negative cash, short positions and leverage are rejected. Research assumptions are not claims that broker rules have been verified.

## Interpretation

Prediction quality is evaluated separately from investment accounting. Accuracy does not establish discrimination, calibration or economic value. A lower regression error need not improve thresholded equal-weight trades. More cash can reduce risk without demonstrating model alpha.

Overlapping five-session labels and correlated assets invalidate a naive independent-observation interpretation. The project does not use ordinary independent-sample significance tests to claim stable alpha. Historical asset selection still carries survivorship and availability limitations.
