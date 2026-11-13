# Follow-Up Robustness Test

The seven-day result is methodologically valid: labels were causal, the full
training data were used, models were frozen, and costs were included. It is still
a small sample.

The same T=600 unified-long model and trading rules were later evaluated through
September 30 without retraining. The aggregate result became negative:

- all 18 dates: -$154.53;
- 16 complete dates after excluding two incomplete-state days: -$222.11;
- September 21 alone: -$397.41.

The model remained profitable on 13 of the 16 complete dates, but a strong trend
day created a large short-inventory loss. This follow-up changes the conclusion:
the model shows useful order selection in ordinary conditions, but the strategy
is not robust enough for deployment.

No result in this repository should be described as proof of a profitable trading
strategy. The defensible claim is that direct net-outcome modeling outperformed
fill-only ranking in the original frozen window and revealed a clear inventory
tail-risk problem in the extension.
