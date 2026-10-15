# One line on an `emmy eval prior --pools measured --json` report: per card, what the prior's pick costs against the
# measured best — regret@1 (the pick ships) and regret@10 (bench the top ten, keep the best), median and worst over
# the pools each covers (n) — and the rows the admission rule dropped before grouping.
def regret(block): if block.median == null then "none" else "\(block.median)x median, \(block.worst)x worst (\(block.groups))" end;
([.summaries[] | select(.metrics.regret1.median != null)
  | "\(.axes.gpu): regret@1 \(regret(.metrics.regret1)); regret@10 \(regret(.metrics.regret10))"] | join(". ")) as $cards
| "Measured pools: \(.header.groups) over \(.header.rows) rows"
  + (if ($cards | length) > 0 then ". " + $cards else "; no pool holds two rows" end)
  + (if ((.header.dropped // {}) | length) > 0
     then ". Dropped before grouping: " + ([.header.dropped | to_entries[] | "\(.value) \(.key)"] | join(", "))
     else "" end)
