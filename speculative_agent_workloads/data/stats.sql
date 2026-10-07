-- Corpus stats over data/corpus/{train,val,test}.parquet (attached as train/val/test).
-- Each `-- @query NAME` starts a named query; `python -m speculative_agent_workloads.data stats`
-- runs them all and writes results/corpus_stats.json.

-- @query by_split
SELECT
  s AS split, COUNT(*) AS blocks, SUM(n_tokens) AS tokens,
  COUNT(DISTINCT run) AS runs
FROM (
  SELECT 'train' s, * FROM train UNION ALL
  SELECT 'val' s, * FROM val UNION ALL
  SELECT 'test' s, * FROM test
)
GROUP BY s ORDER BY s;

-- @query by_kind
SELECT
  s AS split, kind, COUNT(*) AS blocks, SUM(n_tokens) AS tokens
FROM (
  SELECT 'train' s, * FROM train UNION ALL
  SELECT 'val' s, * FROM val UNION ALL
  SELECT 'test' s, * FROM test
)
GROUP BY s, kind ORDER BY s, blocks DESC;

-- @query by_tool
SELECT
  tool, COUNT(*) AS blocks, SUM(n_tokens) AS tokens
FROM (
  SELECT * FROM train UNION ALL SELECT * FROM val UNION ALL SELECT * FROM test
)
WHERE kind = 'tool_use'
GROUP BY tool ORDER BY blocks DESC;

-- @query by_project
SELECT
  project, COUNT(*) AS blocks, SUM(n_tokens) AS tokens, COUNT(DISTINCT run) AS runs
FROM (
  SELECT * FROM train UNION ALL SELECT * FROM val UNION ALL SELECT * FROM test
)
GROUP BY project ORDER BY blocks DESC;

-- @query length_percentiles
SELECT
  kind,
  approx_quantile(n_tokens, 0.5)  AS p50,
  approx_quantile(n_tokens, 0.9)  AS p90,
  approx_quantile(n_tokens, 0.99) AS p99,
  MAX(n_tokens) AS max_tokens
FROM (
  SELECT * FROM train UNION ALL SELECT * FROM val UNION ALL SELECT * FROM test
)
GROUP BY kind ORDER BY p50 DESC;
