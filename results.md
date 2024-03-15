
### Baselines
Datasets                      | BM25<td colspan=2>| Contriver<td colspan=2>  |
                              | R@100   | nDCG@10 | R@100  | nDCG@10         | 
---                           | ---     | ---     | ---    | ---             |  
msmarco-dev-subset            | 0.6703  | 0.2338  |        |                 |
beir-scifact                  | 0.9076  | 0.6647  | 0.9043 | 0.6394          |
beir-trec-covid               | 0.1142  | 0.6534  | 0.0366 | 0.2771          |
beir-scidocs                  | 0.3561  | 0.1581  | 0.3600 | 0.1509          |
---                           | ---     | ---     | ---    | ---             |  

Datasets                      | BM25<td colspan=2>| Contriever<td colspan=2> |
                              | R@100   | nDCG@10 | R@100  | nDCG@10         |
---                           | ---     | ---     | ---    | ---             | 
lotte-science-dev.search      | 0.6870  | 0.3406  | 0.4853 | 0.1896          |
lotte-science-dev.forum       | 0.4854  | 0.2484  | 0.2516 | 0.1178          |
lotte-writing-dev.search      | 0.6052  | 0.3262  | 0.7210 | 0.4505          |
lotte-writing-dev.forum       | 0.5232  | 0.3731  | 0.5735 | 0.3512          |
lotte-lifestyle-dev.search    | 0.6148  | 0.3275  | 0.8126 | 0.4795          |
lotte-lifestyle-dev.forum     | 0.4231  | 0.2076  | 0.6506 | 0.3583          |
lotte-recreation-dev.search   | 0.6803  | 0.3409  | 0.7355 | 0.4147          |
lotte-recreation-dev.forum    | 0.4852  | 0.2660  | 0.6342 | 0.3520          |
lotte-technology-dev.search   | 0.4956  | 0.2183  |        |                 |
lotte-technology-dev.forum    | 0.2757  | 0.1107  |        |                 |
---                           | ---     | ---     | ---    | ---             | 
lotte-science-test.search     | 0.4868  | 0.2043  |        |                 |
lotte-science-test.forum      | 0.3027  | 0.1560  |        |                 |
lotte-writing-test.search     | 0.6639  | 0.4130  |        |                 |
lotte-writing-test.forum      | 0.5413  | 0.3514  |        |                 |
lotte-lifestyle-test.search   | 0.7386  | 0.4255  |        |                 |
lotte-lifestyle-test.forum    | 0.5476  | 0.3049  |        |                 |
lotte-recreation-test.search  | 0.6976  | 0.3928  |        |                 |
lotte-recreation-test.forum   | 0.5699  | 0.3247  |        |                 |
lotte-technology-test.search  | 0.5637  | 0.2525  |        |                 |
lotte-technology-test.forum   | 0.3350  | 0.1508  |        |                 |
---                           | ---     | ---     | ---    | ---             |

# Additional results (Colbert-v2)
```
# [Lotte-dev]
[query_type=search, dataset=writing] Success@5: 47.3
[query_type=search, dataset=recreation] Success@5: 56.3
[query_type=search, dataset=science] Success@5: 52.2
[query_type=search, dataset=technology] Success@5: 35.8
[query_type=search, dataset=lifestyle] Success@5: 54.4

[query_type=forum, dataset=writing] Success@5: 66.2
[query_type=forum, dataset=recreation] Success@5: 56.6
[query_type=forum, dataset=science] Success@5: 51.3
[query_type=forum, dataset=technology] Success@5: 30.7
[query_type=forum, dataset=lifestyle] Success@5: 48.2

# [Lotte-test]
[query_type=search, dataset=writing] Success@5: 60.3
[query_type=search, dataset=recreation] Success@5: 56.5
[query_type=search, dataset=science] Success@5: 32.7
[query_type=search, dataset=technology] Success@5: 41.8
[query_type=search, dataset=lifestyle] Success@5: 63.8
[query_type=search, dataset=pooled] Success@5: ???

[query_type=forum, dataset=writing] Success@5: 64.0
[query_type=forum, dataset=recreation] Success@5: 55.4
[query_type=forum, dataset=science] Success@5: 37.1
[query_type=forum, dataset=technology] Success@5: 39.4
[query_type=forum, dataset=lifestyle] Success@5: 60.6
[query_type=forum, dataset=pooled] Success@5: ???
```
