# Supplementary strengthened experiments

Additional/post-hoc analyses; validation selects hyperparameters and test outcomes are retained in every direction. No practitioner or behavioral validation is claimed.

## Main models

| Model | Hit10 | NDCG10 | Route NDCG | Template NDCG | Coverage | Tail Hit10 |
|---|---:|---:|---:|---:|---:|---:|
| F1 | 0.783429 | 0.527062 | 0.530968 | 0.520621 | 0.474855 | 0.279689 |
| W1 | 0.784856 | 0.528136 | 0.530619 | 0.521745 | 0.462282 | 0.273605 |
| VOM | 0.830962 | 0.588502 | 0.576148 | 0.559018 | 0.629594 | 0.427822 |
| TD-VOM | 0.830723 | 0.590587 | 0.575421 | 0.560855 | 0.621857 | 0.422786 |
| Interpolated-VOM | 0.832673 | 0.589625 | 0.577136 | 0.560324 | 0.630561 | 0.436634 |
| Log-count-SVD | 0.785415 | 0.525109 | 0.528000 | 0.518082 | 0.360735 | 0.207721 |
| Popularity | 0.430961 | 0.255379 | 0.263701 | 0.252639 | 0.010638 | 0.000000 |
| Nearest-distance | 0.170101 | 0.088386 | 0.074091 | 0.074899 | 0.984526 | 0.207092 |
| GRU | 0.829521 | 0.607915 | 0.597570 | 0.578771 | 0.605738 | 0.405931 |
| Transformer | 0.839970 | 0.604278 | 0.603701 | 0.585035 | 0.707286 | 0.396139 |

## Descriptive subgroup diagnostics

| Subgroup | Events | Routes | F1 NDCG | VOM NDCG | TD-VOM NDCG |
|---|---:|---:|---:|---:|---:|
| template_seen | 7908 | 714 | 0.591966 | 0.720022 | 0.707169 |
| template_unseen | 104919 | 4276 | 0.522170 | 0.578589 | 0.581800 |
| route_start | 4990 | 4990 | 0.559956 | 0.559956 | 0.555344 |
| context_seen | 105232 | 4826 | 0.529646 | 0.595520 | 0.597941 |
| context_unseen | 2605 | 1146 | 0.359669 | 0.359669 | 0.361046 |
| edge_seen | 110128 | 4949 | 0.539794 | 0.602740 | 0.604877 |
| edge_unseen | 2699 | 1194 | 0.007548 | 0.007548 | 0.007548 |
| target_training_zero | 114 | 89 | 0.000000 | 0.000000 | 0.000000 |
| target_training_1_20 | 823 | 505 | 0.060261 | 0.169958 | 0.166989 |
| target_training_over20 | 111890 | 4969 | 0.531032 | 0.592180 | 0.594305 |
| length_2_10 | 11153 | 2105 | 0.541828 | 0.573783 | 0.567788 |
| length_11_30 | 25033 | 1514 | 0.520189 | 0.569384 | 0.569754 |
| length_over30 | 76641 | 1371 | 0.527158 | 0.596888 | 0.600710 |
| distance_atmost50km | 48009 | 4527 | 0.691925 | 0.710590 | 0.713754 |
| distance_50_200km | 13061 | 3449 | 0.352468 | 0.444168 | 0.437714 |
| distance_over200km | 51757 | 4708 | 0.418197 | 0.511677 | 0.514918 |

## Overlap denominators

```json
{
  "route_start_events": 4990,
  "order_two_events": 107837,
  "unique_test_edges": 6600,
  "unique_test_contexts_order_two": 6306,
  "unique_test_triplets_order_two": 18444,
  "edge_event_seen_fraction": 0.9760784209453411,
  "unique_edge_seen_fraction": 0.7587878787878788,
  "context_event_seen_fraction_order_two": 0.9758431707113514,
  "unique_context_seen_fraction_order_two": 0.7588011417697431,
  "triplet_event_seen_fraction_order_two": 0.8872093993712733,
  "unique_triplet_seen_fraction_order_two": 0.5832248969854695
}
```

## Paired confidence intervals

```json
{
  "TD-VOM minus F1": [
    {
      "event_weighted_difference": 0.06352557436654001,
      "CI95": [
        0.061128474647611626,
        0.06586328197301698
      ],
      "cluster_count": 4990,
      "bootstrap_repetitions": 2000,
      "cluster": "route"
    },
    {
      "event_weighted_difference": 0.06352557436654001,
      "CI95": [
        0.06005961807763938,
        0.06694925380499744
      ],
      "cluster_count": 3892,
      "bootstrap_repetitions": 2000,
      "cluster": "template"
    }
  ],
  "TD-VOM minus VOM": [
    {
      "event_weighted_difference": 0.0020857359094754867,
      "CI95": [
        0.001534487926849448,
        0.0026295968523515584
      ],
      "cluster_count": 4990,
      "bootstrap_repetitions": 2000,
      "cluster": "route"
    },
    {
      "event_weighted_difference": 0.0020857359094754867,
      "CI95": [
        0.001120674800295076,
        0.0029666775219067865
      ],
      "cluster_count": 3892,
      "bootstrap_repetitions": 2000,
      "cluster": "template"
    }
  ],
  "TD-VOM minus Interpolated-VOM": [
    {
      "event_weighted_difference": 0.0009622780384895511,
      "CI95": [
        0.0003678177475980281,
        0.001529262203765914
      ],
      "cluster_count": 4990,
      "bootstrap_repetitions": 2000,
      "cluster": "route"
    },
    {
      "event_weighted_difference": 0.0009622780384895511,
      "CI95": [
        8.342158799863672e-05,
        0.0017959399549350324
      ],
      "cluster_count": 3892,
      "bootstrap_repetitions": 2000,
      "cluster": "template"
    }
  ],
  "TD-VOM minus Log-count-SVD": [
    {
      "event_weighted_difference": 0.06547836533488385,
      "CI95": [
        0.06295175476854642,
        0.06795404306595994
      ],
      "cluster_count": 4990,
      "bootstrap_repetitions": 2000,
      "cluster": "route"
    },
    {
      "event_weighted_difference": 0.06547836533488385,
      "CI95": [
        0.06182640600975056,
        0.0691879520028294
      ],
      "cluster_count": 3892,
      "bootstrap_repetitions": 2000,
      "cluster": "template"
    }
  ]
}
```

## Sensitivity results

### rolling_2019

```json
{
  "split": {
    "train": {
      "routes": 4529,
      "events": 110154,
      "first_date": "2017-07-13",
      "last_date": "2018-12-31"
    },
    "validation": {
      "routes": 2494,
      "events": 52154,
      "first_date": "2019-01-01",
      "last_date": "2019-06-30"
    },
    "test": {
      "routes": 9578,
      "events": 240132,
      "first_date": "2019-07-01",
      "last_date": "2019-12-31"
    }
  },
  "metrics": {
    "F1": {
      "events": 240132,
      "routes": 9578,
      "Hit@5": 0.6647135742008562,
      "Hit@10": 0.796561890959972,
      "MRR@10": 0.4607951969009637,
      "NDCG@10": 0.5412140132945622,
      "macro_route_NDCG@10": 0.5493506646101937,
      "macro_template_NDCG@10": 0.532539674745213,
      "catalog_coverage@10": 0.3230174081237911,
      "long_tail_events": 19593,
      "long_tail_Hit@10": 0.365232480988108
    },
    "W1": {
      "events": 240132,
      "routes": 9578,
      "Hit@5": 0.6679992670697783,
      "Hit@10": 0.7952459480618993,
      "MRR@10": 0.4599914194018687,
      "NDCG@10": 0.5403105210863581,
      "macro_route_NDCG@10": 0.549436056599637,
      "macro_template_NDCG@10": 0.5336981717030782,
      "catalog_coverage@10": 0.32011605415860733,
      "long_tail_events": 19593,
      "long_tail_Hit@10": 0.3529832082886745
    },
    "VOM": {
      "events": 240132,
      "routes": 9578,
      "Hit@5": 0.7184840004664101,
      "Hit@10": 0.8200531374410741,
      "MRR@10": 0.5195497915961036,
      "NDCG@10": 0.5920563822001135,
      "macro_route_NDCG@10": 0.5804036494973316,
      "macro_template_NDCG@10": 0.5605844539993706,
      "catalog_coverage@10": 0.4410058027079304,
      "long_tail_events": 19593,
      "long_tail_Hit@10": 0.4416373194508243
    },
    "TD-VOM": {
      "events": 240132,
      "routes": 9578,
      "Hit@5": 0.7178010427598155,
      "Hit@10": 0.8190703446437793,
      "MRR@10": 0.5193465964751133,
      "NDCG@10": 0.5916842484779711,
      "macro_route_NDCG@10": 0.5825341420468413,
      "macro_template_NDCG@10": 0.5640420027678408,
      "catalog_coverage@10": 0.4400386847195358,
      "long_tail_events": 19593,
      "long_tail_Hit@10": 0.42790792630020924
    }
  }
}
```

### rolling_2020

```json
{
  "split": {
    "train": {
      "routes": 16601,
      "events": 402440,
      "first_date": "2017-07-13",
      "last_date": "2019-12-31"
    },
    "validation": {
      "routes": 1246,
      "events": 15326,
      "first_date": "2020-01-01",
      "last_date": "2020-06-30"
    },
    "test": {
      "routes": 2576,
      "events": 42687,
      "first_date": "2020-07-01",
      "last_date": "2020-12-31"
    }
  },
  "metrics": {
    "F1": {
      "events": 42687,
      "routes": 2576,
      "Hit@5": 0.6462857544451472,
      "Hit@10": 0.7642842083069787,
      "MRR@10": 0.439507762483727,
      "NDCG@10": 0.517575287677944,
      "macro_route_NDCG@10": 0.5119858275626864,
      "macro_template_NDCG@10": 0.4991191154680292,
      "catalog_coverage@10": 0.39264990328820115,
      "long_tail_events": 3249,
      "long_tail_Hit@10": 0.26408125577100644
    },
    "W1": {
      "events": 42687,
      "routes": 2576,
      "Hit@5": 0.6487689460491485,
      "Hit@10": 0.7639093869327899,
      "MRR@10": 0.44112286890064667,
      "NDCG@10": 0.5186753657243436,
      "macro_route_NDCG@10": 0.513780631279646,
      "macro_template_NDCG@10": 0.5015414906862402,
      "catalog_coverage@10": 0.38781431334622823,
      "long_tail_events": 3249,
      "long_tail_Hit@10": 0.24315173899661435
    },
    "VOM": {
      "events": 42687,
      "routes": 2576,
      "Hit@5": 0.6861339517886007,
      "Hit@10": 0.7949024293110315,
      "MRR@10": 0.4841684729115328,
      "NDCG@10": 0.5589739974366681,
      "macro_route_NDCG@10": 0.5460345310656571,
      "macro_template_NDCG@10": 0.5281455890342471,
      "catalog_coverage@10": 0.5174081237911026,
      "long_tail_events": 3249,
      "long_tail_Hit@10": 0.33641120344721454
    },
    "TD-VOM": {
      "events": 42687,
      "routes": 2576,
      "Hit@5": 0.685032914001921,
      "Hit@10": 0.7920209899969546,
      "MRR@10": 0.4850299020444498,
      "NDCG@10": 0.5590680865024344,
      "macro_route_NDCG@10": 0.5434671372056509,
      "macro_template_NDCG@10": 0.5286055414881371,
      "catalog_coverage@10": 0.511605415860735,
      "long_tail_events": 3249,
      "long_tail_Hit@10": 0.32132963988919666
    }
  }
}
```

### rolling_2021

```json
{
  "split": {
    "train": {
      "routes": 20423,
      "events": 460453,
      "first_date": "2017-07-13",
      "last_date": "2020-12-31"
    },
    "validation": {
      "routes": 5696,
      "events": 111490,
      "first_date": "2021-01-01",
      "last_date": "2021-06-30"
    },
    "test": {
      "routes": 7147,
      "events": 164940,
      "first_date": "2021-07-01",
      "last_date": "2021-12-11"
    }
  },
  "metrics": {
    "F1": {
      "events": 164940,
      "routes": 7147,
      "Hit@5": 0.646313811082818,
      "Hit@10": 0.7676427791924336,
      "MRR@10": 0.4389812914171772,
      "NDCG@10": 0.5178125217796624,
      "macro_route_NDCG@10": 0.5204254935268815,
      "macro_template_NDCG@10": 0.5108146995237225,
      "catalog_coverage@10": 0.46034816247582205,
      "long_tail_events": 7231,
      "long_tail_Hit@10": 0.2312266629788411
    },
    "W1": {
      "events": 164940,
      "routes": 7147,
      "Hit@5": 0.6468170243724991,
      "Hit@10": 0.7763550381957075,
      "MRR@10": 0.44318120490183066,
      "NDCG@10": 0.522963975652185,
      "macro_route_NDCG@10": 0.5258414398399774,
      "macro_template_NDCG@10": 0.5163984421811899,
      "catalog_coverage@10": 0.4506769825918762,
      "long_tail_events": 7231,
      "long_tail_Hit@10": 0.22998202185036648
    },
    "VOM": {
      "events": 164940,
      "routes": 7147,
      "Hit@5": 0.6917424518006547,
      "Hit@10": 0.8046623014429489,
      "MRR@10": 0.48692035920709986,
      "NDCG@10": 0.5634168908828908,
      "macro_route_NDCG@10": 0.555444054485542,
      "macro_template_NDCG@10": 0.540825480338037,
      "catalog_coverage@10": 0.625725338491296,
      "long_tail_events": 7231,
      "long_tail_Hit@10": 0.35223343935831836
    },
    "TD-VOM": {
      "events": 164940,
      "routes": 7147,
      "Hit@5": 0.6955377713107797,
      "Hit@10": 0.8070571116769735,
      "MRR@10": 0.48978159446147806,
      "NDCG@10": 0.5661947008531798,
      "macro_route_NDCG@10": 0.5573619133899238,
      "macro_template_NDCG@10": 0.5434413466096789,
      "catalog_coverage@10": 0.6170212765957447,
      "long_tail_events": 7231,
      "long_tail_Hit@10": 0.3489143963490527
    }
  }
}
```

### exclude_flagged

```json
{
  "split": {
    "train": {
      "routes": 6279,
      "events": 51286,
      "first_date": "2017-12-02",
      "last_date": "2021-06-03"
    },
    "validation": {
      "routes": 1440,
      "events": 15402,
      "first_date": "2021-06-03",
      "last_date": "2021-07-17"
    },
    "test": {
      "routes": 1252,
      "events": 12585,
      "first_date": "2021-07-17",
      "last_date": "2021-11-27"
    }
  },
  "metrics": {
    "F1": {
      "events": 12585,
      "routes": 1252,
      "Hit@5": 0.6486293206197855,
      "Hit@10": 0.7659117997616209,
      "MRR@10": 0.4427552326718,
      "NDCG@10": 0.5203659740577756,
      "macro_route_NDCG@10": 0.5076414018296023,
      "macro_template_NDCG@10": 0.4894006722397255,
      "catalog_coverage@10": 0.40522243713733075,
      "long_tail_events": 1429,
      "long_tail_Hit@10": 0.3554933519944017
    },
    "W1": {
      "events": 12585,
      "routes": 1252,
      "Hit@5": 0.6552244735796583,
      "Hit@10": 0.7660707191100516,
      "MRR@10": 0.44939793530973504,
      "NDCG@10": 0.5254745356239667,
      "macro_route_NDCG@10": 0.511549529586527,
      "macro_template_NDCG@10": 0.49791554868759524,
      "catalog_coverage@10": 0.4032882011605416,
      "long_tail_events": 1429,
      "long_tail_Hit@10": 0.34009797060881736
    },
    "VOM": {
      "events": 12585,
      "routes": 1252,
      "Hit@5": 0.6964640444974176,
      "Hit@10": 0.7919745729042511,
      "MRR@10": 0.5251935410636245,
      "NDCG@10": 0.589454934164098,
      "macro_route_NDCG@10": 0.5591121054336797,
      "macro_template_NDCG@10": 0.5330998186155917,
      "catalog_coverage@10": 0.48549323017408125,
      "long_tail_events": 1429,
      "long_tail_Hit@10": 0.4464660601819454
    },
    "TD-VOM": {
      "events": 12585,
      "routes": 1252,
      "Hit@5": 0.6990862137465237,
      "Hit@10": 0.7942789034564959,
      "MRR@10": 0.523346229764585,
      "NDCG@10": 0.5886429458549713,
      "macro_route_NDCG@10": 0.5549726880087911,
      "macro_template_NDCG@10": 0.535448129233539,
      "catalog_coverage@10": 0.4796905222437137,
      "long_tail_events": 1429,
      "long_tail_Hit@10": 0.44226731980405876
    }
  }
}
```

### length_atmost20

```json
{
  "split": {
    "train": {
      "routes": 14657,
      "events": 112344,
      "first_date": "2017-12-02",
      "last_date": "2021-06-03"
    },
    "validation": {
      "routes": 3058,
      "events": 26261,
      "first_date": "2021-06-03",
      "last_date": "2021-07-17"
    },
    "test": {
      "routes": 3171,
      "events": 25428,
      "first_date": "2021-07-17",
      "last_date": "2021-12-11"
    }
  },
  "metrics": {
    "F1": {
      "events": 25428,
      "routes": 3171,
      "Hit@5": 0.6770095957212522,
      "Hit@10": 0.8007314771118452,
      "MRR@10": 0.46722352999193495,
      "NDCG@10": 0.5474804668755885,
      "macro_route_NDCG@10": 0.5538221929555942,
      "macro_template_NDCG@10": 0.5369025346941267,
      "catalog_coverage@10": 0.42166344294003866,
      "long_tail_events": 1574,
      "long_tail_Hit@10": 0.29860228716645487
    },
    "W1": {
      "events": 25428,
      "routes": 3171,
      "Hit@5": 0.6785826647789838,
      "Hit@10": 0.8087148025798333,
      "MRR@10": 0.47027213158348125,
      "NDCG@10": 0.5515921398112109,
      "macro_route_NDCG@10": 0.5585175236478903,
      "macro_template_NDCG@10": 0.5429414751325847,
      "catalog_coverage@10": 0.41392649903288203,
      "long_tail_events": 1574,
      "long_tail_Hit@10": 0.2909783989834816
    },
    "VOM": {
      "events": 25428,
      "routes": 3171,
      "Hit@5": 0.7223139845839233,
      "Hit@10": 0.8304231555765298,
      "MRR@10": 0.5173383109732803,
      "NDCG@10": 0.5928870427161356,
      "macro_route_NDCG@10": 0.5968120287955744,
      "macro_template_NDCG@10": 0.5701857221813765,
      "catalog_coverage@10": 0.5203094777562862,
      "long_tail_events": 1574,
      "long_tail_Hit@10": 0.3748411689961881
    },
    "TD-VOM": {
      "events": 25428,
      "routes": 3171,
      "Hit@5": 0.7262073305018091,
      "Hit@10": 0.833962560956426,
      "MRR@10": 0.5186141823411762,
      "NDCG@10": 0.5946937517594244,
      "macro_route_NDCG@10": 0.5977609994983544,
      "macro_template_NDCG@10": 0.5736148199434647,
      "catalog_coverage@10": 0.5067698259187621,
      "long_tail_events": 1574,
      "long_tail_Hit@10": 0.37293519695044475
    }
  }
}
```
