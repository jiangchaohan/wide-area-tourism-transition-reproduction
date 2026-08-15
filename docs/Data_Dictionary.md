# Data dictionary

## Raw route-record workbooks

The original reviewer dataset is split across Excel workbooks solely because GitHub rejects files larger than 100 MB. The header row is repeated in each part. Concatenate the rows in ascending part-number order to reconstruct the complete 36,347-record table. No row is sampled, filtered, recoded or reordered during splitting.

| Field | Meaning |
|---|---|
| `source_record_id` | Stable record identifier in the supplied route-text corpus |
| `route_title` | Route title as supplied in the source record |
| `departure_date` | Departure date recorded in the source |
| `itinerary_text` | Original route-description text |
| `province` | Province field in the source record |
| `city_sequence_raw` | Ordered city-name sequence extracted from the source |
| `attraction_sequence_raw` | Ordered attraction-name sequence extracted from the source |

## Processed workbooks

- `Supplementary_Data_S2_Cleaned_Routes.xlsx`: standardized route sequences and complete-route split fields.
- `Supplementary_Data_S3_States_and_Transitions.xlsx`: attraction-state dictionary, coordinates and directed adjacent transitions.
- `Supplementary_Data_S4_Model_Results.xlsx`: model-performance, ablation, stationary-distribution and route-channel outputs reported in the manuscript.

The coordinates support great-circle distance calculations only. They do not represent road distance or driving time.
