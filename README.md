# prithvi-eo-2-300m-tl

Deploy do **[ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL)**. Foundation model multitemporal pra Earth Observation (IBM + NASA + Jülich).

- **Tipo**: ViT-MAE backbone, 300M params
- **Input**: 6-band HLS (Blue, Green, Red, Narrow NIR, SWIR 1, SWIR 2)
- **Features TL**: temporal + location embeddings (latitude/longitude + ano/dia)
- **Output**: embeddings (features pra downstream tasks)
- **Licença**: Apache 2.0

## API

| Input | Default | Descrição |
|---|---|---|
| `image` | obrigatório | GeoTIFF 6-band (HLS) ou RGB (auto-pad) |
| `image_size` | 224 | Resolução (64-1024, múltiplo de 16) |
| `latitude` | 0 | Lat do centro (-90 a 90, TL feature) |
| `longitude` | 0 | Lon do centro (-180 a 180, TL feature) |
| `year` | 2024 | Ano de aquisição |
| `day_of_year` | 180 | Dia do ano 1-365 |
| `return_format` | "summary" | "summary" ou "full" |

## Output (summary)

```json
{
  "model": "prithvi-eo-2-300m-tl",
  "embedding_dim": 1024,
  "image_size": 224,
  "metadata": {"latitude": 0, "longitude": 0, "year": 2024, "day_of_year": 180},
  "mean": 0.012,
  "std": 0.387,
  "min": -1.84,
  "max": 2.11,
  "norm": 22.14,
  "embedding_preview": [...]
}
```

## Hardware
- **gpu-t4** (16 GB) suficiente
- Inferência: ~1-2 s
