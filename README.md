# EMMa-Wrapper

A Python library for reading, searching, and extending the [EMMa dataset](https://emma.stanford.edu/). It provides a convenient interface to EMMa product records and images, metadata-based search, and a way to attach additional per-product data without modifying the original dataset.

## Features

- **Dataset access:** Retrieve products by ASIN and access their images, text, categories, materials, price, mass, and other metadata.
- **Search and filtering:** Query by dataset split, category, materials, custom attributes, text, numerical ranges, and whether selected fields were originally listed rather than estimated.
- **Augmentations:** Associate additional data (such as depth maps, masks, embeddings, or point clouds) with products through interchangeable storage backends.
- **PyTorch integration:** Use selected products in a map-style dataset and DataLoader with a custom transform.

## Installation

Requires **Python 3.10+**. Add the library to a [uv](https://docs.astral.sh/uv/) project:

```bash
uv add "emma-wrapper @ git+https://github.com/AltFLanca/EMMa-Wrapper.git"
```

For HDF5 augmentation sources, install the optional extra:

```bash
uv add "emma-wrapper[hdf5source] @ git+https://github.com/AltFLanca/EMMa-Wrapper.git"
```

Import the library as `emma`. PyTorch must be installed separately to use `emma.dataloader`.

## Dataset Preparation

Download EMMa v1.1 separately and point the wrapper at a directory containing the original FDD archives, search files, and the additional metadata used by the library.

See [Reverse Engineering and Dataset Preparation](docs/reverse_engineering.md) for the file layout, derived metadata, and preparation scripts.

## Usage

### Read and Search Products

```python
from emma import EmmaDataset, EmmaSearcher

ds = EmmaDataset("/path/to/EMMa")
searcher = EmmaSearcher(ds)

asins = searcher.query_asins(
    splits=["Train"],
    require_listed=["mass"],
    quantity_range=(1, 1),
    limit=10,
)

for asin in asins:
    record = ds[asin]
    print(asin, ds.get_title(record), ds.get_mass(record))
```

`ds[asin]` returns the underlying product record. The wrapper also provides accessors for materials, category paths, ratings, images, and other fields.

Searches can also use reusable `SearchQuery` objects, or return matching records or counts instead of ASINs. See [example.ipynb](src/example.ipynb) for additional examples.

### Attach Augmentations

Augmentations are associated with products by ASIN and stored separately from the base dataset. For example, to read one `.npy` file per product:

```python
from emma import AugmentedDataset, DirectorySource

depth = DirectorySource("depth", "/path/to/depth_maps", suffix=".npy")
augmented = AugmentedDataset(ds).register(depth)

for asin in asins:
    if augmented.has_augmentation("depth", asin):
        record = augmented.get_augmented_record(asin)
        print(asin, record["depth"].shape)
```

Other supported sources include `HDF5Source`, `FDDSource`, and `InMemorySource`. Custom backends can implement the `AugmentSource` interface. HDF5 sources require the `hdf5source` extra, and some HDF5 files also require compression filters registered by `hdf5plugin`.

## Additional Resources

- [Example notebook](src/example.ipynb) — dataset access, filtering, augmentation sources, and PyTorch loading examples.
- [Reverse engineering and dataset preparation](docs/reverse_engineering.md) — EMMa file formats, inferred index encodings, and required metadata files.
- [Metadata preparation scripts](src/reverse_engineering/) — build category hierarchy, quantity index, and optional package-dimension presence mask.

## About EMMa

This library is an independent interface to EMMa, introduced in *[An Extensible Multimodal Multi-task Object Dataset with Materials](https://arxiv.org/abs/2305.14352)* (Standley et al., ICLR 2023). Refer to the original EMMa project for the dataset and its terms of use.
