# Reverse Engineering and Dataset Preparation

## 1. Source Files and Record Structure

Four Freeze-Dried-Data (FDD) archives are provided by the original EMMa authors:

| File | Role |
| --- | --- |
| `EMMa_v1.1.fdd` | Product records, keyed by ASIN, containing attributes such as title, description, features, category IDs, material probabilities, price, mass, ratings, quantity, and image hashes. |
| `search_file.fdd` | Search-text representation for each ASIN including title, description, features, category path and materials, with sections separated by `\|\|\|`. |
| `splits_v1.1.fdd` | ASIN lists for the dataset splits. |
| `EMMa_all_images_v1.1.fdd` | Image data, retrieved by the hashes associated with each product. |

Additional search/index files are also provided:

`category_search_file.pkl`, `weight_search_file.pkl`, `price_search_file.pkl`, `listed_search_file.pkl`, `material_search_file.pkl`, `image_search_file.pkl`, `sort_priority.pkl`, and `custom_search_file.pkl`.

### Ordering Convention

The notebooks found that `EMMa_v1.1.fdd` and `search_file.fdd` do **not** expose their keys in the same order. Indexing a search array with the position of an ASIN in the main dataset's key list would silently associate the wrong metadata with a product. The wrapper therefore builds `EmmaDataset.asins` from `search_file.fdd`.

## 2. Derived Name Mappings

### Parsing the Search String

For each ASIN, `search_file.fdd` stores the title, description, features, category path and materials concatenated in a single string, with `|||` separators. This results in no clear separation between features, category, and material tokens, limiting direct extraction of category and material tokens.

However, the corresponding record in `EMMa_v1.1.fdd` reports how many feature and category tokens to expect, which identifies the intervening material tokens. 

Therefore, the search string is interpreted as:

1. One title and one description.
2. One token for each entry in `record["features"]`.
3. Material-name tokens.
4. One category-name token for each non-negative category ID in `record["category"]`.

### Category IDs and Names

A `EMMa_v1.1.fdd` product record stores its category path as a sequence of integer IDs, with negative values used as padding. Each category has `(category_id, depth)` as its key, rather than just `category_id`, since integer IDs can repeat throught each taxonomy depth.

The mapping **`cat_id2name.pkl`** is recovered by associating category names obtaind from the search string parsing with the non-negative IDs in their record path order.

The separate [`build_category_hierarchy.py`](../src/reverse_engineering/build_category_hierarchy.py) script reconstructs parent-child relationships from the stored category paths and writes **`category_hierarchy.pkl`**. Its serialized dictionary includes `children`, `parents`, `roots`, `subtree` (each category and all descendants), `occurrences`, and record-count metadata.

### Material Bit Positions and Names

For each ASIN, `EMMa_v1.1.fdd` stores an unlabeled **182-element** material probability vector. At the same time, `material_search_file.pkl` stores a **23-byte**, yielding 184 bit positions when unpacked, where the first 182 correspond to the same material index positions.

To recover names for those positions, the set of ASINs where each bit is set is compared with the set of ASINs whose search string contains a given material token. The tokens are scored using Jaccard similarity,

```text
Jaccard(bit, token) = |ASINs with bit AND token| / |ASINs with bit OR token|
```
and each highest-scoring token is written, according to bit order, to **`material_names.pkl`**. The investigation reports a unique matching token at Jaccard similarity **1.0** for every material bit.

### Custom attributes

The `EMMa_v1.1.fdd` record also has a five-element `custom` probability vector, while the search index `custom_search_file.pkl` has a one-byte value per ASIN, which yields 8 bits positions when unpacked. By correlating the first five unpacked search bits with the probability vector, the association can be retrieved:

| Bit index | Name |
| --- | --- |
| 0 | `sharp` |
| 1 | `transparent` |
| 2 | `deformable` |
| 3 | `articulating` |
| 4 | `electronic` |

This list is saved as **`custom_names.pkl`**.

## 3. Listed Versus Estimated Metadata

The main `EMMa_v1.1.fdd` records include a `"filled in with EM"` field that indicates which attributes were imputed during the construction of the EMMa dataset. Its entries represent, in order, a material mask, then flags for category, price, mass, and ratings.The separate `listed_search_file.pkl` contains a byte per ASIN, yielding 8 unpacked bits, with the first bits indicating which attributes were originally listed and not imputed. 

Comparison with the record-level EM flags supports:

| Bit index | Attribute | Meaning when bit is `1` |
| --- | --- | --- |
| 0 | Category | Originally listed, **not** filled in by EM |
| 1 | Price | Originally listed, **not** filled in by EM |
| 2 | Mass | Originally listed, **not** filled in by EM |

These bits do not encode the ratings EM flag. Importantly, **`1` means listed** in the search byte, whereas **`True` means estimated** in the corresponding per-record EM flag.

## 4. Additional Indexes for the Wrapper

| File | Source | Used for |
| --- | --- | --- |
| `cat_id2name.pkl` | Reverse-engineering notebook | Mapping `(category_id, depth)` to a category name. |
| `material_names.pkl` | Reverse-engineering notebook | Mapping material vector/bit indices to names. |
| `custom_names.pkl` | Reverse-engineering notebook | Names for the five custom attributes. |
| `category_hierarchy.pkl` | `build_category_hierarchy.py` | Category descendants for hierarchy-aware searches. |
| `quantity_search_file.pkl` | `build_quantity_search_file.py` | Numeric quantity filtering, aligned with the canonical ASIN order. |
| `material_hierarchy.json` | Supplied separately | Material taxonomy metadata loaded by `EmmaDataset`. |
| `sorted_idx_priority.npy` | Reverse-engineering notebook | Cached descending indices for `sort_priority.pkl`. The wrapper can compute the ordering if this file is absent. |
| `package_dimensions_present_search_file.npy` | `build_package_dimensions_present_file.py` | Presence mask for `require_present=["package_dimensions"]`. Without it, that filter is unavailable. |

## 5. Preparation Workflow

1. Obtain the EMMa v1.1 dataset and its associated FDD archives and search/index files from the original project. Put them in a single directory (`/path/to/EMMa` in the examples below).
2. Open [`emma_reverseEngineering.ipynb`](../src/reverse_engineering/emma_reverseEngineering.ipynb), set its `ROOT` variable to that directory, and run the derivation cells to create `cat_id2name.pkl`, `material_names.pkl`, `custom_names.pkl`, and (optionally) `sorted_idx_priority.npy`.
3. Create the category hierarchy and quantity index using the included scripts:

   ```bash
   python src/reverse_engineering/build_category_hierarchy.py /path/to/EMMa
   python src/reverse_engineering/build_quantity_search_file.py /path/to/EMMa
   ```

4. Ensure `material_hierarchy.json` is present in the same directory.
5. If package-dimension presence filtering is needed, create its optional mask **after** the required files above are available:

   ```bash
   python src/reverse_engineering/build_package_dimensions_present_file.py /path/to/EMMa
   ```

6. Load the prepared dataset:

   ```python
   from emma import EmmaDataset, EmmaSearcher

   ds = EmmaDataset("/path/to/EMMa")
   searcher = EmmaSearcher(ds)
   matching_asins = searcher.query_asins(
       require_listed=["mass"],
       require_present=["package_dimensions"],  # optional mask required
       limit=10,
   )
   ```

## 6. Limits and Reproducibility Notes

- **Data not redistributed:** This repository does not include the original FDD archives, search-index pickles, or generated metadata artifacts. Consult the original project's distribution terms.
- **Pickle files:** Only load `.pkl` files from sources you trust. Python pickle loading can execute code.

For background on the original dataset, see Standley et al., *[An Extensible Multimodal Multi-task Object Dataset with Materials](https://arxiv.org/abs/2305.14352)* (ICLR 2023).
