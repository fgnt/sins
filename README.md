This repository provides utility functions to prepare and use the SINS database in python.
It also provides the source code for sound activity detection (SAD) and sound event classification (SEC) systems.

If you are using code or results of SAD or SEC please cite the following paper:

```
@InProceedings{ebbers2019weakly,
  author    = {Ebbers, Janek and Drude, Lukas and Brendel, Andreas and Kellermann, Walter and Haeb-Umbach Reinhold},
  title     = {Weakly Supervised Sound Activity Detection and Event Classification in Acoustic Sensor Networks},
  booktitle = {{IEEE} {International} {Workshop} on {Computational} {Advances} in {Multi}-{Sensor} {Adaptive} {Processing} ({CAMSAP})},
  year      = {2019},
  address   = {Guadeloupe, West Indies},
  month     = dec
}
```

If you are using the SINS database please cite the following paper:

```
@InProceedings{dekkers2017sins,
  author    = {Dekkers, G. and Lauwereins, S. and Thoen, B. and Adhana, M. W. and Brouckxon, H. and van Waterschoot, T. and Vanrumste, B. and Verhelst, M. and Karsmakers, P.},
  title     = {The {SINS} database for detection of daily activities in a home environment using an acoustic sensor network},
  booktitle = {Proceedings of the Detection and Classification of Acoustic Scenes and Events 2017 Workshop (DCASE2017)},
  year      = {2017},
  pages     = {32--36},
}
```

## Installation
Install requirements
```bash
$ pip install --user git+https://github.com/fgnt/lazy_dataset.git@ec06c1e8ff4ccb09420d2d641db8f6d9b1099a4f
$ pip install --user git+https://github.com/fgnt/paderbox.git@7b3b4e9d00e07664596108f987292b8c78d846b1
$ pip install --user git+https://github.com/fgnt/padertorch.git@88233a0c33ddcc33a6842a5f8dc6c24df84d9f09
```

Clone the repo
```bash
$ git clone https://github.com/fgnt/sins.git
$ cd sins
```

Install this package
```bash
$ pip install --user -e .
```

Download the database
```bash
$ export SINS_DB_DIR=/desired/path/to/sins
$ python -m sins.database.download
```

Create database description json
```bash
$ python -m sins.database.create_json
```

## Notebooks
See jupyter notebooks in the directory `notebooks/` for usage examples.
