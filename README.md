This repository provides utility functions to prepare and use the SINS database in python.
It also provides the source code for sound activity detection (SAD) and sound event classification (SEC) systems.

If you are using code for SAD or SEC please cite the following paper:

```
@InProceedings{ebbers2019weakly,
  author    = {Ebbers, Janek and Drude, Lukas and Brendel, Andreas and Kellermann, Walter and Haeb-Umbach Reinhold},
  title     = {Weakly Supervised Sound Activity Detection and event classification in acoustic sensor networks},
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
$ pip install --user git+https://github.com/fgnt/lazy_dataset.git
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
